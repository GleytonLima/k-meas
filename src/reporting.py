"""
Módulo de geração de relatórios e insights.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
import yaml


def generate_executive_summary(config, results, output_path):
    """Gera resumo executivo em Markdown."""
    run_name = config['run']['name']
    run_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    n_samples = results['n_samples']
    k_best = results.get('k_best', 'N/A')
    k_gmm_best = results.get('k_gmm_best', 'N/A')
    
    content = f"""# Resumo Executivo - Análise de Clustering

**Run:** {run_name}  
**Data:** {run_date}  
**Tamanho do Dataset:** {n_samples} amostras

## Seleção de K

- **K sugerido (K-Means):** {k_best}
- **K sugerido (GMM):** {k_gmm_best}

## Principais Achados

"""
    
    # Adicionar informações sobre clusters
    if 'gmm_results' in results:
        gmm_labels = results['gmm_results']['labels']
        cluster_sizes = pd.Series(gmm_labels).value_counts().sort_index()
        
        # Clusters grandes
        large_clusters = cluster_sizes[cluster_sizes >= n_samples * 0.1]
        if len(large_clusters) > 0:
            content += "### Clusters Representativos (≥10% dos dados)\n\n"
            for cluster_id, size in large_clusters.items():
                pct = (size / n_samples) * 100
                content += f"- **Cluster {cluster_id}:** {size} amostras ({pct:.1f}%)\n"
            content += "\n"
        
        # Clusters pequenos
        small_clusters = cluster_sizes[cluster_sizes < n_samples * 0.03]
        if len(small_clusters) > 0:
            content += "### Clusters Pequenos (<3% dos dados)\n\n"
            for cluster_id, size in small_clusters.items():
                pct = (size / n_samples) * 100
                content += f"- **Cluster {cluster_id}:** {size} amostras ({pct:.1f}%)\n"
            content += "\n"
        
        # Confiança do GMM
        if 'max_probabilities' in results['gmm_results']:
            mean_conf = np.mean(results['gmm_results']['max_probabilities'])
            low_conf = np.sum(results['gmm_results']['max_probabilities'] < 0.5)
            if low_conf > 0:
                content += f"### Clusters com Baixa Confiança\n\n"
                content += f"- {low_conf} amostras ({low_conf/n_samples*100:.1f}%) com confiança < 0.5\n"
                content += f"- Confiança média: {mean_conf:.3f}\n\n"
    
    content += """## Recomendações

1. **Análise de Variáveis Diferenciadoras:**
   - Investigar quais variáveis numéricas e categóricas mais diferenciam os clusters
   - Verificar se há padrões de negócio claros por cluster

2. **Validação de Negócio:**
   - Validar se os clusters fazem sentido do ponto de vista de negócio
   - Identificar oportunidades de segmentação de produtos/marketing

3. **Próximos Passos:**
   - Análise mais profunda dos clusters pequenos (nichos)
   - Desenvolvimento de perfis de cliente por cluster
   - Teste de estratégias diferenciadas por cluster

---
*Relatório gerado automaticamente pelo pipeline de clustering*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def generate_cluster_cards(df_original, results, config, output_path):
    """Gera cards descritivos para cada cluster."""
    run_name = config['run']['name']
    
    if 'gmm_results' not in results:
        return
    
    gmm_labels = results['gmm_results']['labels']
    n_samples = len(gmm_labels)
    numeric_cols = config['columns']['numeric']
    categorical_cols = config['columns']['categorical']
    
    # Filtrar colunas que existem
    numeric_cols = [c for c in numeric_cols if c in df_original.columns]
    categorical_cols = [c for c in categorical_cols if c in df_original.columns]
    
    content = f"""# Cluster Cards - Análise Detalhada

**Run:** {run_name}  
**Método:** GMM

---

"""
    
    unique_clusters = np.unique(gmm_labels)
    
    for cluster_id in sorted(unique_clusters):
        mask = gmm_labels == cluster_id
        cluster_data = df_original[mask]
        cluster_size = np.sum(mask)
        cluster_pct = (cluster_size / n_samples) * 100
        
        content += f"""## Cluster {cluster_id}

**Tamanho:** {cluster_size} amostras ({cluster_pct:.1f}%)

### Características Numéricas (Médias)

"""
        
        # Médias numéricas
        for col in numeric_cols[:5]:  # Top 5
            mean_val = cluster_data[col].mean()
            content += f"- **{col}:** {mean_val:.2f}\n"
        
        content += "\n### Top Categorias\n\n"
        
        # Top categorias
        for col in categorical_cols:
            if col in cluster_data.columns:
                top_cats = cluster_data[col].value_counts().head(3)
                content += f"**{col}:**\n"
                for cat, count in top_cats.items():
                    pct = (count / cluster_size) * 100
                    content += f"  - {cat}: {count} ({pct:.1f}%)\n"
        
        # Exemplos de IDs
        id_col = config['input']['id_column']
        if id_col in df_original.columns:
            example_ids = cluster_data[id_col].head(5).tolist()
            content += f"\n### Exemplos de IDs\n\n"
            content += f"{', '.join(map(str, example_ids))}\n"
        
        # Hipótese de negócio (template genérico)
        content += f"\n### Hipótese de Negócio\n\n"
        if cluster_pct > 20:
            content += "Cluster representativo da base. Características típicas do perfil principal.\n"
        elif cluster_pct < 3:
            content += "Cluster de nicho. Requer análise específica para entender padrões únicos.\n"
        else:
            content += "Cluster intermediário. Padrões distintos mas não extremos.\n"
        
        content += "\n---\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def generate_cluster_profiles_table(df_original, results, config, output_path):
    """Gera tabela CSV com perfis de cluster."""
    if 'gmm_results' not in results:
        return
    
    gmm_labels = results['gmm_results']['labels']
    numeric_cols = config['columns']['numeric']
    numeric_cols = [c for c in numeric_cols if c in df_original.columns]
    
    df_with_labels = df_original[numeric_cols].copy()
    df_with_labels['cluster'] = gmm_labels
    
    # Calcular estatísticas por cluster
    cluster_stats = df_with_labels.groupby('cluster')[numeric_cols].agg(['mean', 'std', 'count'])
    
    # Flatten multi-index columns
    cluster_stats.columns = [f'{col}_{stat}' for col, stat in cluster_stats.columns]
    cluster_stats = cluster_stats.reset_index()
    
    cluster_stats.to_csv(output_path, index=False)


def save_metadata(config, results, output_path):
    """Salva metadados da execução."""
    metadata = {
        'run_name': config['run']['name'],
        'timestamp': datetime.now().isoformat(),
        'n_samples': results.get('n_samples', 0),
        'n_features': results.get('n_features', 0),
        'k_best': results.get('k_best', None),
        'k_gmm_best': results.get('k_gmm_best', None),
        'config': config
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_resolved_config(config, output_path):
    """Salva configuração resolvida (após validações)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

