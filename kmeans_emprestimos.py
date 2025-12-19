"""
Implementação de K-Means para análise de clusters de contratos de empréstimos
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def carregar_dados(arquivo='dados_emprestimos.csv'):
    """Carrega o dataset de empréstimos"""
    df = pd.read_csv(arquivo)
    df['data_contratacao'] = pd.to_datetime(df['data_contratacao'])
    return df

def preparar_dados(df, features=None):
    """
    Prepara os dados para clustering
    
    Args:
        df: DataFrame com os dados
        features: Lista de features a usar. Se None, usa features numéricas relevantes
    """
    if features is None:
        # Features numéricas relevantes para clustering
        features = [
            'valor_emprestimo',
            'quantidade_parcelas',
            'taxa_juros_anual',
            'score_cliente',
            'idade_cliente',
            'renda_mensal',
            'valor_parcela',
            'parcela_sobre_renda'
        ]
    
    # Selecionar apenas as features numéricas
    X = df[features].copy()
    
    # Normalizar os dados (importante para k-means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X, scaler, features

def encontrar_melhor_k(X_scaled, k_range=range(2, 11), pasta_imagens='imagens'):
    """
    Encontra o melhor número de clusters usando método do cotovelo e silhouette score
    
    Args:
        X_scaled: Dados normalizados
        k_range: Range de valores de k para testar
        pasta_imagens: Pasta para salvar as imagens
    """
    import os
    os.makedirs(pasta_imagens, exist_ok=True)
    
    inertias = []
    silhouette_scores = []
    k_values = list(k_range)
    
    print("Testando diferentes valores de k...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        print(f"  k={k}: Inertia={inertias[-1]:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
    
    # Plotar gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Método do cotovelo
    ax1.plot(k_values, inertias, 'bo-')
    ax1.set_xlabel('Número de Clusters (k)')
    ax1.set_ylabel('Inércia (Within-cluster Sum of Squares)')
    ax1.set_title('Método do Cotovelo')
    ax1.grid(True)
    
    # Silhouette Score
    ax2.plot(k_values, silhouette_scores, 'ro-')
    ax2.set_xlabel('Número de Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score por k')
    ax2.grid(True)
    
    plt.tight_layout()
    caminho_imagem = os.path.join(pasta_imagens, 'analise_melhor_k.png')
    plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')
    print(f"\nGráfico salvo em '{caminho_imagem}'")
    
    # Encontrar melhor k (maior silhouette score)
    melhor_k = k_values[np.argmax(silhouette_scores)]
    print(f"\nMelhor k baseado em Silhouette Score: {melhor_k}")
    
    return melhor_k, inertias, silhouette_scores

def aplicar_kmeans(X_scaled, n_clusters=4, random_state=42):
    """Aplica K-Means aos dados"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels

def analisar_clusters(df, labels, features):
    """
    Analisa e descreve os clusters encontrados
    """
    df_clusters = df.copy()
    df_clusters['cluster'] = labels
    
    print("\n" + "="*80)
    print("ANÁLISE DOS CLUSTERS")
    print("="*80)
    
    # Estatísticas por cluster
    print("\n=== Tamanho dos Clusters ===")
    print(df_clusters['cluster'].value_counts().sort_index())
    
    print("\n=== Estatísticas por Cluster ===")
    stats_features = features + ['parcela_sobre_renda']
    cluster_stats = df_clusters.groupby('cluster')[stats_features].agg(['mean', 'std', 'min', 'max'])
    print(cluster_stats)
    
    print("\n=== Distribuição de Produtos por Cluster ===")
    produto_cluster = pd.crosstab(df_clusters['cluster'], df_clusters['nome_produto'], normalize='index') * 100
    print(produto_cluster.round(2))
    
    # Perfil de cada cluster
    print("\n=== PERFIL DOS CLUSTERS ===")
    for cluster_id in sorted(df_clusters['cluster'].unique()):
        cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
        print(f"\n--- CLUSTER {cluster_id} ({len(cluster_data)} contratos) ---")
        print(f"Valor médio do empréstimo: R$ {cluster_data['valor_emprestimo'].mean():,.2f}")
        print(f"Parcelas médias: {cluster_data['quantidade_parcelas'].mean():.1f}")
        print(f"Taxa de juros média: {cluster_data['taxa_juros_anual'].mean():.2f}% a.a.")
        print(f"Score médio do cliente: {cluster_data['score_cliente'].mean():.0f}")
        print(f"Renda mensal média: R$ {cluster_data['renda_mensal'].mean():,.2f}")
        print(f"Parcela sobre renda média: {cluster_data['parcela_sobre_renda'].mean():.2f}%")
        print(f"Produto mais comum: {cluster_data['nome_produto'].mode()[0]}")
    
    return df_clusters

def visualizar_clusters(df_clusters, features, labels, kmeans, pares_visualizacao=None, pasta_imagens='imagens'):
    """
    Cria visualizações dos clusters
    
    Args:
        df_clusters: DataFrame com os dados e clusters
        features: Lista de features disponíveis
        labels: Labels dos clusters
        kmeans: Modelo K-Means treinado
        pares_visualizacao: Lista de tuplas (feat1, feat2) para visualizar.
                           Se None, usa pares padrão.
        pasta_imagens: Pasta para salvar as imagens
    """
    import os
    os.makedirs(pasta_imagens, exist_ok=True)
    # Selecionar pares de features para visualização
    if pares_visualizacao is None:
        # Pares padrão
        pares_visualizacao = [
            ('valor_emprestimo', 'quantidade_parcelas'),
            ('score_cliente', 'taxa_juros_anual'),
            ('renda_mensal', 'valor_parcela'),
            ('valor_emprestimo', 'parcela_sobre_renda')
        ]
    
    # Validar pares
    pares_validos = []
    for feat1, feat2 in pares_visualizacao:
        if feat1 in df_clusters.columns and feat2 in df_clusters.columns:
            pares_validos.append((feat1, feat2))
        else:
            print(f"[AVISO] Par ({feat1}, {feat2}) ignorado - features não encontradas no dataset")
    
    if not pares_validos:
        print("❌ Erro: Nenhum par válido para visualização!")
        return
    
    # Calcular layout do grid
    n_pares = len(pares_validos)
    n_cols = 2
    n_rows = (n_pares + 1) // 2  # Arredondar para cima
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
    if n_pares == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (feat1, feat2) in enumerate(pares_validos):
        ax = axes[idx]
        scatter = ax.scatter(df_clusters[feat1], df_clusters[feat2], 
                           c=labels, cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel(feat1.replace('_', ' ').title())
        ax.set_ylabel(feat2.replace('_', ' ').title())
        ax.set_title(f'Clusters: {feat1.replace("_", " ")} vs {feat2.replace("_", " ")}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # Ocultar subplots não utilizados
    for idx in range(n_pares, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    caminho_imagem = os.path.join(pasta_imagens, 'visualizacao_clusters.png')
    plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Visualização dos clusters salva em '{caminho_imagem}' ({n_pares} pares)")
    
    # Gráfico de distribuição por produto e cluster
    fig, ax = plt.subplots(figsize=(12, 6))
    produto_cluster = pd.crosstab(df_clusters['cluster'], df_clusters['nome_produto'])
    produto_cluster.plot(kind='bar', ax=ax, stacked=True)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Número de Contratos')
    ax.set_title('Distribuição de Produtos por Cluster')
    ax.legend(title='Produto', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    caminho_imagem2 = os.path.join(pasta_imagens, 'distribuicao_produtos_clusters.png')
    plt.savefig(caminho_imagem2, dpi=300, bbox_inches='tight')
    print(f"Distribuição de produtos por cluster salva em '{caminho_imagem2}'")

def gerar_insights(df_clusters):
    """
    Gera insights sobre os clusters
    """
    print("\n" + "="*80)
    print("INSIGHTS E RECOMENDAÇÕES")
    print("="*80)
    
    for cluster_id in sorted(df_clusters['cluster'].unique()):
        cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
        
        print(f"\n### CLUSTER {cluster_id} ###")
        
        # Identificar características principais
        valor_medio = cluster_data['valor_emprestimo'].mean()
        score_medio = cluster_data['score_cliente'].mean()
        taxa_media = cluster_data['taxa_juros_anual'].mean()
        parcela_renda = cluster_data['parcela_sobre_renda'].mean()
        
        # Classificar o cluster
        if valor_medio > df_clusters['valor_emprestimo'].quantile(0.75):
            tamanho = "ALTO VALOR"
        elif valor_medio < df_clusters['valor_emprestimo'].quantile(0.25):
            tamanho = "BAIXO VALOR"
        else:
            tamanho = "VALOR MÉDIO"
        
        if score_medio > 700:
            risco = "BAIXO RISCO"
        elif score_medio < 600:
            risco = "ALTO RISCO"
        else:
            risco = "RISCO MÉDIO"
        
        print(f"Perfil: {tamanho} - {risco}")
        print(f"Características principais:")
        print(f"  - Valor médio: R$ {valor_medio:,.2f}")
        print(f"  - Score médio: {score_medio:.0f}")
        print(f"  - Taxa média: {taxa_media:.2f}% a.a.")
        print(f"  - Parcela representa {parcela_renda:.1f}% da renda")
        
        # Recomendações
        print(f"\nRecomendações:")
        if parcela_renda > 30:
            print(f"  [ATENCAO] Parcela muito alta em relação à renda ({parcela_renda:.1f}%)")
        if score_medio < 600:
            print(f"  [ATENCAO] Score baixo indica maior risco de inadimplência")
        if taxa_media > 20:
            print(f"  [OPORTUNIDADE] Taxa alta pode ser negociada para clientes de baixo risco")
        
        produto_principal = cluster_data['nome_produto'].mode()[0]
        print(f"  [INFO] Produto principal: {produto_principal}")

def carregar_config(arquivo_config='config.json'):
    """
    Carrega configuração do arquivo JSON
    
    Args:
        arquivo_config: Caminho para o arquivo de configuração
    
    Returns:
        Dicionário com configurações ou None se arquivo não existir
    """
    if not os.path.exists(arquivo_config):
        return None
    
    try:
        with open(arquivo_config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('kmeans_emprestimos', {})
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[ERRO] Erro ao carregar config: {e}")
        return None

def parse_pares_visualizacao(pares_str):
    """
    Converte string de pares em lista de tuplas
    Formato esperado: "feat1,feat2;feat3,feat4" ou "feat1,feat2 feat3,feat4"
    Também aceita lista de listas do JSON
    """
    if not pares_str:
        return None
    
    # Se já é uma lista (do JSON), retornar como está
    if isinstance(pares_str, list):
        return [tuple(par) if isinstance(par, list) else par for par in pares_str]
    
    # Se é string, fazer parse
    pares = []
    if ';' in pares_str:
        partes = pares_str.split(';')
    else:
        partes = pares_str.split()
    
    for parte in partes:
        parte = parte.strip()
        if ',' in parte:
            feat1, feat2 = parte.split(',')
            pares.append((feat1.strip(), feat2.strip()))
    
    return pares if pares else None

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Análise de Clusters de Contratos de Empréstimos usando K-Means',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  
  # Usar k automático e pares padrão
  python kmeans_emprestimos.py
  
  # Especificar k=3
  python kmeans_emprestimos.py --k 3
  
  # Especificar pares de visualização
  python kmeans_emprestimos.py --pares "valor_emprestimo,quantidade_parcelas;score_cliente,taxa_juros_anual"
  
  # Especificar k e pares
  python kmeans_emprestimos.py --k 4 --pares "valor_emprestimo,quantidade_parcelas renda_mensal,valor_parcela"
  
  # Usar arquivo de entrada diferente
  python kmeans_emprestimos.py --arquivo dados_customizados.csv --k 5
        """
    )
    
    parser.add_argument(
        '--k', 
        type=int, 
        default=None,
        help='Número de clusters (k). Se não especificado, será determinado automaticamente.'
    )
    
    parser.add_argument(
        '--pares',
        type=str,
        default=None,
        help='Pares de features para visualização. Formato: "feat1,feat2;feat3,feat4" ou "feat1,feat2 feat3,feat4"'
    )
    
    parser.add_argument(
        '--arquivo',
        type=str,
        default='dados_emprestimos.csv',
        help='Arquivo CSV com os dados de empréstimos (padrão: dados_emprestimos.csv)'
    )
    
    parser.add_argument(
        '--saida',
        type=str,
        default='resultados_clusters.csv',
        help='Arquivo CSV de saída com os resultados (padrão: resultados_clusters.csv)'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        default=None,
        help='Lista de features a usar no clustering (separadas por espaço). Se não especificado, usa features padrão.'
    )
    
    parser.add_argument(
        '--k-range',
        type=str,
        default=None,
        help='Range de k para busca automática no formato min-max (padrão: 2-10)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Arquivo de configuração JSON (padrão: config.json). Use --config "" para desabilitar.'
    )
    
    parser.add_argument(
        '--pasta_imagens',
        type=str,
        default='imagens',
        help='Pasta para salvar as imagens geradas (padrão: imagens)'
    )
    
    args = parser.parse_args()
    
    # Carregar configuração do JSON (se não foi desabilitado)
    config = None
    if args.config:
        config = carregar_config(args.config)
        if config:
            print(f"[OK] Configuração carregada de '{args.config}'")
    
    # Mesclar config com args (args têm prioridade)
    if config:
        # k
        if args.k is None and 'k' in config:
            args.k = config['k']
        
        # k_range
        if args.k_range is None and 'k_range' in config:
            k_range = config['k_range']
            if isinstance(k_range, dict):
                args.k_range = f"{k_range.get('min', 2)}-{k_range.get('max', 10)}"
            elif isinstance(k_range, str):
                args.k_range = k_range
        
        # pares
        if args.pares is None and 'pares_visualizacao' in config:
            args.pares = config['pares_visualizacao']
        
        # arquivo
        if args.arquivo == 'dados_emprestimos.csv' and 'arquivo_entrada' in config:
            args.arquivo = config['arquivo_entrada']
        
        # saida
        if args.saida == 'resultados_clusters.csv' and 'arquivo_saida' in config:
            args.saida = config['arquivo_saida']
        
        # features
        if args.features is None and 'features' in config:
            args.features = config['features']
        
        # pasta_imagens
        if args.pasta_imagens == 'imagens' and 'pasta_imagens' in config:
            args.pasta_imagens = config['pasta_imagens']
    
    # Definir k_range padrão se ainda não foi definido
    if args.k_range is None:
        args.k_range = '2-10'
    
    # Obter pasta_imagens
    pasta_imagens = getattr(args, 'pasta_imagens', 'imagens')
    
    print("="*80)
    print("ANÁLISE DE CLUSTERS DE CONTRATOS DE EMPRÉSTIMOS USANDO K-MEANS")
    print("="*80)
    
    # Carregar dados
    print(f"\n1. Carregando dados de '{args.arquivo}'...")
    df = carregar_dados(args.arquivo)
    print(f"   [OK] Dataset carregado: {len(df)} contratos")
    
    # Preparar dados
    print("\n2. Preparando dados para clustering...")
    X_scaled, X, scaler, features = preparar_dados(df, features=args.features)
    print(f"   [OK] Features selecionadas: {', '.join(features)}")
    
    # Determinar k
    if args.k is not None:
        melhor_k = args.k
        print(f"\n3. Usando k={melhor_k} (especificado pelo usuário)")
    else:
        print("\n3. Encontrando melhor número de clusters...")
        # Parse k-range
        k_min, k_max = map(int, args.k_range.split('-'))
        melhor_k, inertias, silhouette_scores = encontrar_melhor_k(X_scaled, k_range=range(k_min, k_max+1), pasta_imagens=pasta_imagens)
    
    # Aplicar k-means
    print(f"\n4. Aplicando K-Means com k={melhor_k}...")
    kmeans, labels = aplicar_kmeans(X_scaled, n_clusters=melhor_k)
    
    # Calcular métricas
    silhouette = silhouette_score(X_scaled, labels)
    print(f"   [OK] Silhouette Score: {silhouette:.3f}")
    
    # Analisar clusters
    print("\n5. Analisando clusters...")
    df_clusters = analisar_clusters(df, labels, features)
    
    # Parse pares de visualização
    pares_visualizacao = parse_pares_visualizacao(args.pares)
    if pares_visualizacao:
        print(f"\n   Usando {len(pares_visualizacao)} par(es) customizado(s) para visualização")
    else:
        print(f"\n   Usando pares padrão para visualização")
    
    # Visualizar
    print("\n6. Gerando visualizações...")
    visualizar_clusters(df_clusters, features, labels, kmeans, pares_visualizacao=pares_visualizacao, pasta_imagens=pasta_imagens)
    
    # Gerar insights
    print("\n7. Gerando insights...")
    gerar_insights(df_clusters)
    
    # Salvar resultados
    df_clusters.to_csv(args.saida, index=False)
    print(f"\n8. Resultados salvos em '{args.saida}'")
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA!")
    print("="*80)

if __name__ == "__main__":
    main()

