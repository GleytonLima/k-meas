"""
Script para determinar o número ideal de clusters (k) usando o método do cotovelo
com visualização gráfica detalhada.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def carregar_dados(arquivo='dados_emprestimos.csv'):
    """Carrega o dataset de empréstimos"""
    try:
        df = pd.read_csv(arquivo)
        df['data_contratacao'] = pd.to_datetime(df['data_contratacao'])
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo '{arquivo}' não encontrado!")
        print("Execute primeiro: python gerar_dados.py")
        return None

def preparar_dados(df, features=None):
    """
    Prepara os dados para clustering
    
    Args:
        df: DataFrame com os dados
        features: Lista de features a usar. Se None, usa features numéricas relevantes
    """
    if features is None:
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
    
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X, scaler, features

def calcular_metricas_k(X_scaled, k_range=range(2, 11)):
    """
    Calcula métricas para diferentes valores de k
    
    Args:
        X_scaled: Dados normalizados
        k_range: Range de valores de k para testar
    
    Returns:
        Dicionário com todas as métricas calculadas
    """
    k_values = list(k_range)
    metricas = {
        'k': k_values,
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    print("Calculando métricas para diferentes valores de k...")
    print("-" * 80)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        
        metricas['inertia'].append(inertia)
        metricas['silhouette'].append(silhouette)
        metricas['davies_bouldin'].append(davies_bouldin)
        metricas['calinski_harabasz'].append(calinski_harabasz)
        
        print(f"k={k:2d} | Inércia: {inertia:10.2f} | "
              f"Silhouette: {silhouette:.4f} | "
              f"Davies-Bouldin: {davies_bouldin:.4f} | "
              f"Calinski-Harabasz: {calinski_harabasz:.2f}")
    
    return metricas

def calcular_derivada_segunda(inertias):
    """
    Calcula a derivada segunda (diferença das diferenças) para encontrar o "cotovelo"
    """
    # Primeira derivada (diferenças)
    primeira_derivada = np.diff(inertias)
    
    # Segunda derivada (diferença das diferenças)
    segunda_derivada = np.diff(primeira_derivada)
    
    return primeira_derivada, segunda_derivada

def encontrar_k_otimo(metricas):
    """
    Encontra o k ótimo usando múltiplas métricas
    """
    k_values = metricas['k']
    
    # Melhor k por Silhouette (maior é melhor)
    melhor_k_silhouette = k_values[np.argmax(metricas['silhouette'])]
    
    # Melhor k por Davies-Bouldin (menor é melhor)
    melhor_k_db = k_values[np.argmin(metricas['davies_bouldin'])]
    
    # Melhor k por Calinski-Harabasz (maior é melhor)
    melhor_k_ch = k_values[np.argmax(metricas['calinski_harabasz'])]
    
    # Método do cotovelo (usando derivada segunda)
    primeira_derivada, segunda_derivada = calcular_derivada_segunda(metricas['inertia'])
    # O cotovelo está onde a segunda derivada é máxima (mudança mais acentuada)
    if len(segunda_derivada) > 0:
        idx_cotovelo = np.argmax(segunda_derivada)
        melhor_k_cotovelo = k_values[idx_cotovelo + 1]  # +1 porque perdemos um elemento no diff
    else:
        melhor_k_cotovelo = k_values[0]
    
    return {
        'silhouette': melhor_k_silhouette,
        'davies_bouldin': melhor_k_db,
        'calinski_harabasz': melhor_k_ch,
        'cotovelo': melhor_k_cotovelo
    }

def visualizar_analise_k(metricas, k_otimos, arquivo_saida='analise_k_ideal.png', pasta_imagens='imagens'):
    """
    Cria visualizações completas para análise do k ideal
    
    Args:
        metricas: Dicionário com métricas calculadas
        k_otimos: Dicionário com k ótimos por métrica
        arquivo_saida: Nome do arquivo para salvar o gráfico
        pasta_imagens: Pasta para salvar as imagens
    """
    import os
    os.makedirs(pasta_imagens, exist_ok=True)
    
    k_values = metricas['k']
    
    # Se arquivo_saida não tem caminho completo, adicionar pasta_imagens
    if os.path.dirname(arquivo_saida) == '':
        arquivo_saida = os.path.join(pasta_imagens, arquivo_saida)
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Método do Cotovelo (Inércia)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(k_values, metricas['inertia'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inércia (Within-cluster Sum of Squares)', fontsize=12, fontweight='bold')
    ax1.set_title('Método do Cotovelo - Inércia', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=k_otimos['cotovelo'], color='r', linestyle='--', 
                linewidth=2, label=f"k ótimo (cotovelo): {k_otimos['cotovelo']}")
    ax1.legend()
    
    # 2. Silhouette Score
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(k_values, metricas['silhouette'], 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_title('Silhouette Score (maior é melhor)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=k_otimos['silhouette'], color='r', linestyle='--', 
                linewidth=2, label=f"k ótimo: {k_otimos['silhouette']}")
    ax2.legend()
    
    # 3. Davies-Bouldin Index
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(k_values, metricas['davies_bouldin'], 'mo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
    ax3.set_title('Davies-Bouldin Index (menor é melhor)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=k_otimos['davies_bouldin'], color='r', linestyle='--', 
                linewidth=2, label=f"k ótimo: {k_otimos['davies_bouldin']}")
    ax3.legend()
    
    # 4. Calinski-Harabasz Score
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(k_values, metricas['calinski_harabasz'], 'co-', linewidth=2, markersize=8)
    ax4.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
    ax4.set_title('Calinski-Harabasz Score (maior é melhor)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=k_otimos['calinski_harabasz'], color='r', linestyle='--', 
                linewidth=2, label=f"k ótimo: {k_otimos['calinski_harabasz']}")
    ax4.legend()
    
    # 5. Método do Cotovelo com Derivadas
    ax5 = fig.add_subplot(gs[2, :])
    primeira_derivada, segunda_derivada = calcular_derivada_segunda(metricas['inertia'])
    
    # Plotar inércia
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(k_values, metricas['inertia'], 'bo-', linewidth=2, 
                     markersize=8, label='Inércia')
    ax5.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Inércia', fontsize=12, fontweight='bold', color='b')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5.grid(True, alpha=0.3)
    
    # Plotar segunda derivada
    k_derivada = k_values[2:]  # Perdemos 2 elementos com diff duplo
    line2 = ax5_twin.plot(k_derivada, segunda_derivada, 'r^-', linewidth=2, 
                          markersize=8, label='Segunda Derivada')
    ax5_twin.set_ylabel('Segunda Derivada (Mudança na Taxa de Decréscimo)', 
                        fontsize=12, fontweight='bold', color='r')
    ax5_twin.tick_params(axis='y', labelcolor='r')
    
    # Marcar o cotovelo
    if len(segunda_derivada) > 0:
        idx_max = np.argmax(segunda_derivada)
        k_cotovelo = k_derivada[idx_max]
        ax5.axvline(x=k_cotovelo, color='g', linestyle='--', linewidth=2, 
                   label=f'Cotovelo: k={k_cotovelo}')
    
    ax5.set_title('Método do Cotovelo Detalhado - Inércia e Segunda Derivada', 
                  fontsize=14, fontweight='bold')
    
    # Combinar legendas
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper right')
    
    plt.suptitle('Análise Completa para Determinação do k Ideal', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Gráfico completo salvo em '{arquivo_saida}'")
    
    # Criar tabela resumo
    criar_tabela_resumo(metricas, k_otimos)

def criar_tabela_resumo(metricas, k_otimos):
    """
    Cria uma tabela resumo com as métricas e k ótimos
    """
    df_resumo = pd.DataFrame({
        'k': metricas['k'],
        'Inércia': [f"{x:.2f}" for x in metricas['inertia']],
        'Silhouette': [f"{x:.4f}" for x in metricas['silhouette']],
        'Davies-Bouldin': [f"{x:.4f}" for x in metricas['davies_bouldin']],
        'Calinski-Harabasz': [f"{x:.2f}" for x in metricas['calinski_harabasz']]
    })
    
    print("\n" + "="*80)
    print("TABELA RESUMO DE MÉTRICAS")
    print("="*80)
    print(df_resumo.to_string(index=False))
    
    print("\n" + "="*80)
    print("K ÓTIMO POR MÉTRICA")
    print("="*80)
    print(f"Método do Cotovelo:        k = {k_otimos['cotovelo']}")
    print(f"Silhouette Score:          k = {k_otimos['silhouette']} "
          f"(score: {metricas['silhouette'][k_otimos['silhouette']-2]:.4f})")
    print(f"Davies-Bouldin Index:      k = {k_otimos['davies_bouldin']} "
          f"(score: {metricas['davies_bouldin'][k_otimos['davies_bouldin']-2]:.4f})")
    print(f"Calinski-Harabasz Score:   k = {k_otimos['calinski_harabasz']} "
          f"(score: {metricas['calinski_harabasz'][k_otimos['calinski_harabasz']-2]:.2f})")
    
    # Recomendação final
    print("\n" + "="*80)
    print("RECOMENDAÇÃO")
    print("="*80)
    
    # Contar quantas métricas sugerem cada k
    k_sugeridos = list(k_otimos.values())
    k_mais_comum = max(set(k_sugeridos), key=k_sugeridos.count)
    
    print(f"K mais recomendado: {k_mais_comum} (sugerido por {k_sugeridos.count(k_mais_comum)} métricas)")
    print(f"\nObservação: O método do cotovelo é útil para visualização, mas recomenda-se")
    print(f"também considerar o Silhouette Score para uma análise mais completa.")

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
        return config.get('analisar_k_ideal', {})
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[ERRO] Erro ao carregar config: {e}")
        return None

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Análise do K Ideal para K-Means - Método do Cotovelo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  
  # Usar range padrão (k de 2 a 10)
  python analisar_k_ideal.py
  
  # Especificar range customizado
  python analisar_k_ideal.py --k-min 3 --k-max 15
  
  # Usar arquivo de entrada diferente
  python analisar_k_ideal.py --arquivo dados_customizados.csv --k-min 2 --k-max 8
  
  # Usar arquivo de configuração
  python analisar_k_ideal.py --config config.json
        """
    )
    
    parser.add_argument(
        '--k-min',
        type=int,
        default=None,
        help='Valor mínimo de k para testar (padrão: 2 ou do config.json)'
    )
    
    parser.add_argument(
        '--k-max',
        type=int,
        default=None,
        help='Valor máximo de k para testar (padrão: 10 ou do config.json)'
    )
    
    parser.add_argument(
        '--arquivo',
        type=str,
        default=None,
        help='Arquivo CSV com os dados de empréstimos (padrão: dados_emprestimos.csv ou do config.json)'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        default=None,
        help='Lista de features a usar no clustering (separadas por espaço). Se não especificado, usa features padrão ou do config.json.'
    )
    
    parser.add_argument(
        '--saida',
        type=str,
        default=None,
        help='Nome do arquivo de saída para o gráfico (padrão: analise_k_ideal.png ou do config.json)'
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
        if args.k_min is None and 'k_min' in config:
            args.k_min = config['k_min']
        if args.k_max is None and 'k_max' in config:
            args.k_max = config['k_max']
        if args.arquivo is None and 'arquivo_entrada' in config:
            args.arquivo = config['arquivo_entrada']
        if args.saida is None and 'arquivo_saida' in config:
            args.saida = config['arquivo_saida']
        if args.features is None and 'features' in config:
            args.features = config['features']
        if getattr(args, 'pasta_imagens', 'imagens') == 'imagens' and 'pasta_imagens' in config:
            args.pasta_imagens = config['pasta_imagens']
    
    # Definir valores padrão se ainda não foram definidos
    if args.k_min is None:
        args.k_min = 2
    if args.k_max is None:
        args.k_max = 10
    if args.arquivo is None:
        args.arquivo = 'dados_emprestimos.csv'
    if args.saida is None:
        args.saida = 'analise_k_ideal.png'
    
    # Obter pasta_imagens
    pasta_imagens = getattr(args, 'pasta_imagens', 'imagens')
    
    # Validar range
    if args.k_min >= args.k_max:
        print("❌ Erro: k-min deve ser menor que k-max!")
        return
    
    if args.k_min < 2:
        print("[AVISO] k-min deve ser pelo menos 2. Ajustando para 2.")
        args.k_min = 2
    
    print("="*80)
    print("ANÁLISE DO K IDEAL PARA K-MEANS - MÉTODO DO COTOVELO")
    print("="*80)
    
    # Carregar dados
    print(f"\n1. Carregando dados de '{args.arquivo}'...")
    df = carregar_dados(args.arquivo)
    if df is None:
        return
    
    print(f"   [OK] Dataset carregado: {len(df)} contratos")
    
    # Preparar dados
    print("\n2. Preparando dados...")
    X_scaled, X, scaler, features = preparar_dados(df, features=args.features)
    print(f"   [OK] Features selecionadas: {len(features)}")
    
    # Calcular métricas
    print(f"\n3. Calculando métricas para k de {args.k_min} a {args.k_max}...")
    metricas = calcular_metricas_k(X_scaled, k_range=range(args.k_min, args.k_max + 1))
    
    # Encontrar k ótimos
    print("\n4. Analisando k ótimo por métrica...")
    k_otimos = encontrar_k_otimo(metricas)
    
    # Visualizar
    print("\n5. Gerando visualizações...")
    # Atualizar função para aceitar nome de arquivo customizado
    visualizar_analise_k(metricas, k_otimos, arquivo_saida=args.saida, pasta_imagens=pasta_imagens)
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA!")
    print("="*80)
    print("\nPróximos passos:")
    print(f"1. Analise os gráficos gerados em '{args.saida}'")
    print("2. Escolha o k baseado nas métricas e no método do cotovelo")
    print("3. Execute 'python kmeans_emprestimos.py --k <valor>' para aplicar o clustering")

if __name__ == "__main__":
    main()

