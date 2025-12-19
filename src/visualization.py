"""
Módulo de visualizações para clustering.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings('ignore')


def setup_plot_style(dpi=150, style='default'):
    """Configura estilo dos plots."""
    plt.style.use(style)
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    sns.set_palette("husl")


def compute_embedding(X, method='pca', n_components=2, random_state=42):
    """Computa embedding 2D para visualização."""
    if method == 'pca':
        pca = PCA(n_components=n_components, random_state=random_state)
        X_embed = pca.fit_transform(X)
        return X_embed, pca
    else:
        # UMAP seria opcional, mas não está no requirements
        # Por enquanto, usar PCA como fallback
        pca = PCA(n_components=n_components, random_state=random_state)
        X_embed = pca.fit_transform(X)
        return X_embed, pca


def plot_scatter(X_embed, labels, title, run_name, output_path, method_name=''):
    """Plota scatter 2D colorido por cluster."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        count = np.sum(mask)
        pct = (count / len(labels)) * 100
        ax.scatter(
            X_embed[mask, 0],
            X_embed[mask, 1],
            c=[colors[i]],
            label=f'Cluster {label} (n={count}, {pct:.1f}%)',
            alpha=0.6,
            s=50
        )
    
    ax.set_xlabel('Componente Principal 1', fontsize=12)
    ax.set_ylabel('Componente Principal 2', fontsize=12)
    ax.set_title(f'{title}\n{run_name}', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Nota de rodapé
    fig.text(0.5, 0.02, 
             'Embedding (PCA) é só visualização. Clustering foi feito no espaço pré-processado.',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_elbow(k_values, inertias, output_path, run_name):
    """Plota gráfico de elbow."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax.set_ylabel('Inércia', fontsize=12)
    ax.set_title(f'Elbow Method - K-Means\n{run_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_silhouette_metrics(df_metrics, output_path, run_name):
    """Plota métricas de silhouette, CH e DB."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    k_values = df_metrics['k'].values
    
    # Silhouette
    axes[0].plot(k_values, df_metrics['silhouette'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('k', fontsize=12)
    axes[0].set_ylabel('Silhouette Score', fontsize=12)
    axes[0].set_title('Silhouette Score', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Calinski-Harabasz
    axes[1].plot(k_values, df_metrics['calinski_harabasz'], 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('k', fontsize=12)
    axes[1].set_ylabel('Calinski-Harabasz Score', fontsize=12)
    axes[1].set_title('Calinski-Harabasz Score', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Davies-Bouldin
    axes[2].plot(k_values, df_metrics['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('k', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Score', fontsize=12)
    axes[2].set_title('Davies-Bouldin Score (menor é melhor)', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(f'Métricas de Clustering vs k\n{run_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_dendrogram(linkage_matrix, output_path, run_name, max_points=1000):
    """Plota dendrograma."""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    dendrogram(linkage_matrix, ax=ax, leaf_font_size=10)
    ax.set_xlabel('Amostra', fontsize=12)
    ax.set_ylabel('Distância', fontsize=12)
    ax.set_title(f'Dendrograma - Clustering Hierárquico\n{run_name}', fontsize=14, fontweight='bold')
    
    if max_points < 1000:
        ax.text(0.5, 0.02, f'Amostragem: {max_points} pontos', 
                transform=ax.transAxes, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_confidence_hist(probs, output_path, run_name):
    """Plota histograma de confiança do GMM."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(probs, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Probabilidade Máxima (Confiança)', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    ax.set_title(f'Distribuição de Confiança - GMM\n{run_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Linha vertical na média
    mean_prob = np.mean(probs)
    ax.axvline(mean_prob, color='red', linestyle='--', linewidth=2, 
               label=f'Média: {mean_prob:.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_numeric_profiles(df_original, labels, numeric_cols, output_path, run_name, method_name=''):
    """Plota heatmap de perfis numéricos por cluster."""
    # Calcular médias por cluster
    df_with_labels = df_original[numeric_cols].copy()
    df_with_labels['cluster'] = labels
    
    cluster_means = df_with_labels.groupby('cluster')[numeric_cols].mean()
    
    # Normalizar para visualização (z-score por coluna)
    cluster_means_norm = cluster_means.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(cluster_means) * 0.5)))
    
    sns.heatmap(
        cluster_means_norm.T,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0,
        ax=ax,
        cbar_kws={'label': 'Z-score normalizado'}
    )
    
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Variável Numérica', fontsize=12)
    ax.set_title(f'Perfis Numéricos por Cluster - {method_name}\n{run_name}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_categorical_profiles(df_original, labels, categorical_cols, output_path, run_name, 
                             method_name='', top_n=10):
    """Plota barras com top categorias por cluster."""
    n_clusters = len(np.unique(labels))
    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 6))
    
    if n_clusters == 1:
        axes = [axes]
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_data = df_original[mask]
        
        # Para cada coluna categórica
        for col in categorical_cols:
            if col not in cluster_data.columns:
                continue
            
            value_counts = cluster_data[col].value_counts().head(top_n)
            
            axes[cluster_id].barh(range(len(value_counts)), value_counts.values)
            axes[cluster_id].set_yticks(range(len(value_counts)))
            axes[cluster_id].set_yticklabels(value_counts.index, fontsize=9)
            axes[cluster_id].set_xlabel('Frequência', fontsize=10)
            axes[cluster_id].set_title(f'Cluster {cluster_id}\n(n={np.sum(mask)})', 
                                     fontsize=11, fontweight='bold')
            axes[cluster_id].grid(True, alpha=0.3, axis='x')
    
    fig.suptitle(f'Top {top_n} Categorias por Cluster - {method_name}\n{run_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_missingness(df, output_path):
    """Plota visualização de valores faltantes."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if len(missing) == 0:
        # Criar gráfico vazio indicando que não há faltantes
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Nenhum valor faltante encontrado', 
               ha='center', va='center', fontsize=14)
        ax.set_title('Análise de Valores Faltantes', fontsize=14, fontweight='bold')
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        missing.sort_values(ascending=False).plot(kind='bar', ax=ax)
        ax.set_xlabel('Coluna', fontsize=12)
        ax.set_ylabel('Número de Valores Faltantes', fontsize=12)
        ax.set_title('Análise de Valores Faltantes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_numeric_distributions(df, numeric_cols, output_path):
    """Plota distribuições das variáveis numéricas."""
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(col, fontsize=10)
            axes[i].set_ylabel('Frequência', fontsize=10)
            axes[i].set_title(f'Distribuição: {col}', fontsize=11, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
    
    # Ocultar eixos extras
    for i in range(len(numeric_cols), len(axes)):
        axes[i].axis('off')
    
    fig.suptitle('Distribuições das Variáveis Numéricas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

