"""
Módulo de clustering: k-means, GMM e hierárquico.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')


def compute_elbow_kmeans(X, k_range, random_state=42):
    """Calcula inércia para diferentes valores de k (elbow method)."""
    inertias = []
    k_values = range(k_range[0], k_range[1] + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return list(k_values), inertias


def suggest_k_from_elbow(k_values, inertias, drop_threshold_pct=10):
    """
    Sugere intervalo de k baseado no elbow method.
    Retorna intervalo sugerido ou None se não encontrar.
    """
    if len(inertias) < 3:
        return None
    
    # Calcular redução percentual
    reductions = []
    for i in range(1, len(inertias)):
        reduction = ((inertias[i-1] - inertias[i]) / inertias[i-1]) * 100
        reductions.append(reduction)
    
    # Encontrar onde a redução cai abaixo do threshold
    for i, reduction in enumerate(reductions):
        if reduction < drop_threshold_pct:
            # Sugerir intervalo ao redor deste ponto
            k_at_point = k_values[i+1]
            suggested_min = max(k_values[0], k_at_point - 2)
            suggested_max = min(k_values[-1], k_at_point + 2)
            return [suggested_min, suggested_max]
    
    return None


def compute_clustering_metrics(X, labels):
    """Calcula métricas de qualidade do clustering."""
    if len(np.unique(labels)) < 2:
        return {
            'silhouette': -1,
            'calinski_harabasz': 0,
            'davies_bouldin': float('inf')
        }
    
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = -1
    
    try:
        calinski = calinski_harabasz_score(X, labels)
    except:
        calinski = 0
    
    try:
        davies = davies_bouldin_score(X, labels)
    except:
        davies = float('inf')
    
    return {
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies
    }


def kmeans_clustering(X, k, random_state=42, n_init='auto'):
    """Executa k-means clustering."""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X)
    
    # Distância ao centróide
    distances = np.sqrt(((X - kmeans.cluster_centers_[labels]) ** 2).sum(axis=1))
    
    return {
        'labels': labels,
        'model': kmeans,
        'distances': distances
    }


def gmm_clustering(X, k_range, random_state=42, covariance_type='full', criterion='bic'):
    """
    Executa GMM para diferentes valores de k e escolhe o melhor baseado em BIC/AIC.
    """
    k_values = range(k_range[0], k_range[1] + 1)
    scores = []
    models = []
    
    for k in k_values:
        gmm = GaussianMixture(
            n_components=k,
            random_state=random_state,
            covariance_type=covariance_type,
            n_init=10
        )
        gmm.fit(X)
        models.append(gmm)
        
        if criterion == 'bic':
            scores.append(gmm.bic(X))
        else:  # aic
            scores.append(gmm.aic(X))
    
    # Melhor k (menor BIC/AIC)
    best_idx = np.argmin(scores)
    best_k = k_values[best_idx]
    best_model = models[best_idx]
    
    labels = best_model.predict(X)
    probs = best_model.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    
    return {
        'labels': labels,
        'model': best_model,
        'k_best': best_k,
        'k_scores': {k: score for k, score in zip(k_values, scores)},
        'probabilities': probs,
        'max_probabilities': max_probs
    }


def hierarchical_clustering(X, k, linkage_method='ward'):
    """Executa clustering hierárquico."""
    clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = clustering.fit_predict(X)
    
    return {
        'labels': labels,
        'model': clustering
    }


def compute_linkage_matrix(X, linkage_method='ward', max_points=1000):
    """
    Computa matriz de linkage para dendrograma.
    Se dataset for muito grande, amostra.
    """
    if len(X) > max_points:
        indices = np.random.choice(len(X), max_points, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
        indices = np.arange(len(X))
    
    linkage_matrix = linkage(X_sample, method=linkage_method)
    
    return linkage_matrix, indices


def select_best_k(X, k_range, random_state=42):
    """
    Seleciona melhor k baseado em múltiplas métricas.
    Retorna k sugerido e métricas.
    """
    k_values = range(k_range[0], k_range[1] + 1)
    results = []
    
    for k in k_values:
        labels = kmeans_clustering(X, k, random_state)['labels']
        metrics = compute_clustering_metrics(X, labels)
        metrics['k'] = k
        results.append(metrics)
    
    df_metrics = pd.DataFrame(results)
    
    # Sugerir k baseado em silhouette (maior é melhor)
    best_k = df_metrics.loc[df_metrics['silhouette'].idxmax(), 'k']
    
    return int(best_k), df_metrics

