"""
Entrypoint principal do pipeline de clustering.
"""
import argparse
import sys
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Adicionar src ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_pipeline, load_data
from clustering import (
    compute_elbow_kmeans, suggest_k_from_elbow, select_best_k,
    kmeans_clustering, gmm_clustering, hierarchical_clustering,
    compute_linkage_matrix
)
from visualization import (
    setup_plot_style, compute_embedding, plot_scatter, plot_elbow,
    plot_silhouette_metrics, plot_dendrogram, plot_confidence_hist,
    plot_numeric_profiles, plot_categorical_profiles,
    plot_missingness, plot_numeric_distributions
)
from reporting import (
    generate_executive_summary, generate_cluster_cards,
    generate_cluster_profiles_table, save_metadata, save_resolved_config
)


def setup_logging(output_dir):
    """Configura logging."""
    log_path = output_dir / 'logs.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_output_structure(output_dir, run_name):
    """Cria estrutura de pastas de output."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    run_dir = output_dir / f"{run_name}__{timestamp}"
    
    subdirs = [
        'data_quality',
        'k_selection',
        'clustering/kmeans',
        'clustering/gmm',
        'clustering/hierarchical',
        'insights'
    ]
    
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return run_dir


def load_config(config_path):
    """Carrega configuração YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config, df, logger):
    """Valida configuração e ajusta se necessário."""
    # Validar colunas numéricas
    numeric_cols = config['columns']['numeric']
    available_numeric = [c for c in numeric_cols if c in df.columns]
    missing_numeric = [c for c in numeric_cols if c not in df.columns]
    
    if missing_numeric:
        logger.warning(f"Colunas numéricas não encontradas: {missing_numeric}")
    config['columns']['numeric'] = available_numeric
    
    # Validar colunas categóricas
    categorical_cols = config['columns']['categorical']
    available_categorical = [c for c in categorical_cols if c in df.columns]
    missing_categorical = [c for c in categorical_cols if c not in df.columns]
    
    if missing_categorical:
        logger.warning(f"Colunas categóricas não encontradas: {missing_categorical}")
    config['columns']['categorical'] = available_categorical
    
    # Validar número mínimo de colunas
    if len(available_numeric) + len(available_categorical) < 2:
        raise ValueError("Menos de 2 colunas úteis encontradas após validação")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Pipeline de Clustering para Empréstimos PJ')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Caminho para arquivo de configuração YAML')
    
    args = parser.parse_args()
    
    # Carregar configuração
    config = load_config(args.config)
    run_name = config['run']['name']
    output_dir = Path(config['run']['output_dir'])
    random_state = config['run']['random_state']
    
    # Criar estrutura de output
    run_dir = create_output_structure(output_dir, run_name)
    
    # Setup logging
    logger = setup_logging(run_dir)
    logger.info(f"Iniciando pipeline: {run_name}")
    
    try:
        # Carregar dados
        logger.info("Carregando dados...")
        df = load_data(config)
        logger.info(f"Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        
        # Validar configuração
        config = validate_config(config, df, logger)
        
        # Salvar config resolvida
        save_resolved_config(config, run_dir / 'run_config_resolved.yaml')
        
        # Data quality - visualizações antes do pré-processamento
        logger.info("Gerando visualizações de qualidade de dados...")
        plot_missingness(df, run_dir / 'data_quality' / 'missingness.png')
        
        numeric_cols = config['columns']['numeric']
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        if numeric_cols:
            plot_numeric_distributions(df, numeric_cols, 
                                     run_dir / 'data_quality' / 'numeric_distributions.png')
        
        # Pré-processamento
        logger.info("Executando pré-processamento...")
        id_col = config['input']['id_column']
        df_original = df.copy()
        df_processed, fit_objects = preprocess_pipeline(df, config)
        
        # Separar features e ID
        X = df_processed.drop(columns=[id_col]).values
        ids = df_processed[id_col].values
        
        logger.info(f"Após pré-processamento: {X.shape[0]} amostras, {X.shape[1]} features")
        
        # Seleção de K
        logger.info("Executando seleção de K...")
        k_selection_config = config['k_selection']
        initial_k_range = k_selection_config['initial_k_range']
        
        # Elbow method
        k_values, inertias = compute_elbow_kmeans(X, initial_k_range, random_state)
        plot_elbow(k_values, inertias, 
                  run_dir / 'k_selection' / 'elbow_kmeans.png', run_name)
        
        # Sugerir intervalo de k
        suggested_range = suggest_k_from_elbow(
            k_values, inertias, 
            k_selection_config['elbow'].get('drop_threshold_pct', 10)
        )
        
        if suggested_range is None:
            suggested_range = k_selection_config['elbow'].get('suggested_k_window', [4, 8])
            logger.info(f"Elbow não encontrou intervalo claro, usando fallback: {suggested_range}")
        else:
            logger.info(f"Intervalo sugerido pelo elbow: {suggested_range}")
        
        # Selecionar melhor k
        k_best, df_metrics = select_best_k(X, suggested_range, random_state)
        logger.info(f"Melhor k (K-Means): {k_best}")
        
        # Salvar métricas
        df_metrics.to_csv(run_dir / 'k_selection' / 'metrics_summary.csv', index=False)
        plot_silhouette_metrics(df_metrics, 
                               run_dir / 'k_selection' / 'silhouette_vs_k.png', run_name)
        
        k_selection_summary = {
            'k_best': int(k_best),
            'suggested_range': suggested_range,
            'metrics': df_metrics.to_dict('records')
        }
        
        import json
        with open(run_dir / 'k_selection' / 'k_selection_summary.json', 'w') as f:
            json.dump(k_selection_summary, f, indent=2, default=str)
        
        # Computar embedding para visualização
        viz_config = config['visualization']
        X_embed, pca_model = compute_embedding(
            X, 
            method=viz_config['embedding']['method'],
            random_state=random_state
        )
        
        setup_plot_style(
            dpi=viz_config['plots']['dpi'],
            style=viz_config['plots']['style']
        )
        
        results = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'k_best': k_best
        }
        
        # K-Means
        if config['methods']['kmeans']['enabled']:
            logger.info(f"Executando K-Means com k={k_best}...")
            kmeans_results = kmeans_clustering(
                X, k_best, random_state,
                n_init=config['methods']['kmeans'].get('n_init', 'auto')
            )
            
            plot_scatter(
                X_embed, kmeans_results['labels'],
                f'K-Means (k={k_best})',
                run_name,
                run_dir / 'clustering' / 'kmeans' / 'pca_scatter.png',
                'K-Means'
            )
            
            # Perfis
            numeric_cols_orig = [c for c in config['columns']['numeric'] if c in df_original.columns]
            if numeric_cols_orig:
                plot_numeric_profiles(
                    df_original, kmeans_results['labels'], numeric_cols_orig,
                    run_dir / 'clustering' / 'kmeans' / 'cluster_profiles_numeric.png',
                    run_name, 'K-Means'
                )
            
            categorical_cols_orig = [c for c in config['columns']['categorical'] if c in df_original.columns]
            if categorical_cols_orig:
                plot_categorical_profiles(
                    df_original, kmeans_results['labels'], categorical_cols_orig,
                    run_dir / 'clustering' / 'kmeans' / 'cluster_profiles_categorical.png',
                    run_name, 'K-Means',
                    top_n=config['reporting']['top_categories_per_feature']
                )
            
            results['kmeans_results'] = kmeans_results
        
        # GMM
        if config['methods']['gmm']['enabled']:
            logger.info("Executando GMM...")
            gmm_config = config['methods']['gmm']
            gmm_results = gmm_clustering(
                X, suggested_range, random_state,
                covariance_type=gmm_config.get('covariance_type', 'full'),
                criterion=gmm_config.get('criterion', 'bic')
            )
            
            k_gmm_best = gmm_results['k_best']
            logger.info(f"Melhor k (GMM): {k_gmm_best}")
            results['k_gmm_best'] = k_gmm_best
            
            plot_scatter(
                X_embed, gmm_results['labels'],
                f'GMM (k={k_gmm_best})',
                run_name,
                run_dir / 'clustering' / 'gmm' / 'pca_scatter.png',
                'GMM'
            )
            
            plot_confidence_hist(
                gmm_results['max_probabilities'],
                run_dir / 'clustering' / 'gmm' / 'confidence_hist.png',
                run_name
            )
            
            # Perfis
            numeric_cols_orig = [c for c in config['columns']['numeric'] if c in df_original.columns]
            if numeric_cols_orig:
                plot_numeric_profiles(
                    df_original, gmm_results['labels'], numeric_cols_orig,
                    run_dir / 'clustering' / 'gmm' / 'cluster_profiles_numeric.png',
                    run_name, 'GMM'
                )
            
            categorical_cols_orig = [c for c in config['columns']['categorical'] if c in df_original.columns]
            if categorical_cols_orig:
                plot_categorical_profiles(
                    df_original, gmm_results['labels'], categorical_cols_orig,
                    run_dir / 'clustering' / 'gmm' / 'cluster_profiles_categorical.png',
                    run_name, 'GMM',
                    top_n=config['reporting']['top_categories_per_feature']
                )
            
            results['gmm_results'] = gmm_results
        
        # Hierárquico
        if config['methods']['hierarchical']['enabled']:
            logger.info(f"Executando Clustering Hierárquico com k={k_best}...")
            hier_config = config['methods']['hierarchical']
            
            # Dendrograma
            linkage_matrix, sample_indices = compute_linkage_matrix(
                X,
                linkage_method=hier_config.get('linkage', 'ward'),
                max_points=hier_config['dendrogram'].get('max_points', 1000)
            )
            plot_dendrogram(
                linkage_matrix,
                run_dir / 'clustering' / 'hierarchical' / 'dendrogram.png',
                run_name,
                max_points=hier_config['dendrogram'].get('max_points', 1000)
            )
            
            # Clustering com k_best
            hier_results = hierarchical_clustering(
                X, k_best,
                linkage_method=hier_config.get('linkage', 'ward')
            )
            
            plot_scatter(
                X_embed, hier_results['labels'],
                f'Clustering Hierárquico (k={k_best})',
                run_name,
                run_dir / 'clustering' / 'hierarchical' / 'pca_scatter_cut_kbest.png',
                'Hierarchical'
            )
            
            results['hierarchical_results'] = hier_results
        
        # Gerar assignments.csv
        logger.info("Gerando assignments.csv...")
        assignments = pd.DataFrame({
            id_col: ids
        })
        
        if 'kmeans_results' in results:
            assignments['cluster_kmeans'] = results['kmeans_results']['labels']
            assignments['kmeans_distance_to_centroid'] = results['kmeans_results']['distances']
        
        if 'gmm_results' in results:
            assignments['cluster_gmm'] = results['gmm_results']['labels']
            assignments['gmm_max_prob'] = results['gmm_results']['max_probabilities']
        
        if 'hierarchical_results' in results:
            assignments['cluster_hier_kbest'] = results['hierarchical_results']['labels']
        
        assignments.to_csv(run_dir / 'clustering' / 'assignments.csv', index=False)
        
        # Cluster sizes
        cluster_sizes = []
        if 'gmm_results' in results:
            sizes = pd.Series(results['gmm_results']['labels']).value_counts().sort_index()
            for cluster_id, size in sizes.items():
                cluster_sizes.append({
                    'method': 'gmm',
                    'cluster': cluster_id,
                    'size': size,
                    'percentage': (size / len(X)) * 100
                })
        
        if cluster_sizes:
            pd.DataFrame(cluster_sizes).to_csv(
                run_dir / 'clustering' / 'cluster_sizes.csv', index=False
            )
        
        # Relatórios
        logger.info("Gerando relatórios...")
        generate_executive_summary(config, results, run_dir / 'insights' / 'executive_summary.md')
        
        if 'gmm_results' in results:
            generate_cluster_cards(df_original, results, config, 
                                 run_dir / 'insights' / 'cluster_cards.md')
            generate_cluster_profiles_table(df_original, results, config,
                                          run_dir / 'insights' / 'cluster_profiles_table.csv')
        
        # Metadados
        save_metadata(config, results, run_dir / 'run_metadata.json')
        
        logger.info(f"Pipeline concluído com sucesso! Outputs em: {run_dir}")
        
    except Exception as e:
        logger.error(f"Erro no pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

