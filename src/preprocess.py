"""
Módulo de pré-processamento de dados para clustering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')


def load_data(config):
    """Carrega o CSV conforme configuração."""
    input_config = config['input']
    
    df = pd.read_csv(
        input_config['path'],
        sep=input_config['separator'],
        encoding=input_config['encoding'],
        decimal=input_config['decimal']
    )
    
    return df


def handle_missing_values(df, config):
    """Trata valores faltantes conforme estratégia configurada."""
    preprocess_config = config['preprocess']
    missing_config = preprocess_config['missing']
    
    # Numéricas
    numeric_cols = config['columns']['numeric']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    if numeric_cols:
        strategy = missing_config['numeric_strategy']
        if strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Categóricas
    categorical_cols = config['columns']['categorical']
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    if categorical_cols:
        fill_value = missing_config.get('categorical_fill_value', 'MISSING')
        df[categorical_cols] = df[categorical_cols].fillna(fill_value)
    
    return df


def apply_transforms(df, config):
    """Aplica transformações (log1p) nas colunas especificadas."""
    transforms_config = config['preprocess']['transforms']
    log_cols = transforms_config.get('log1p_columns', [])
    
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    
    return df


def handle_outliers(df, config):
    """Trata outliers usando winsorização."""
    preprocess_config = config['preprocess']
    outliers_config = preprocess_config.get('outliers', {})
    
    if not outliers_config.get('enabled', False):
        return df
    
    method = outliers_config.get('method', 'winsorize')
    if method != 'winsorize':
        return df
    
    lower_q = outliers_config.get('lower_quantile', 0.01)
    upper_q = outliers_config.get('upper_quantile', 0.99)
    
    numeric_cols = config['columns']['numeric']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    for col in numeric_cols:
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        df[col] = df[col].clip(lower=lower, upper=upper)
    
    return df


def encode_categorical(df, config, fit_encoder=None):
    """Aplica one-hot encoding nas variáveis categóricas."""
    categorical_cols = config['columns']['categorical']
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    if not categorical_cols:
        return df, None
    
    encoding_config = config['preprocess']['encoding']
    min_freq = encoding_config.get('onehot_min_frequency', 0.01)
    
    encoded_dfs = []
    encoders = {}
    
    for col in categorical_cols:
        # Agrupar categorias raras
        value_counts = df[col].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < min_freq].index
        
        df_col = df[col].copy()
        if len(rare_categories) > 0:
            df_col = df_col.replace(rare_categories, 'RARE')
        
        # One-hot encoding
        if fit_encoder is None:
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            encoded = encoder.fit_transform(df_col.values.reshape(-1, 1))
            encoders[col] = encoder
        else:
            encoder = fit_encoder[col]
            encoded = encoder.transform(df_col.values.reshape(-1, 1))
            encoders[col] = encoder
        
        feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
        encoded_dfs.append(encoded_df)
    
    # Remover colunas categóricas originais e adicionar encoded
    df = df.drop(columns=categorical_cols)
    for encoded_df in encoded_dfs:
        df = pd.concat([df, encoded_df], axis=1)
    
    return df, encoders


def scale_numeric(df, config, fit_scaler=None):
    """Aplica scaling nas variáveis numéricas."""
    numeric_cols = config['columns']['numeric']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    if not numeric_cols:
        return df, None
    
    scaling_config = config['preprocess']['scaling']
    scaler_type = scaling_config.get('numeric_scaler', 'robust')
    
    if scaler_type == 'robust':
        scaler = RobustScaler() if fit_scaler is None else fit_scaler
    elif scaler_type == 'standard':
        scaler = StandardScaler() if fit_scaler is None else fit_scaler
    else:
        return df, None
    
    if fit_scaler is None:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df, scaler


def preprocess_pipeline(df, config, fit_objects=None):
    """
    Pipeline completo de pré-processamento.
    
    Args:
        df: DataFrame original
        config: Configuração do pipeline
        fit_objects: Dict com encoders/scalers já treinados (para inferência)
    
    Returns:
        df_processed: DataFrame processado
        fit_objects: Dict com encoders/scalers treinados
    """
    df = df.copy()
    
    # Validar colunas
    id_col = config['input']['id_column']
    if id_col not in df.columns:
        df['row_id'] = range(len(df))
        id_col = 'row_id'
        config['input']['id_column'] = id_col
    
    # Remover colunas a serem descartadas
    drop_cols = config['columns'].get('drop', [])
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    # Separar ID
    id_series = df[id_col]
    df = df.drop(columns=[id_col])
    
    # Tratar faltantes
    df = handle_missing_values(df, config)
    
    # Aplicar transformações
    df = apply_transforms(df, config)
    
    # Tratar outliers
    df = handle_outliers(df, config)
    
    # Encoding categóricas
    if fit_objects is None:
        df, encoders = encode_categorical(df, config)
        fit_objects = {'encoders': encoders}
    else:
        df, _ = encode_categorical(df, config, fit_objects['encoders'])
    
    # Scaling numéricas
    if fit_objects is None or 'scaler' not in fit_objects:
        df, scaler = scale_numeric(df, config)
        fit_objects['scaler'] = scaler
    else:
        df, _ = scale_numeric(df, config, fit_objects['scaler'])
    
    # Validar número mínimo de colunas
    if df.shape[1] < 2:
        raise ValueError(f"Após pré-processamento, sobraram apenas {df.shape[1]} colunas. Mínimo necessário: 2")
    
    # Reinserir ID
    df[id_col] = id_series
    
    return df, fit_objects

