"""
Script para gerar dataset fictício de contratos de empréstimos
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def gerar_dados_emprestimos(n_contratos=1000, seed=42):
    """
    Gera um dataset fictício de contratos de empréstimos
    
    Args:
        n_contratos: Número de contratos a gerar
        seed: Seed para reprodutibilidade
    """
    np.random.seed(seed)
    
    # Produtos de empréstimo disponíveis
    produtos = [
        'Empréstimo Pessoal',
        'Empréstimo Consignado',
        'Empréstimo com Garantia',
        'Crédito Rotativo',
        'Financiamento Veículo',
        'Financiamento Imóvel'
    ]
    
    # Gerar dados
    dados = {
        'id_contrato': range(1, n_contratos + 1),
        'valor_emprestimo': np.random.lognormal(mean=9.5, sigma=0.8, size=n_contratos).round(2),
        'quantidade_parcelas': np.random.choice([12, 24, 36, 48, 60, 72, 84], size=n_contratos, p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.07, 0.03]),
        'taxa_juros_anual': np.random.normal(loc=15, scale=5, size=n_contratos).clip(8, 30).round(2),
        'score_cliente': np.random.normal(loc=650, scale=100, size=n_contratos).clip(300, 850).round(0).astype(int),
        'idade_cliente': np.random.normal(loc=42, scale=12, size=n_contratos).clip(18, 75).round(0).astype(int),
        'renda_mensal': np.random.lognormal(mean=8.5, sigma=0.7, size=n_contratos).round(2),
        'nome_produto': np.random.choice(produtos, size=n_contratos, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]),
    }
    
    df = pd.DataFrame(dados)
    
    # Ajustar valores para criar correlações realistas
    # Clientes com maior score tendem a ter taxas menores
    df['taxa_juros_anual'] = df['taxa_juros_anual'] - (df['score_cliente'] - 650) * 0.02
    df['taxa_juros_anual'] = df['taxa_juros_anual'].clip(8, 30).round(2)
    
    # Produtos diferentes têm características diferentes
    df.loc[df['nome_produto'] == 'Empréstimo Consignado', 'taxa_juros_anual'] = np.random.normal(12, 2, 
        size=len(df[df['nome_produto'] == 'Empréstimo Consignado'])).clip(8, 18)
    df.loc[df['nome_produto'] == 'Financiamento Imóvel', 'quantidade_parcelas'] = np.random.choice([120, 180, 240, 300], 
        size=len(df[df['nome_produto'] == 'Financiamento Imóvel']))
    df.loc[df['nome_produto'] == 'Financiamento Imóvel', 'valor_emprestimo'] = np.random.lognormal(11, 0.5, 
        size=len(df[df['nome_produto'] == 'Financiamento Imóvel'])).round(2)
    
    # Calcular valor da parcela
    taxa_mensal = df['taxa_juros_anual'] / 100 / 12
    df['valor_parcela'] = (df['valor_emprestimo'] * 
                          (taxa_mensal * (1 + taxa_mensal)**df['quantidade_parcelas']) / 
                          ((1 + taxa_mensal)**df['quantidade_parcelas'] - 1)).round(2)
    
    # Calcular prazo em dias
    df['prazo_dias'] = df['quantidade_parcelas'] * 30
    
    # Calcular relação parcela/renda (importante para análise de risco)
    df['parcela_sobre_renda'] = (df['valor_parcela'] / df['renda_mensal'] * 100).round(2)
    
    # Data de contratação (últimos 2 anos)
    data_inicio = datetime.now() - timedelta(days=730)
    df['data_contratacao'] = pd.date_range(start=data_inicio, periods=n_contratos, freq='D')
    np.random.shuffle(df['data_contratacao'].values)
    
    # Reordenar colunas
    colunas_ordenadas = [
        'id_contrato',
        'data_contratacao',
        'nome_produto',
        'valor_emprestimo',
        'quantidade_parcelas',
        'prazo_dias',
        'taxa_juros_anual',
        'valor_parcela',
        'score_cliente',
        'idade_cliente',
        'renda_mensal',
        'parcela_sobre_renda'
    ]
    
    df = df[colunas_ordenadas]
    
    return df

if __name__ == "__main__":
    # Gerar dataset
    print("Gerando dataset de empréstimos...")
    df = gerar_dados_emprestimos(n_contratos=1000)
    
    # Salvar em CSV
    df.to_csv('dados_emprestimos.csv', index=False)
    print(f"Dataset gerado com sucesso! {len(df)} contratos salvos em 'dados_emprestimos.csv'")
    
    # Estatísticas básicas
    print("\n=== Estatísticas do Dataset ===")
    print(df.describe())
    print("\n=== Distribuição por Produto ===")
    print(df['nome_produto'].value_counts())

