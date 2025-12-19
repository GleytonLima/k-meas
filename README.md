# Análise de Clusters de Empréstimos usando K-Means

Este projeto implementa uma análise de clustering usando o algoritmo K-Means para segmentar contratos de empréstimos fictícios. O objetivo é identificar grupos de contratos com características similares para obter insights sobre o perfil dos clientes e produtos.

## 📋 Sobre o Projeto

O projeto utiliza dados fictícios de contratos de empréstimos com os seguintes atributos:

- **valor_emprestimo**: Valor total do empréstimo
- **quantidade_parcelas**: Número de parcelas do contrato
- **taxa_juros_anual**: Taxa de juros anual (%)
- **score_cliente**: Score de crédito do cliente (300-850)
- **idade_cliente**: Idade do cliente
- **renda_mensal**: Renda mensal do cliente
- **valor_parcela**: Valor de cada parcela
- **parcela_sobre_renda**: Percentual da parcela sobre a renda
- **nome_produto**: Tipo de produto (Pessoal, Consignado, Garantia, etc.)
- **prazo_dias**: Prazo total em dias
- **data_contratacao**: Data de contratação do empréstimo

## 🚀 Como Usar

### 1. Criar e Ativar Ambiente Virtual

**Windows (Git Bash/PowerShell):**
```bash
python -m venv venv
source venv/Scripts/activate  # Git Bash
# ou
venv\Scripts\activate  # PowerShell/CMD
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Instalar Dependências

Com o ambiente virtual ativado:

```bash
pip install -r requirements.txt
```

### 3. Gerar Dataset

Execute o script para gerar o dataset fictício:

```bash
python gerar_dados.py
```

Isso criará o arquivo `dados_emprestimos.csv` com 1000 contratos fictícios.

### 4. (Opcional) Analisar K Ideal

Para uma análise detalhada do número ideal de clusters usando o método do cotovelo:

```bash
python analisar_k_ideal.py
```

**Parâmetros disponíveis:**
- `--k-min`: Valor mínimo de k para testar (padrão: 2)
- `--k-max`: Valor máximo de k para testar (padrão: 10)
- `--arquivo`: Arquivo CSV de entrada (padrão: dados_emprestimos.csv)
- `--features`: Lista de features a usar (separadas por espaço)
- `--saida`: Nome do arquivo de saída (padrão: analise_k_ideal.png)

**Exemplos:**
```bash
# Range customizado
python analisar_k_ideal.py --k-min 3 --k-max 15

# Com arquivo diferente
python analisar_k_ideal.py --arquivo meus_dados.csv --k-min 2 --k-max 8
```

Este script gera:
- Gráficos detalhados do método do cotovelo
- Análise de múltiplas métricas (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Visualização da segunda derivada para identificar o "cotovelo"
- Tabela resumo com recomendações

### 5. Executar Análise K-Means

Execute o script principal de análise:

```bash
python kmeans_emprestimos.py
```

**Parâmetros disponíveis:**
- `--k`: Número de clusters (k). Se não especificado, será determinado automaticamente
- `--pares`: Pares de features para visualização. Formato: `"feat1,feat2;feat3,feat4"` ou `"feat1,feat2 feat3,feat4"`
- `--arquivo`: Arquivo CSV de entrada (padrão: dados_emprestimos.csv)
- `--saida`: Arquivo CSV de saída (padrão: resultados_clusters.csv)
- `--features`: Lista de features a usar no clustering (separadas por espaço)
- `--k-range`: Range de k para busca automática no formato min-max (padrão: 2-10)

**Exemplos:**
```bash
# Especificar k=3
python kmeans_emprestimos.py --k 3

# Especificar k e pares de visualização
python kmeans_emprestimos.py --k 4 --pares "valor_emprestimo,quantidade_parcelas;score_cliente,taxa_juros_anual"

# Usar arquivo diferente e k customizado
python kmeans_emprestimos.py --arquivo meus_dados.csv --k 5

# Busca automática com range customizado
python kmeans_emprestimos.py --k-range 3-12
```

Este script irá:
- Carregar e preparar os dados
- Encontrar o melhor número de clusters (k) ou usar o k especificado
- Aplicar o algoritmo K-Means
- Gerar visualizações com os pares especificados (ou padrão)
- Criar análise detalhada dos clusters
- Salvar resultados em CSV

## 📊 Saídas do Projeto

Os scripts geram os seguintes arquivos:

### Arquivos de Dados
1. **dados_emprestimos.csv**: Dataset original com os contratos
2. **resultados_clusters.csv**: Dataset com a coluna de cluster adicionada

### Imagens (salvas em `imagens/`)
3. **analise_k_ideal.png**: Análise completa do k ideal (gerado por `analisar_k_ideal.py`)
4. **analise_melhor_k.png**: Gráficos para escolha do melhor k (método do cotovelo e silhouette score)
5. **visualizacao_clusters.png**: Visualizações 2D dos clusters em diferentes dimensões
6. **distribuicao_produtos_clusters.png**: Distribuição de produtos por cluster

**Nota:** Todas as imagens são salvas na pasta `imagens/` por padrão. Você pode alterar isso usando o parâmetro `--pasta_imagens` ou no `config.json`.

## 🔍 Metodologia

### Preparação dos Dados

- **Normalização**: Os dados são normalizados usando `StandardScaler` para garantir que todas as features tenham a mesma escala
- **Seleção de Features**: Utiliza features numéricas relevantes para o clustering

### Escolha do Número de Clusters

O projeto utiliza múltiplas técnicas para determinar o k ideal:

1. **Método do Cotovelo (Elbow Method)**: Analisa a inércia (within-cluster sum of squares) para diferentes valores de k. O "cotovelo" é identificado através da análise da segunda derivada.
2. **Silhouette Score**: Mede a qualidade dos clusters, escolhendo o k com maior score (varia de -1 a 1, maior é melhor)
3. **Davies-Bouldin Index**: Mede a separação entre clusters (menor é melhor)
4. **Calinski-Harabasz Score**: Mede a razão entre dispersão entre clusters e dentro dos clusters (maior é melhor)

O script `analisar_k_ideal.py` fornece uma análise visual completa de todas essas métricas.

### Algoritmo K-Means

- Implementação do scikit-learn
- Inicialização aleatória com seed fixa para reprodutibilidade
- Múltiplas inicializações (n_init=10) para encontrar melhor resultado

## 📈 Insights Gerados

A análise fornece:

- **Perfil de cada cluster**: Características médias de valor, score, taxa, etc.
- **Distribuição de produtos**: Quais produtos são mais comuns em cada cluster
- **Recomendações**: Alertas sobre risco e oportunidades de negócio
- **Segmentação de clientes**: Identificação de grupos com comportamentos similares

## 🎯 Casos de Uso

Este tipo de análise pode ser útil para:

- **Segmentação de clientes**: Identificar perfis distintos de clientes
- **Gestão de risco**: Agrupar contratos por nível de risco
- **Otimização de produtos**: Entender quais produtos atraem quais perfis
- **Estratégia de precificação**: Ajustar taxas baseado em perfis de cluster
- **Marketing direcionado**: Criar campanhas específicas para cada segmento

## 📝 Estrutura do Projeto

```
k-means/
├── venv/                        # Ambiente virtual (não versionado)
├── gerar_dados.py               # Script para gerar dataset fictício
├── analisar_k_ideal.py          # Script para análise detalhada do k ideal
├── kmeans_emprestimos.py        # Script principal de análise
├── config.json                  # Arquivo de configuração (criar/editar)
├── config.exemplo.json          # Exemplo de configuração
├── requirements.txt             # Dependências do projeto
├── .gitignore                   # Arquivos ignorados pelo Git
├── README.md                    # Este arquivo
├── EXEMPLOS_USO.md              # Exemplos de uso dos scripts
├── dados_emprestimos.csv        # Dataset gerado (após execução)
├── resultados_clusters.csv      # Resultados com clusters (após execução)
└── imagens/                     # Pasta com todas as imagens geradas
    ├── analise_k_ideal.png
    ├── analise_melhor_k.png
    ├── visualizacao_clusters.png
    └── distribuicao_produtos_clusters.png
```

## 🔧 Personalização

Você pode personalizar a análise de várias formas:

### Via Arquivo de Configuração JSON (Recomendado)

Crie ou edite o arquivo `config.json` para definir todas as configurações:

```json
{
  "kmeans_emprestimos": {
    "k": 4,
    "k_range": {"min": 2, "max": 10},
    "pares_visualizacao": [
      ["valor_emprestimo", "quantidade_parcelas"],
      ["score_cliente", "taxa_juros_anual"]
    ],
    "features": ["valor_emprestimo", "quantidade_parcelas", "taxa_juros_anual"],
    "arquivo_entrada": "dados_emprestimos.csv",
    "arquivo_saida": "resultados_clusters.csv"
  },
  "analisar_k_ideal": {
    "k_min": 2,
    "k_max": 10,
    "arquivo_entrada": "dados_emprestimos.csv",
    "arquivo_saida": "analise_k_ideal.png"
  }
}
```

**Vantagens do config.json:**
- ✅ Fácil de versionar e compartilhar
- ✅ Reutilizável para diferentes experimentos
- ✅ Organiza todas as configurações em um só lugar
- ✅ Parâmetros CLI têm prioridade sobre o config (permite sobrescrever)

**Uso:**
```bash
# Usar config.json padrão
python kmeans_emprestimos.py

# Usar config customizado
python kmeans_emprestimos.py --config meu_config.json

# Desabilitar config e usar apenas CLI
python kmeans_emprestimos.py --config ""
```

### Via Parâmetros de Linha de Comando

- **Número de clusters (k)**: Use `--k` para especificar diretamente ou `--k-range` para busca automática
- **Pares de visualização**: Use `--pares` para especificar quais pares de features visualizar
- **Features utilizadas**: Use `--features` para escolher quais features usar no clustering
- **Arquivos**: Use `--arquivo` e `--saida` para especificar arquivos de entrada/saída
- **Config**: Use `--config` para especificar arquivo de configuração ou `--config ""` para desabilitar

**Nota:** Parâmetros CLI sempre têm prioridade sobre o config.json

### Via Código

- **Número de contratos**: Altere o parâmetro `n_contratos` em `gerar_dados.py`
- **Features padrão**: Modifique a lista `features` nas funções `preparar_dados()`
- **Pares padrão de visualização**: Modifique a lista em `visualizar_clusters()`

## 📚 Referências

- [Scikit-learn K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

## 📄 Licença

Este é um projeto educacional para estudo de algoritmos de clustering.

