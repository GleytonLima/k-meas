# Pipeline de Clustering para Empréstimos PJ

Pipeline reprodutível para análise de clustering de empréstimos pessoa jurídica, gerando insights acionáveis através de múltiplos algoritmos de clustering.

## 🚀 Início Rápido

### Pré-requisitos

- Docker e Docker Compose instalados
- Arquivo CSV com dados de empréstimos

### Execução

1. **Coloque seus dados:**
   ```bash
   cp seu_arquivo.csv data/input.csv
   ```

2. **Configure (opcional):**
   Edite `configs/config.yaml` conforme necessário

3. **Execute:**
   ```bash
   docker compose build
   docker compose run --rm pipeline
   ```

4. **Resultados:**
   Os resultados estarão em `outputs/[nome_do_run]__[timestamp]/`

## 📁 Estrutura do Projeto

```
pj-loans-clustering-insights/
├── README.md                 # Este arquivo
├── docker-compose.yml        # Configuração Docker
├── Dockerfile                # Imagem Docker
├── requirements.txt          # Dependências Python
│
├── data/                     # Dados de entrada
│   ├── input.csv            # ⭐ Coloque seu CSV aqui
│   └── README.md
│
├── configs/                  # Configurações
│   ├── config.yaml          # ⭐ Edite este arquivo
│   ├── config.example.yaml
│   └── README.md
│
├── src/                      # Código fonte
│   ├── run_pipeline.py      # Entrypoint principal
│   ├── preprocess.py
│   ├── clustering.py
│   ├── visualization.py
│   └── reporting.py
│
├── outputs/                  # Resultados (gerado automaticamente)
│   └── .gitkeep
│
└── docs/                     # Documentação
    ├── analyst_guide.md     # ⭐ Guia do analista
    ├── methodology.md
    └── faq.md
```

## 🎯 Funcionalidades

### Métodos de Clustering

- **K-Means**: Baseline rápido e interpretável
- **GMM (Gaussian Mixture Model)**: Identifica padrões ricos e fornece probabilidades
- **Hierárquico**: Validação e storytelling através de dendrogramas

### Seleção de K Assistida

- **Elbow Method**: Filtro inicial de intervalo
- **Silhouette/CH/DB**: Métricas para comparação e sugestão de k
- **BIC/AIC**: Seleção automática de k para GMM

### Pré-processamento

- Tratamento de valores faltantes (median/mean para numéricas, constante para categóricas)
- Transformações (log1p para variáveis com distribuição assimétrica)
- Scaling (RobustScaler ou StandardScaler)
- Encoding (One-Hot para categóricas)
- Tratamento de outliers (winsorização)

### Visualizações

- Scatter plots 2D (PCA embedding)
- Elbow plots
- Métricas de qualidade (Silhouette, CH, DB)
- Dendrogramas
- Perfis numéricos (heatmaps)
- Perfis categóricos (barras)
- Histogramas de confiança (GMM)

### Relatórios

- **Executive Summary**: Resumo executivo com principais achados
- **Cluster Cards**: Descrição detalhada de cada cluster
- **Assignments CSV**: Atribuições de cluster para cada amostra
- **Métricas**: CSV com métricas de qualidade

## 📊 Exemplo de Output

Após execução, você terá:

```
outputs/
  emprestimos_pj_2025__2025-12-19_1530/
    ├── run_config_resolved.yaml
    ├── run_metadata.json
    ├── logs.txt
    │
    ├── data_quality/
    │   ├── missingness.png
    │   └── numeric_distributions.png
    │
    ├── k_selection/
    │   ├── elbow_kmeans.png
    │   ├── silhouette_vs_k.png
    │   ├── metrics_summary.csv
    │   └── k_selection_summary.json
    │
    ├── clustering/
    │   ├── assignments.csv          # ⭐ Principal
    │   ├── cluster_sizes.csv
    │   ├── kmeans/
    │   ├── gmm/
    │   └── hierarchical/
    │
    └── insights/
        ├── executive_summary.md     # ⭐ Resumo
        ├── cluster_cards.md
        └── cluster_profiles_table.csv
```

## ⚙️ Configuração

Edite `configs/config.yaml` para:

- Definir colunas numéricas e categóricas
- Ajustar parâmetros de pré-processamento
- Configurar métodos de clustering
- Personalizar visualizações

Consulte `docs/analyst_guide.md` para guia detalhado.

## 📖 Documentação

- **[Guia do Analista](docs/analyst_guide.md)**: Como usar o pipeline
- **[Especificação](INSTRUCOES_IMPLEMENTACAO.md)**: Especificação completa do projeto

## 🔧 Desenvolvimento

### Sem Docker

```bash
# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar
python src/run_pipeline.py --config configs/config.yaml
```

## 📝 Licença

Este projeto é para uso interno.

## 🤝 Contribuindo

Para sugestões e melhorias, consulte a equipe de desenvolvimento.

---

**Desenvolvido para análise de empréstimos PJ**

