# Especificação do Projeto: `pj-loans-clustering-insights`

## 1) Objetivo do projeto

Criar um pipeline reprodutível que:

1. Lê um CSV de empréstimos PJ (ex.: 2025).
2. Faz pré-processamento padronizado (faltantes, log, escala, one-hot).
3. Executa clustering com:

   * **k-means** (baseline)
   * **GMM** (principal para padrões ricos)
   * **Hierárquico** (validação/storytelling)
4. Seleciona `k` de forma **assistida**:

   * **Elbow** = filtro de intervalo (k-means)
   * **Silhouette/CH/DB** = sugestão de `k_best` (k-means)
   * **BIC/AIC** = escolha de `k_gmm_best` (GMM)
5. Gera saídas fáceis de consumir:

   * imagens “prontas para PPT”
   * CSVs com métricas, perfis e atribuições de cluster
   * config “auditable” do run

---

# 2) Requisitos não-funcionais (simplicidade e operação)

* **Modo batch**: rodar e terminar (não é webapp).
* **1 comando** para rodar via Docker.
* Zero dependência local além de Docker/Compose.
* **Config em YAML** (analista edita sem mexer no código).
* Saídas determinísticas com `random_state`.
* Estrutura de outputs “sempre igual” para facilitar comparação.

---

# 3) Estrutura de pastas do repositório

```
pj-loans-clustering-insights/
  README.md
  docker-compose.yml
  Dockerfile
  requirements.txt

  data/
    README.md
    input.csv              # (analista troca este arquivo)
    dictionary.xlsx        # (opcional) dicionário de dados

  configs/
    README.md
    config.example.yaml    # modelo
    config.yaml            # (analista copia do example e ajusta)
    profiles.example.yaml  # (opcional) perfis de execução (rápido/completo)

  src/
    README.md
    run_pipeline.py        # entrypoint (orquestra tudo)
    (demais módulos a critério do time de dev)

  outputs/
    README.md
    .gitkeep               # manter pasta no repo

  docs/
    analyst_guide.md       # guia curto para analistas
    methodology.md         # explicação técnica (leve, para referência)
    faq.md                 # dúvidas comuns

  notebooks/
    README.md              # (opcional) só para exploração, não obrigatório
```

**Regra:** o analista só precisa mexer em:

* `data/input.csv`
* `configs/config.yaml`

---

# 4) Contrato de entrada (CSV)

## 4.1 Formato

* CSV com cabeçalho
* UTF-8
* Separador padrão: vírgula (configurável)
* Decimal: ponto (recomendado). Se vier com vírgula, tratar via config.

## 4.2 Campos mínimos recomendados (exemplos)

Não é obrigatório ter tudo, mas quanto mais completo, melhor:

* Identificador: `id` (contrato ou cliente)
* Variáveis numéricas típicas:

  * `valor_contratado`
  * `prazo_meses`
  * `taxa`
  * `spread`
  * `receita_anual`
  * `ebitda`
  * `divida_liquida`
  * `dl_ebitda`
* Variáveis categóricas típicas:

  * `produto`
  * `setor` ou `cnae`
  * `garantia`
  * `regiao`

## 4.3 Regras

* Linhas duplicadas: política definida em config (ex.: manter última).
* Colunas desconhecidas: ignorar (não quebrar pipeline).
* Valores faltantes: tratados no pré-processamento.

---

# 5) Configuração (YAML) — único ponto de controle

Arquivo: `configs/config.yaml`

## 5.1 Exemplo de layout (especificação)

```yaml
run:
  name: "itau_pj_2025"
  random_state: 42
  output_dir: "/app/outputs"
  overwrite_output_dir: true

input:
  path: "/app/data/input.csv"
  separator: ","
  encoding: "utf-8"
  decimal: "."
  id_column: "id"

columns:
  numeric:
    - valor_contratado
    - prazo_meses
    - taxa
    - spread
    - receita_anual
    - ebitda
  categorical:
    - setor
    - produto
    - garantia
    - regiao
  drop:
    - coluna_inutil

preprocess:
  missing:
    numeric_strategy: "median"        # median|mean
    categorical_strategy: "constant"  # constant
    categorical_fill_value: "MISSING"
  transforms:
    log1p_columns:
      - valor_contratado
      - receita_anual
      - ebitda
  scaling:
    numeric_scaler: "robust"          # robust|standard
  encoding:
    categorical_encoder: "onehot"     # onehot
    onehot_min_frequency: 0.01        # opcional (agrupar raros)
  outliers:
    enabled: true
    method: "winsorize"               # winsorize|none
    lower_quantile: 0.01
    upper_quantile: 0.99

k_selection:
  initial_k_range: [2, 15]            # para elbow (kmeans)
  elbow:
    enabled: true
    drop_threshold_pct: 10            # heurística para sugerir intervalo
    suggested_k_window: [4, 8]        # fallback se heurística falhar
  candidate_k_range: "elbow"          # elbow|manual
  manual_candidate_k_range: [4, 10]   # usado se candidate_k_range=manual

methods:
  kmeans:
    enabled: true
    n_init: "auto"
  gmm:
    enabled: true
    covariance_type: "full"
    criterion: "bic"                  # bic|aic
  hierarchical:
    enabled: true
    linkage: "ward"
    dendrogram:
      max_points: 1000                # amostragem para dendrograma

visualization:
  embedding:
    method: "pca"                     # pca|umap (umap opcional)
  plots:
    dpi: 150
    style: "default"
    save_format: "png"

reporting:
  top_categories_per_feature: 10
  min_cluster_size_pct: 0.03          # alertar clusters muito pequenos
```

## 5.2 Validações obrigatórias (o script deve checar)

* Se `numeric` e `categorical` existem no CSV; se não, avisar e seguir com o que existir.
* Se `id_column` não existir: criar `row_id` automaticamente.
* Se sobrar menos de 2 colunas úteis: abortar com erro claro.

---

# 6) Execução via Docker

## 6.1 Comandos para o analista

**Padrão (recomendado):**

```bash
docker compose build
docker compose run --rm pipeline
```

## 6.2 `docker-compose.yml` (especificação)

* Volume para `data/` e `outputs/`
* O container roda `src/run_pipeline.py --config configs/config.yaml`

Exemplo esperado (sem “subir serviço” contínuo; é job):

* serviço: `pipeline`

---

# 7) Saídas (contrato de outputs)

Todas as execuções devem criar uma subpasta datada (melhor para histórico):

```
outputs/
  itau_pj_2025__2025-12-19_1530/
    run_config_resolved.yaml
    run_metadata.json
    logs.txt

    data_quality/
      missingness.png
      numeric_distributions.png

    k_selection/
      elbow_kmeans.png
      silhouette_vs_k.png
      metrics_summary.csv
      k_selection_summary.json

    clustering/
      assignments.csv
      cluster_sizes.csv

      kmeans/
        pca_scatter.png
        cluster_profiles_numeric.png
        cluster_profiles_categorical.png

      gmm/
        pca_scatter.png
        confidence_hist.png
        cluster_profiles_numeric.png
        cluster_profiles_categorical.png

      hierarchical/
        dendrogram.png
        pca_scatter_cut_kbest.png

    insights/
      executive_summary.md
      cluster_cards.md
      cluster_profiles_table.csv
```

## 7.1 Arquivos essenciais (mínimo)

* `clustering/assignments.csv`
* `k_selection/metrics_summary.csv`
* `k_selection/elbow_kmeans.png`
* `clustering/kmeans/pca_scatter.png`
* `clustering/gmm/pca_scatter.png`
* `clustering/hierarchical/dendrogram.png`
* `insights/executive_summary.md`

## 7.2 Conteúdo de `assignments.csv`

Colunas mínimas:

* `id` (ou `row_id`)
* `cluster_kmeans`
* `cluster_gmm`
* `cluster_hier_kbest`
* `gmm_max_prob` (confiança)
* (opcional) `kmeans_distance_to_centroid`

---

# 8) Padrões visuais (para PPT e leitura rápida)

Todos os gráficos devem:

* ter título com: método + K + nome do run
* ter legenda com tamanho de cluster (n e %)
* ter nota de rodapé:

  * “Embedding (PCA/UMAP) é só visualização”
  * “Clustering foi feito no espaço pré-processado”

**Scatter 2D:**

* PCA (default) para estabilidade e simplicidade
* UMAP opcional via config

**Perfis:**

* numéricas: heatmap de médias padronizadas por cluster
* categóricas: barras com top categorias (top N configurável)

---

# 9) “Pacote de insights” para o analista (outputs/insights)

## 9.1 `executive_summary.md` (template esperado)

* Data do run, tamanho do dataset
* K sugerido para k-means (`k_best`) e para GMM (`k_gmm_best`)
* 3–6 bullets de achados:

  * clusters grandes (representativos)
  * clusters pequenos (nichos)
  * clusters “mistos” (baixa confiança do GMM)
* recomendações de próximos passos:

  * checar variáveis que diferenciam clusters
  * ideias de produto/marketing

## 9.2 `cluster_cards.md`

Um “card” por cluster (principalmente GMM), contendo:

* tamanho (n, %)
* médias das principais numéricas
* top categorias por feature
* “hipótese de negócio” (gerada por template, sem inventar fatos)
* exemplos de 5 IDs (para investigação)

---

# 10) Guia do analista (docs/analyst_guide.md)

Conteúdo mínimo:

1. Onde colocar o CSV (`data/input.csv`)
2. Como editar `configs/config.yaml`
3. Como rodar Docker
4. Onde achar os resultados
5. Como interpretar:

   * elbow (apenas filtro)
   * silhouette (comparação)
   * GMM confidence (clusters bem definidos vs mistos)
6. Checklist de sanity check:

   * clusters muito pequenos
   * variáveis dominando (ex.: “valor” sozinho)
   * outliers

---

# 11) Convenções e governança de execução

## 11.1 Versionamento

* O `run_metadata.json` deve salvar:

  * hash do CSV (ou tamanho+timestamp)
  * versão do container (tag)
  * config resolvida
  * biblioteca/versões (pip freeze)

## 11.2 Reprodutibilidade

* `random_state` sempre explícito.
* Se usar UMAP, fixar `random_state`.

## 11.3 Falhas e mensagens

Erros devem ser “amigáveis”:

* “coluna X não encontrada”
* “nenhuma coluna numérica válida”
* “após limpeza sobraram N linhas — insuficiente”

---

# 12) Definição do que é “simples” (critérios de aceitação)

O projeto está OK quando:

* Um analista consegue rodar sem IDE, só Docker.
* Trocar CSV + ajustar YAML gera outputs coerentes.
* A pasta de outputs é autoexplicativa.
* Existem pelo menos:

  * elbow + silhouette
  * scatter 2D por método
  * perfis por cluster
  * assignments.csv
  * summary.md

