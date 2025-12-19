# Guia do Analista

## 1. Preparação dos Dados

### Onde colocar o CSV

Coloque seu arquivo CSV em `data/input.csv`.

**Importante:** O arquivo deve ter:
- Cabeçalho na primeira linha
- Encoding UTF-8
- Separador: vírgula (ou ajuste em `configs/config.yaml`)

### Formato esperado

Mínimo necessário:
- Uma coluna de identificação (ex: `id_contrato`)
- Pelo menos 2 colunas numéricas ou categóricas

## 2. Configuração

### Editar `configs/config.yaml`

Principais ajustes:

1. **Colunas numéricas:**
```yaml
columns:
  numeric:
    - valor_emprestimo
    - taxa_juros_anual
    # ... adicione suas colunas
```

2. **Colunas categóricas:**
```yaml
columns:
  categorical:
    - nome_produto
    # ... adicione suas colunas
```

3. **Coluna de ID:**
```yaml
input:
  id_column: "id_contrato"  # Nome da sua coluna de ID
```

4. **Nome do run:**
```yaml
run:
  name: "meu_analise_2025"  # Nome identificador
```

## 3. Execução

### Via Docker (recomendado)

```bash
# Construir imagem
docker compose build

# Executar pipeline
docker compose run --rm pipeline
```

### Sem Docker (desenvolvimento)

```bash
# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar
python src/run_pipeline.py --config configs/config.yaml
```

## 4. Resultados

Os resultados são salvos em `outputs/[nome_do_run]__[timestamp]/`

### Estrutura de outputs

```
outputs/
  meu_analise_2025__2025-12-19_1530/
    ├── run_config_resolved.yaml      # Config usada
    ├── run_metadata.json             # Metadados
    ├── logs.txt                      # Logs da execução
    │
    ├── data_quality/                 # Análise de qualidade
    │   ├── missingness.png
    │   └── numeric_distributions.png
    │
    ├── k_selection/                  # Seleção de K
    │   ├── elbow_kmeans.png
    │   ├── silhouette_vs_k.png
    │   ├── metrics_summary.csv
    │   └── k_selection_summary.json
    │
    ├── clustering/                    # Resultados de clustering
    │   ├── assignments.csv           # ⭐ Principal: atribuições
    │   ├── cluster_sizes.csv
    │   ├── kmeans/
    │   ├── gmm/
    │   └── hierarchical/
    │
    └── insights/                      # Relatórios
        ├── executive_summary.md      # ⭐ Resumo executivo
        ├── cluster_cards.md          # Cards por cluster
        └── cluster_profiles_table.csv
```

## 5. Interpretação dos Resultados

### Elbow Method

- **O que é:** Gráfico mostrando inércia vs número de clusters
- **Como usar:** Identifica "cotovelo" onde a redução de inércia desacelera
- **Importante:** É apenas um filtro, não a resposta final

### Silhouette Score

- **O que é:** Mede quão bem cada ponto se encaixa em seu cluster
- **Valores:** -1 (ruim) a +1 (excelente)
- **Como usar:** Maior é melhor. Compare diferentes valores de k

### GMM Confidence

- **O que é:** Probabilidade de cada ponto pertencer ao cluster atribuído
- **Interpretação:**
  - > 0.7: Alta confiança (cluster bem definido)
  - 0.5-0.7: Confiança moderada
  - < 0.5: Baixa confiança (cluster "misto")

### Clusters Pequenos

- **Atenção:** Clusters com < 3% dos dados podem ser:
  - Nichos interessantes
  - Outliers agrupados
  - Ruído do algoritmo

## 6. Checklist de Sanity Check

Antes de usar os resultados, verifique:

- [ ] **Tamanho dos clusters:** Nenhum cluster muito pequeno (< 1% dos dados)?
- [ ] **Variáveis dominantes:** Alguma variável está dominando o clustering?
  - Verifique se clusters são separados principalmente por uma variável
- [ ] **Outliers:** Há muitos outliers afetando os resultados?
  - Verifique `data_quality/numeric_distributions.png`
- [ ] **Faz sentido de negócio:** Os clusters fazem sentido do ponto de vista de negócio?
  - Revise `insights/cluster_cards.md`
- [ ] **Confiança do GMM:** Muitos pontos com baixa confiança?
  - Verifique `clustering/gmm/confidence_hist.png`

## 7. Próximos Passos

Após análise inicial:

1. **Validação de negócio:** Discutir clusters com equipe de negócio
2. **Análise profunda:** Investigar clusters pequenos ou interessantes
3. **Segmentação:** Desenvolver estratégias por cluster
4. **Iteração:** Ajustar configuração e re-executar se necessário

## 8. Troubleshooting

### Erro: "coluna X não encontrada"
- Verifique se o nome da coluna em `config.yaml` está correto
- Verifique se há espaços extras no nome

### Erro: "menos de 2 colunas úteis"
- Adicione mais colunas numéricas ou categóricas em `config.yaml`
- Verifique se as colunas existem no CSV

### Clusters muito pequenos
- Ajuste `min_cluster_size_pct` em `config.yaml`
- Considere remover outliers mais agressivamente

### Pipeline muito lento
- Reduza `max_points` no dendrograma
- Use menos colunas
- Reduza `initial_k_range`

## Suporte

Para dúvidas, consulte:
- `docs/methodology.md` - Explicação técnica
- `docs/faq.md` - Perguntas frequentes
- `INSTRUCOES_IMPLEMENTACAO.md` - Especificação completa

