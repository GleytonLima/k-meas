# Metodologia Técnica

## Visão Geral

Este pipeline implementa uma abordagem sistemática para clustering de dados de empréstimos PJ, combinando múltiplos algoritmos e métricas para gerar insights acionáveis.

## Pré-processamento

### 1. Tratamento de Valores Faltantes

- **Numéricas:** Preenchimento com mediana (robusto a outliers) ou média
- **Categóricas:** Preenchimento com valor constante "MISSING"

### 2. Transformações

- **Log1p:** Aplicado a variáveis com distribuição assimétrica (ex: valores monetários)
- Reduz impacto de outliers e normaliza distribuições

### 3. Tratamento de Outliers

- **Winsorização:** Limita valores extremos aos percentis 1% e 99%
- Preserva informação sem permitir que outliers dominem o clustering

### 4. Encoding

- **One-Hot Encoding:** Para variáveis categóricas
- Categorias raras (< 1% por padrão) são agrupadas em "RARE"

### 5. Scaling

- **RobustScaler (padrão):** Usa mediana e IQR, robusto a outliers
- **StandardScaler (opcional):** Usa média e desvio padrão

## Seleção de K

### Elbow Method

- Calcula inércia para k de 2 a 15 (configurável)
- Identifica "cotovelo" onde redução de inércia desacelera
- Usado como filtro para sugerir intervalo de k

### Métricas de Qualidade

- **Silhouette Score:** Mede separação entre clusters (-1 a +1, maior é melhor)
- **Calinski-Harabasz:** Razão entre variância entre/interna clusters (maior é melhor)
- **Davies-Bouldin:** Média de similaridade entre clusters (menor é melhor)

### Seleção Automática

- K-Means: Escolhe k com maior Silhouette Score no intervalo sugerido
- GMM: Escolhe k com menor BIC (Bayesian Information Criterion) ou AIC

## Algoritmos de Clustering

### K-Means

- **Vantagens:** Rápido, interpretável, escalável
- **Limitações:** Assume clusters esféricos, sensível a inicialização
- **Uso:** Baseline e comparação

### Gaussian Mixture Model (GMM)

- **Vantagens:** 
  - Identifica clusters não-esféricos
  - Fornece probabilidades (confiança)
  - Seleção automática de k via BIC/AIC
- **Limitações:** Mais lento que K-Means
- **Uso:** Método principal para insights

### Clustering Hierárquico

- **Vantagens:** 
  - Dendrograma mostra estrutura hierárquica
  - Não requer k pré-definido (mas usamos k sugerido)
- **Limitações:** Computacionalmente caro para grandes datasets
- **Uso:** Validação e storytelling

## Visualização

### Embedding 2D

- **PCA (Principal Component Analysis):** Reduz dimensionalidade para 2D
- **Nota importante:** Embedding é apenas para visualização; clustering é feito no espaço completo

### Interpretação de Gráficos

- **Scatter plots:** Mostram separação visual dos clusters
- **Elbow plots:** Ajudam a identificar número ótimo de clusters
- **Dendrogramas:** Mostram hierarquia de agrupamento
- **Perfis:** Mostram características médias por cluster

## Validação

### Métricas Internas

- Não requer labels verdadeiros
- Baseadas em separação e compactação dos clusters

### Validação de Negócio

- **Essencial:** Validar se clusters fazem sentido do ponto de vista de negócio
- **Checklist:**
  - Clusters representam segmentos distintos?
  - Há padrões interpretáveis?
  - Clusters pequenos são nichos ou ruído?

## Limitações e Considerações

1. **Dimensionalidade:** Alta dimensionalidade pode afetar qualidade
2. **Escalabilidade:** GMM e hierárquico podem ser lentos para datasets muito grandes
3. **Inicialização:** K-Means pode convergir para mínimos locais
4. **Assumções:** Cada algoritmo faz suposições sobre forma dos clusters

## Boas Práticas

1. **Explorar dados primeiro:** Use visualizações de qualidade
2. **Testar múltiplos k:** Não confie apenas em uma métrica
3. **Validar com negócio:** Clusters devem fazer sentido
4. **Documentar decisões:** Anote escolhas de k e parâmetros
5. **Iterar:** Ajuste configuração baseado em resultados

## Referências

- Scikit-learn documentation: https://scikit-learn.org/
- "Pattern Recognition and Machine Learning" - Bishop (GMM)
- "Introduction to Data Mining" - Tan et al. (Clustering)

