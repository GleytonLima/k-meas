# Exemplos de Uso dos Scripts

Este arquivo contém exemplos práticos de como usar os scripts com diferentes parâmetros.

## 📋 Usando Arquivo de Configuração (config.json)

A forma mais prática de configurar os scripts é usando um arquivo `config.json`:

### Criar config.json

```bash
# Copiar exemplo
cp config.exemplo.json config.json

# Ou criar manualmente editando config.json
```

### Exemplo de config.json

```json
{
  "kmeans_emprestimos": {
    "k": 4,
    "pares_visualizacao": [
      ["valor_emprestimo", "quantidade_parcelas"],
      ["score_cliente", "taxa_juros_anual"]
    ],
    "features": ["valor_emprestimo", "quantidade_parcelas", "taxa_juros_anual"],
    "arquivo_entrada": "dados_emprestimos.csv",
    "arquivo_saida": "resultados_clusters.csv"
  }
}
```

### Usar config.json

```bash
# Usar config.json padrão
python kmeans_emprestimos.py

# Usar config customizado
python kmeans_emprestimos.py --config meu_config.json

# Sobrescrever k do config via CLI
python kmeans_emprestimos.py --config config.json --k 5
```

**Nota:** Parâmetros CLI sempre têm prioridade sobre o config.json

## 📊 Análise do K Ideal (`analisar_k_ideal.py`)

### Uso Básico
```bash
python analisar_k_ideal.py
```
Testa k de 2 a 10 (padrão) e gera `analise_k_ideal.png`

### Range Customizado
```bash
python analisar_k_ideal.py --k-min 3 --k-max 15
```
Testa k de 3 a 15

### Com Arquivo Diferente
```bash
python analisar_k_ideal.py --arquivo meus_dados.csv --k-min 2 --k-max 8
```

### Features Customizadas
```bash
python analisar_k_ideal.py --features valor_emprestimo quantidade_parcelas taxa_juros_anual
```

## 🎯 K-Means Clustering (`kmeans_emprestimos.py`)

### Uso Básico (k automático)
```bash
python kmeans_emprestimos.py
```
Determina k automaticamente e usa pares padrão de visualização

### Especificar k Manualmente
```bash
python kmeans_emprestimos.py --k 3
```
Usa k=3 para o clustering

### Especificar k e Pares de Visualização
```bash
python kmeans_emprestimos.py --k 4 --pares "valor_emprestimo,quantidade_parcelas;score_cliente,taxa_juros_anual"
```

### Múltiplos Pares (separados por espaço)
```bash
python kmeans_emprestimos.py --k 5 --pares "valor_emprestimo,quantidade_parcelas renda_mensal,valor_parcela"
```

### Busca Automática com Range Customizado
```bash
python kmeans_emprestimos.py --k-range 3-12
```
Busca o melhor k entre 3 e 12

### Features Customizadas
```bash
python kmeans_emprestimos.py --k 4 --features valor_emprestimo quantidade_parcelas score_cliente
```

### Arquivos Customizados
```bash
python kmeans_emprestimos.py --arquivo meus_dados.csv --saida meus_resultados.csv --k 3
```

## 🔄 Fluxo Completo Recomendado

### 1. Gerar Dados
```bash
python gerar_dados.py
```

### 2. Analisar K Ideal
```bash
python analisar_k_ideal.py --k-min 2 --k-max 10
```

### 3. Aplicar K-Means com k Escolhido
```bash
python kmeans_emprestimos.py --k 4 --pares "valor_emprestimo,quantidade_parcelas;score_cliente,taxa_juros_anual;renda_mensal,valor_parcela"
```

## 📋 Features Disponíveis

As seguintes features estão disponíveis no dataset padrão:

- `valor_emprestimo`
- `quantidade_parcelas`
- `taxa_juros_anual`
- `score_cliente`
- `idade_cliente`
- `renda_mensal`
- `valor_parcela`
- `parcela_sobre_renda`
- `prazo_dias`

## 💡 Dicas

1. **Escolha do k**: Use `analisar_k_ideal.py` primeiro para visualizar as métricas e escolher o k ideal
2. **Pares de Visualização**: Escolha pares que façam sentido para sua análise. Exemplos:
   - `valor_emprestimo,quantidade_parcelas` - Relação valor vs prazo
   - `score_cliente,taxa_juros_anual` - Relação risco vs taxa
   - `renda_mensal,valor_parcela` - Capacidade de pagamento
   - `valor_emprestimo,parcela_sobre_renda` - Impacto na renda

3. **Número de Pares**: Você pode especificar quantos pares quiser. O layout será ajustado automaticamente.

4. **Validação**: O script valida se as features existem no dataset antes de usar.

