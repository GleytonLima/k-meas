# Perguntas Frequentes (FAQ)

## Configuração

### Q: Como adicionar mais colunas ao clustering?

**A:** Edite `configs/config.yaml` e adicione as colunas nas seções `numeric` ou `categorical`:

```yaml
columns:
  numeric:
    - valor_emprestimo
    - nova_coluna_numerica  # Adicione aqui
  categorical:
    - nome_produto
    - nova_coluna_categorica  # Adicione aqui
```

### Q: Posso usar apenas algumas colunas?

**A:** Sim! Liste apenas as colunas que deseja usar em `config.yaml`. Colunas não listadas serão ignoradas (exceto as em `drop`).

### Q: Como mudar o número de clusters?

**A:** O pipeline seleciona k automaticamente, mas você pode:
- Ajustar `initial_k_range` para testar diferentes intervalos
- Usar `manual_candidate_k_range` se `candidate_k_range: "manual"`

## Execução

### Q: O pipeline está muito lento. Como acelerar?

**A:** 
- Reduza `max_points` no dendrograma (hierárquico)
- Use menos colunas
- Reduza `initial_k_range`
- Desabilite métodos não essenciais em `methods`

### Q: Posso executar sem Docker?

**A:** Sim! Veja seção "Desenvolvimento" no README principal.

### Q: Como executar com diferentes configurações?

**A:** Crie múltiplos arquivos de config (ex: `config_rapido.yaml`) e execute:
```bash
docker compose run --rm pipeline --config configs/config_rapido.yaml
```

## Resultados

### Q: Qual arquivo é o mais importante?

**A:** 
- `clustering/assignments.csv`: Atribuições de cluster para cada amostra
- `insights/executive_summary.md`: Resumo executivo

### Q: Como interpretar o Silhouette Score?

**A:**
- > 0.5: Boa separação
- 0.25-0.5: Separação razoável
- < 0.25: Separação fraca (considere ajustar parâmetros)

### Q: O que significa "baixa confiança" no GMM?

**A:** Probabilidade < 0.5 de pertencer ao cluster atribuído. Pode indicar:
- Ponto está entre clusters (região de fronteira)
- Número de clusters pode não ser ideal
- Ponto é outlier

### Q: Por que alguns clusters são muito pequenos?

**A:** Pode ser:
- **Nicho legítimo:** Segmento pequeno mas válido
- **Outliers agrupados:** Pontos anômalos
- **K muito alto:** Muitos clusters para o tamanho do dataset

**Ação:** Analise `cluster_cards.md` para entender características.

## Problemas Comuns

### Q: Erro "coluna X não encontrada"

**A:** 
- Verifique se o nome da coluna está correto (case-sensitive)
- Verifique se há espaços extras
- Confirme que a coluna existe no CSV

### Q: Erro "menos de 2 colunas úteis"

**A:**
- Adicione mais colunas em `config.yaml`
- Verifique se as colunas listadas existem no CSV
- Remova colunas de `drop` se necessário

### Q: Clusters não fazem sentido

**A:**
- Verifique se há variáveis dominantes (ex: apenas "valor" separando)
- Considere remover ou transformar variáveis muito dominantes
- Teste diferentes valores de k
- Valide com conhecimento de negócio

### Q: Gráficos não aparecem

**A:**
- Verifique permissões da pasta `outputs/`
- Verifique logs em `outputs/[run]/logs.txt`
- Confirme que matplotlib está instalado

### Q: Docker não funciona

**A:**
- Verifique se Docker está rodando
- Tente `docker compose build --no-cache`
- Verifique se há espaço em disco suficiente

## Interpretação

### Q: Devo usar K-Means ou GMM?

**A:** 
- **K-Means:** Mais rápido, bom para baseline
- **GMM:** Mais rico, fornece probabilidades, melhor para insights finais
- **Recomendação:** Use ambos e compare

### Q: Quantos clusters devo ter?

**A:** Não há resposta única. Considere:
- Métricas (Silhouette, BIC)
- Interpretabilidade de negócio
- Tamanho mínimo útil (ex: > 3% dos dados)
- Objetivo da análise

### Q: Como validar se os clusters são bons?

**A:**
1. **Métricas internas:** Silhouette, CH, DB
2. **Validação de negócio:** Fazem sentido?
3. **Estabilidade:** Re-executar com `random_state` diferente
4. **Tamanho:** Clusters muito pequenos podem ser problemáticos

## Próximos Passos

### Q: O que fazer após o clustering?

**A:**
1. Analisar `cluster_cards.md` para entender cada cluster
2. Validar com equipe de negócio
3. Desenvolver estratégias por cluster
4. Implementar segmentação
5. Monitorar e iterar

### Q: Posso usar os clusters em produção?

**A:** 
- Sim, mas valide primeiro
- Considere retreinar periodicamente
- Monitore qualidade dos clusters ao longo do tempo
- Documente decisões e parâmetros

## Suporte

### Q: Onde encontrar mais ajuda?

**A:**
- `docs/analyst_guide.md`: Guia completo do analista
- `docs/methodology.md`: Explicação técnica
- `INSTRUCOES_IMPLEMENTACAO.md`: Especificação completa
- Logs em `outputs/[run]/logs.txt`

---

**Não encontrou sua resposta?** Consulte a documentação completa ou entre em contato com a equipe de desenvolvimento.

