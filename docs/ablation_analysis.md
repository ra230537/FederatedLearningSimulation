# Análise do Estudo de Ablação — FL Assíncrono com CIFAR-10

## Resumo do Experimento

14 experimentos one-at-a-time, variando isoladamente cada parâmetro do fator de agregação:

```
agg_factor = α · β^version · 1/(1 + γ · staleness)
```

Configuração: p50, 40 clientes, 40 atualizações por experimento, ~7 horas de execução total (~1150–1175 avaliações registradas por experimento).

---

## 1. Impacto do Base Alpha (α) — Taxa de incorporação

Fixos: β=0.999, γ=0.075

| α   | IID max | IID avg_last10 | Non-IID max | Non-IID avg_last10 | Non-IID tail_std |
|-----|---------|----------------|-------------|-------------------|-----------------|
| 0.3 | 0.6578  | 0.6569         | 0.4609      | **0.4350**        | 0.0329          |
| 0.5 | 0.6785  | 0.6771         | 0.4581      | 0.3857            | 0.0435          |
| 0.8 | **0.6812** | **0.6800**  | 0.4604      | 0.3760            | **0.0588**      |

### IID
O impacto de α é pequeno (~2% entre o pior e o melhor). Valores mais altos convergem marginalmente mais rápido e atingem acurácia ligeiramente superior. Sob dados homogêneos, o modelo é robusto a essa variação.

### Non-IID
Os três valores atingem picos similares (~0.46), mas a **estabilidade diverge drasticamente**. α=0.8 tem o dobro do desvio-padrão de α=0.3 na cauda (0.059 vs 0.033). Valores altos de α amplificam as oscilações porque incorporam com muito peso atualizações de clientes com distribuições de dados enviesadas. **α mais baixo atua como um filtro passa-baixa**, suavizando o impacto de updates heterogêneos.

---

## 2. Impacto do Decay (β) — Decaimento temporal ⚠️ PARÂMETRO MAIS CRÍTICO

Fixos: α=0.8, γ=0.075

| β     | IID max    | IID avg_last10 | Non-IID max | Non-IID avg_last10 |
|-------|------------|----------------|-------------|-------------------|
| 0.999 | **0.6812** | **0.6800**     | **0.4604**  | **0.3760**        |
| 0.99  | 0.4982     | 0.4970         | 0.3685      | 0.3521            |
| 0.95  | 0.3180     | 0.3180         | 0.1684      | 0.1000            |

### IID
β é o parâmetro com maior impacto na acurácia final. β=0.95 causa **estagnação em ~32%** — o fator de agregação decai rápido demais e o modelo para de aprender. Com β=0.99, já há uma perda de ~18 pontos percentuais em relação a β=0.999. Nos gráficos, β=0.95 e β=0.99 formam platôs bem visíveis, enquanto β=0.999 continua crescendo.

### Non-IID
O efeito é ainda mais severo. β=0.95 **colapsa para acurácia aleatória** (0.10 = 1/10 classes), efetivamente anulando o aprendizado. Isso ocorre porque o decaimento agressivo reduz o fator de agregação a valores próximos de zero antes que os clientes com dados heterogêneos consigam contribuir o suficiente para a convergência.

### Interpretação
O decaimento exponencial funciona como um "orçamento de aprendizado". β=0.95 esgota esse orçamento cedo demais — após ~100 atualizações, `0.95^100 ≈ 0.006`, ou seja, o modelo quase ignora novos updates. Já `0.999^100 ≈ 0.905`, preservando capacidade de aprendizado por muito mais tempo.

---

## 3. Impacto do Tardiness Sensitivity (γ) — Penalização por staleness

Fixos: α=0.8, β=0.999

| γ     | IID max    | IID avg_last10 | IID tail_std | Non-IID max | Non-IID avg_last10 |
|-------|------------|----------------|-------------|-------------|-------------------|
| 0.0   | 0.6788     | 0.6761         | 0.0098      | 0.2689      | 0.1600            |
| 0.075 | **0.6812** | **0.6800**     | 0.0066      | **0.4604**  | 0.3760            |
| 0.5   | 0.6404     | 0.6394         | 0.0060      | 0.4547      | **0.4379**        |

### IID
As três configurações convergem para acurácias próximas (0.64–0.68). γ=0.075 é marginalmente melhor. Sob dados homogêneos, mesmo updates desatualizados ainda carregam informação útil, então a penalização tem pouco efeito.

### Non-IID
Este é o resultado mais revelador do estudo:
- **γ=0.0 (sem penalização) colapsa para ~16%** — praticamente falha total. Sem penalização, updates muito defasados (de clientes que treinaram em apenas 2–3 classes) são incorporados com o mesmo peso de updates recentes, **corrompendo o modelo global**.
- **γ=0.075 alcança ~46%** — o modelo aprende, mas com oscilações significativas na cauda.
- **γ=0.5 alcança ~45% com a maior estabilidade** (avg_last10=0.44, a mais alta entre os três).

### Interpretação
A penalização por staleness é **essencial para cenários não-IID**. Ela age como um mecanismo de controle de qualidade: updates de clientes lentos (que tendem a ser os com distribuições mais enviesadas no cenário assíncrono) recebem peso reduzido, protegendo o modelo global. Penalização mais forte (γ=0.5) troca velocidade de convergência por estabilidade.

---

## 4. Comparação IID vs Non-IID

Configuração padrão: α=0.8, β=0.999, γ=0.075

| Métrica                          | IID    | Non-IID |
|----------------------------------|--------|---------|
| Acurácia máxima                  | 0.6812 | 0.4604  |
| Acurácia média final (last 10)   | 0.6800 | 0.3760  |
| Desvio-padrão na cauda           | 0.0066 | 0.0588  |
| Convergência (90% do máx.)       | update 632 | update 876 |

Non-IID é fundamentalmente mais difícil: acurácia ~32% menor, convergência ~39% mais lenta, e **9x mais instável**. Isso é esperado e consistente com a literatura (FedAvg, FedAsync).

---

## 5. Conclusões Principais

### C1 — O decay (β) é o parâmetro mais sensível da fórmula
Valores abaixo de 0.999 causam degradação significativa, e β=0.95 é catastrófico em ambos os cenários. O decaimento exponencial precisa ser muito suave para não "silenciar" o aprendizado prematuramente. **Recomendação: manter β ≥ 0.999.**

### C2 — A penalização por staleness (γ) é dispensável em IID mas indispensável em Non-IID
Sem ela (γ=0), o modelo não converge em cenários heterogêneos. Isso demonstra empiricamente que a componente de staleness não é um refinamento cosmético — é **estruturalmente necessária** para que a agregação assíncrona funcione em cenários realistas.

### C3 — O base alpha (α) tem trade-off estabilidade vs. velocidade
Particularmente em Non-IID: valores menores convergem mais devagar mas com menos oscilação. A escolha ótima depende da aplicação:
- Priorizar robustez → α=0.3
- Priorizar velocidade com dados IID → α=0.8

### C4 — A fórmula tripartite é validada
A combinação α · β^t · 1/(1+γ·s) com os valores padrão (0.8, 0.999, 0.075) produz os melhores resultados ou próximo dos melhores em ambos os cenários. Os três componentes têm papéis complementares e não redundantes, como comprovado pela ablação.

### C5 — Cenários Non-IID amplificam todas as sensibilidades
Parâmetros que "funcionam" em IID podem colapsar em Non-IID (ex: γ=0 e β=0.95). Isso reforça a importância de sempre avaliar sob ambas as distribuições.

---

## 6. Possível Ângulo de Inovação

O resultado de que **γ é o fator determinante para viabilidade em Non-IID** pode ser explorado como contribuição científica. Na literatura (FedAsync, Fedasmu), a penalização por staleness é frequentemente apresentada como uma otimização incremental. Os dados deste estudo mostram que, na verdade, ela é um **pré-requisito funcional** — sem ela o sistema simplesmente não funciona em cenários heterogêneos.

Uma extensão natural seria investigar se existe um valor ótimo de γ que varie **adaptativamente** (em vez de ser fixo), por exemplo em função do staleness médio corrente ou da distribuição estimada dos dados dos clientes.

---

## Apêndice — Dados Brutos

| Arquivo | max_acc | avg_last10 | conv@90%max | tail_std |
|---------|---------|-----------|-------------|---------|
| iid_A0.3_B0.999_C0.075 | 0.6578 | 0.6569 | 605/1140 | 0.0059 |
| iid_A0.5_B0.999_C0.075 | 0.6785 | 0.6771 | 616/1148 | 0.0065 |
| iid_A0.8_B0.95_C0.075  | 0.3180 | 0.3180 | 6/1161   | 0.0000 |
| iid_A0.8_B0.999_C0.075 | 0.6812 | 0.6800 | 632/1175 | 0.0066 |
| iid_A0.8_B0.999_C0.0   | 0.6788 | 0.6761 | 718/1156 | 0.0098 |
| iid_A0.8_B0.999_C0.5   | 0.6404 | 0.6394 | 579/1112 | 0.0060 |
| iid_A0.8_B0.99_C0.075  | 0.4982 | 0.4970 | 211/1158 | 0.0000 |
| non_iid_A0.3_B0.999_C0.075 | 0.4609 | 0.4350 | 731/1175  | 0.0329 |
| non_iid_A0.5_B0.999_C0.075 | 0.4581 | 0.3857 | 825/1175  | 0.0435 |
| non_iid_A0.8_B0.95_C0.075  | 0.1684 | 0.1000 | 14/1177   | 0.0000 |
| non_iid_A0.8_B0.999_C0.075 | 0.4604 | 0.3760 | 876/1175  | 0.0588 |
| non_iid_A0.8_B0.999_C0.0   | 0.2689 | 0.1600 | 951/1105  | 0.0460 |
| non_iid_A0.8_B0.999_C0.5   | 0.4547 | 0.4379 | 739/1116  | 0.0379 |
| non_iid_A0.8_B0.99_C0.075  | 0.3685 | 0.3521 | 310/1175  | 0.0001 |

*conv@90%max = número de atualizações para atingir 90% da acurácia máxima*
*tail_std = desvio-padrão nos últimos 20% das atualizações*
