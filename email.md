# Referência para E-mail — Resultados dos Novos Experimentos

> **Contexto:** Após o e-mail de 5/março (onde os professores aprovaram o plano de trabalho), foram concluídos três estudos:
> 1. Estudo de ablação do fator de agregação assíncrono (CIFAR-10)
> 2. Generalização para múltiplos datasets: MNIST, Fashion MNIST, CIFAR-10, GTSRB (assíncrono)
> 3. Impacto do timeout no cenário síncrono (CIFAR-10)
>
> **Último e-mail recebido:** Prof. Luiz Fernando Bittencourt, 5/março às 13:24 — "Oi, Julio. Também estou de acordo"

---

## Estudo 1 — Ablação do Fator de Agregação Assíncrono (CIFAR-10)

**Configuração:** 14 experimentos one-at-a-time, variando α, β e γ isoladamente.
Fórmula: `agg_factor = α · β^version · 1/(1 + γ · staleness)`
Dataset: CIFAR-10 | p50 | 40 clientes | 40 atualizações | ~7h total

### Impacto de α (taxa de incorporação) — fixos: β=0.999, γ=0.075

| α   | IID máx. | IID avg_last10 | Non-IID máx. | Non-IID avg_last10 | Non-IID tail_std |
|-----|---------|----------------|-------------|-------------------|-----------------|
| 0.3 | 0.6578  | 0.6569         | 0.4609      | **0.4350**        | 0.0329          |
| 0.5 | 0.6785  | 0.6771         | 0.4581      | 0.3857            | 0.0435          |
| 0.8 | **0.6812** | **0.6800**  | 0.4604      | 0.3760            | **0.0588**      |

**Insight:** Em IID, α tem impacto pequeno (~2%). Em Non-IID, α=0.8 tem o dobro da instabilidade de α=0.3. Valores menores de α funcionam como filtro passa-baixa, suavizando updates heterogêneos.

### Impacto de β (decaimento temporal) — fixos: α=0.8, γ=0.075 ⚠️ PARÂMETRO MAIS CRÍTICO

| β     | IID máx.   | IID avg_last10 | Non-IID máx. | Non-IID avg_last10 |
|-------|------------|----------------|-------------|-------------------|
| 0.999 | **0.6812** | **0.6800**     | **0.4604**  | **0.3760**        |
| 0.99  | 0.4982     | 0.4970         | 0.3685      | 0.3521            |
| 0.95  | 0.3180     | 0.3180         | 0.1684      | 0.1000 (≈ aleatório) |

**Insight:** β=0.95 causa estagnação em ~32% (IID) e colapso para acurácia aleatória (Non-IID). `0.95^100 ≈ 0.006` vs `0.999^100 ≈ 0.905` — o orçamento de aprendizado se esgota rapidamente. **Recomendação: β ≥ 0.999.**

### Impacto de γ (penalização por staleness) — fixos: α=0.8, β=0.999

| γ     | IID máx.   | IID avg_last10 | IID tail_std | Non-IID máx. | Non-IID avg_last10 |
|-------|------------|----------------|-------------|-------------|-------------------|
| 0.0   | 0.6788     | 0.6761         | 0.0098      | 0.2689      | 0.1600            |
| 0.075 | **0.6812** | **0.6800**     | 0.0066      | **0.4604**  | 0.3760            |
| 0.5   | 0.6404     | 0.6394         | 0.0060      | 0.4547      | **0.4379**        |

**Insight:** γ=0 em Non-IID colapsa para ~16% — **falha total**. A penalização por staleness é estruturalmente necessária (não cosmética) para FL assíncrono em cenários heterogêneos. γ=0.5 oferece maior estabilidade com ligeira redução de velocidade.

### IID vs Non-IID — configuração padrão (α=0.8, β=0.999, γ=0.075)

| Métrica                        | IID    | Non-IID |
|-------------------------------|--------|---------|
| Acurácia máxima               | 0.6812 | 0.4604  |
| Acurácia média final (last10) | 0.6800 | 0.3760  |
| Desvio-padrão na cauda        | 0.0066 | 0.0588 (9× maior) |
| Convergência (90% do máx.)    | upd 632 | upd 876 |

### Conclusões C1–C5

- **C1 — β é o parâmetro mais sensível**: valores < 0.999 causam degradação severa; β=0.95 é catastrófico.
- **C2 — γ é indispensável em Non-IID**: sem γ o sistema simplesmente não funciona em cenários heterogêneos.
- **C3 — α tem trade-off estabilidade vs. velocidade**: α=0.3 para robustez; α=0.8 para velocidade em IID.
- **C4 — A fórmula tripartite é validada**: os três componentes têm papéis complementares e não redundantes.
- **C5 — Non-IID amplifica todas as sensibilidades**: parâmetros que "funcionam" em IID podem colapsar em Non-IID.

**Ângulo de inovação:** γ como pré-requisito funcional (não refinamento) é uma contribuição diferencial em relação a FedAsync e FedAsmu, que o tratam como otimização incremental. Extensão natural: γ adaptativo em função do staleness médio corrente.

---

## Estudo 2 — Generalização para Múltiplos Datasets (FL Assíncrono)

**Configuração fixa:** α=0.8, β=0.999, γ=0.075 | p50 | 40 clientes | 1h33min por experimento

### Cenário IID

| Dataset      | Avaliações | Acurácia Máx. | Média Last10 | Tail Std | Conv. @90% | Tempo Total | Loss Final |
|-------------|-----------|---------------|-------------|----------|------------|-------------|------------|
| MNIST       | 4731      | 0.9938        | 0.9937      | 0.0004   | 3/4731     | 1h33min     | 0.0191     |
| Fashion MNIST | 3809    | 0.9185        | 0.9181      | 0.0030   | 348/3809   | 1h33min     | 0.2223     |
| CIFAR-10    | 4645      | 0.7362        | 0.7349      | 0.0108   | 2715/4645  | 1h33min     | 0.7573     |
| GTSRB       | 2534      | 0.9625        | 0.9613      | 0.0023   | 813/2534   | 1h33min     | 0.1531     |

### Cenário Non-IID

| Dataset      | Avaliações | Acurácia Máx. | Média Last10 | Tail Std | Conv. @90% | Tempo Total | Loss Final |
|-------------|-----------|---------------|-------------|----------|------------|-------------|------------|
| MNIST       | 4828      | 0.9859        | 0.9837      | 0.0033   | 953/4828   | 1h33min     | 0.0546     |
| Fashion MNIST | 3860    | 0.8472        | 0.8369      | 0.0278   | 1554/3860  | 1h33min     | 0.4945     |
| CIFAR-10    | 5253      | 0.4765        | 0.4465      | 0.0391   | 4002/5253  | 1h33min     | 1.4638     |
| GTSRB       | 2633      | 0.4899        | 0.2374      | 0.0759   | 2466/2633  | 1h32min     | 2.6594     |

### Impacto IID → Non-IID

| Dataset      | Δ Acurácia Máx. | Δ Avg Last10 | Razão Tail Std (non/iid) |
|-------------|----------------|-------------|--------------------------|
| MNIST       | −0.0079        | −0.0100     | **8.1×**                 |
| Fashion MNIST | −0.0713      | −0.0812     | **9.3×**                 |
| CIFAR-10    | −0.2597        | −0.2884     | **3.6×**                 |
| GTSRB       | **−0.4726**    | **−0.7239** | **33.3×**                |

### Conclusões

1. O comportamento observado no CIFAR-10 **generaliza para todos os datasets**: convergência consistente em IID, queda de acurácia e maior instabilidade em Non-IID.
2. A fórmula de agregação `α·β^v·1/(1+γ·s)` com os parâmetros identificados na ablação é **robusta** — funciona em todos os quatro datasets sem ajuste fino.
3. **GTSRB é o caso mais extremo**: queda de 47.26 p.p. e 33.3× mais instabilidade em Non-IID. Com 43 classes e dados naturalmente desbalanceados por região, reforça a importância do γ adaptativo em datasets multiclasse com alta heterogeneidade.
4. C5 (Non-IID amplifica sensibilidades) se confirma em todos os datasets testados.

---

## Estudo 3 — Impacto do Timeout no FL Síncrono (CIFAR-10)

**Configuração:** CIFAR-10 | 40 clientes | 80 rodadas | 1 época | batch 32
Timeouts derivados via simulação de Monte Carlo (10⁶ realizações de Δt = Δtconn + Δttrain):

| Configuração | Timeout (s) | Significado |
|---|---|---|
| P25 | 24,99 s | Descarta 75% dos clientes lentos |
| P50 | 47,45 s | Descarta 50% dos clientes lentos |
| P75 | 69,95 s | Descarta apenas 25% dos clientes lentos |
| Sem timeout | 95,00 s | Aguarda todos — nenhum descartado |

### Acurácia Final e Máxima — IID

| Configuração | Acurácia R80 | Acurácia Máx. | Loss Final |
|---|---|---|---|
| P25 | 73,7% | 73,8% | 0,7526 |
| P50 | 76,6% | 76,7% | 0,6777 |
| P75 | 77,7% | 78,1% | 0,6511 |
| **Sem timeout** | **79,1%** | **79,2%** | **0,6311** |

### Acurácia Final e Máxima — Non-IID

| Configuração | Acurácia R80 | Acurácia Máx. | Loss Final |
|---|---|---|---|
| P25 | 38,5% | 38,5% | 1,8796 |
| P50 | 40,4% | 49,5% | 1,6215 |
| P75 | 41,5% | 52,5% | 1,5789 |
| **Sem timeout** | **50,6%** | **51,4%** | **1,3255** |

> Em Non-IID, a diferença entre acurácia final e máxima indica instabilidade nas rodadas finais.

### Progressão da Acurácia — IID

| Configuração | Rodada 20 | Rodada 40 | Rodada 60 | Rodada 80 |
|---|---|---|---|---|
| P25 | 56,2% | 66,0% | 70,8% | 73,7% |
| P50 | 60,4% | 70,1% | 74,4% | 76,6% |
| P75 | 64,0% | 72,5% | 75,9% | 77,7% |
| Sem timeout | **67,3%** | **74,5%** | **77,8%** | **79,1%** |

### Tempo de Execução — IID

| Configuração | Tempo/rodada | Tempo total | Ganho vs. sem timeout |
|---|---|---|---|
| P25 | 28,2 s | 37,7 min | −70,4% |
| P50 | 50,7 s | 67,6 min | −48,8% |
| P75 | 73,8 s | 98,5 min | −25,4% |
| Sem timeout | 99,0 s | 132,0 min | — |

### Trade-off: Acurácia × Tempo — IID

| Configuração | Acurácia R80 | Tempo Total | Acurácia/hora |
|---|---|---|---|
| P25 | 73,7% | 37,7 min | 117,3%/h |
| P50 | 76,6% | 67,6 min | 68,0%/h |
| P75 | 77,7% | 98,5 min | 47,3%/h |
| Sem timeout | 79,1% | 132,0 min | 35,9%/h |

**Para atingir 70% (IID):**
- P25: rodada 58 → **27,5 min**
- P50: rodada 40 → **33,9 min**
- P75: rodada 33 → **40,6 min**
- Sem timeout: rodada 25 → **41,2 min**

O P25 atinge 70% em menos tempo absoluto que as demais configurações.

### Estabilidade Nas Últimas 10 Rodadas — Non-IID

| Configuração | Desvio Padrão | Mín. | Máx. |
|---|---|---|---|
| P25 | 0,0559 | 20,7% | 38,5% |
| P50 | 0,0640 | 27,9% | 49,5% |
| P75 | 0,0644 | 32,0% | 52,5% |
| **Sem timeout** | **0,0374** | **40,4%** | **51,4%** |

### Conclusões

1. **O timeout tem impacto significativo e mensurável** em acurácia e tempo.
2. **Em IID**: impacto moderado — P25 converge a 73,7% (−5,4 p.p.) com 3,5× menos tempo.
3. **Em Non-IID**: impacto grave — P25 nunca atinge 40%; diferença de 12,1 p.p. entre P25 e sem-timeout (vs. 5,4 p.p. no IID). Excluir clientes lentos exclui justamente os com dados raros.
4. **Para ambientes móveis** (dado o Prof. Miguel sugerir esse contexto): timeout curto pode excluir sistematicamente clientes com dados sub-representados. Non-IID amplifica o dano.
5. A **simulação de Monte Carlo para calibrar o timeout via percentis** é uma abordagem principiada e controlada.

---

## Rascunho do E-mail

> **Para:** miguel@gta.ufrj.br, bit@unicamp.br
> **Assunto:** [Resultados] Aprendizado federado assíncrono — novos experimentos
> **Thread:** Responder ao e-mail de Luiz de 5/março (13:24)

---

Bom dia, professores!

Conforme combinado, conclui os três estudos que planejamos em 5 de março. Em anexo, envio o relatório atualizado com todos os resultados.

**Estudo de ablação — FL assíncrono (CIFAR-10)**

Conduzi 14 experimentos one-at-a-time variando isoladamente cada parâmetro do fator de agregação (`α · β^v · 1/(1 + γ · s)`). Os resultados confirmam que:

- **β é o parâmetro mais crítico**: valores abaixo de 0,999 causam degradação severa (β=0,95 colapsa para acurácia aleatória em Non-IID). O decaimento exponencial funciona como um "orçamento de aprendizado" que se esgota rapidamente se configurado de forma agressiva.
- **γ é estruturalmente necessário em Non-IID**: sem penalização por staleness (γ=0), o modelo colapsa para ~16% de acurácia — falha total. Isso sugere que γ não é um refinamento incremental, mas um pré-requisito funcional para FL assíncrono em cenários heterogêneos, o que pode ser um ângulo de contribuição em relação à literatura (FedAsync, FedAsmu).
- **α apresenta trade-off estabilidade vs. velocidade**: α=0,8 maximiza acurácia em IID, mas em Non-IID gera 2× mais instabilidade que α=0,3.

**Generalização para múltiplos datasets — FL assíncrono**

Rodei os experimentos com MNIST, Fashion MNIST, CIFAR-10 e GTSRB, mantendo os mesmos hiperparâmetros identificados na ablação (α=0,8, β=0,999, γ=0,075). Os comportamentos observados no CIFAR-10 se confirmaram em todos os datasets:

- Em IID: convergência consistente em todos os casos (MNIST: 99,4%, GTSRB: 96,3%).
- Em Non-IID: queda de acurácia e aumento de instabilidade são padrões universais.
- O **GTSRB é o caso mais extremo**: queda de 47,3 p.p. e instabilidade 33,3× maior em Non-IID, provavelmente por conta das 43 classes e da heterogeneidade natural das placas de trânsito por região.

**Impacto do timeout — FL síncrono (CIFAR-10)**

Implementei a estratégia de Monte Carlo para derivar os timeouts em segundos (conforme sugestão do Prof. Miguel). Os resultados revelam um trade-off claro entre acurácia e tempo de execução:

- Em **IID**: timeout P25 entrega 73,7% com 3,5× menos tempo que sem timeout (79,1%). Em termos de acurácia por hora, P25 (117,3%/h) supera amplamente o sem-timeout (35,9%/h).
- Em **Non-IID**: o impacto é muito mais severo — P25 nunca atinge 40% e o sem-timeout é o único que ultrapassa 50%. A diferença de 12,1 p.p. entre P25 e sem-timeout (vs. 5,4 p.p. no IID) confirma que excluir clientes lentos em cenários heterogêneos prejudica a generalização, pois esses clientes tendem a ter os dados mais raros.

Para ambientes dinâmicos como o móvel (cenário mencionado pelo Prof. Miguel), o timeout deve ser escolhido com cuidado: um timeout muito curto pode excluir sistematicamente clientes com dados sub-representados.

Fico à disposição para eventuais dúvidas ou ajustes. Podemos agendar uma conversa para discutir os próximos passos quando for conveniente para vocês.

Atenciosamente,
Julio
