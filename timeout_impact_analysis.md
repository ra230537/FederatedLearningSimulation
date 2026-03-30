# Análise do Impacto do Timeout no Aprendizado Federado Síncrono

> Experimento executado com `python main.py --include-no-timeout`
> Dataset: **CIFAR-10** | Clientes: **40** | Rodadas: **80** | Épocas locais: **1** | Batch: **32**

---

## 1. Contexto e Motivação

O Prof. Miguel sugeriu investigar o impacto do timeout no cenário síncrono, sobretudo pensando em ambientes dinâmicos como o de dispositivos móveis. Em cenários síncronos, o servidor espera todos os clientes selecionados antes de agregar — o timeout define quanto tempo máximo o servidor aguarda por cada cliente antes de descartá-lo na rodada.

Os timeouts foram derivados via **simulação de Monte Carlo** a partir das distribuições de tempo de conexão (U[0, 5]s) e de treino (U[0, 90]s):

| Configuração | Timeout (s) | Significado |
|---|---|---|
| P25 | 24,99 s | Descarta 75% dos clientes lentos |
| P50 | 47,45 s | Descarta 50% dos clientes lentos |
| P75 | 69,95 s | Descarta apenas 25% dos clientes lentos |
| Sem timeout | 95,00 s | Aguarda todos — nenhum cliente é descartado |

---

## 2. Resultados — Cenário IID

### 2.1 Acurácia Final e Máxima

| Configuração | Acurácia R80 | Acurácia Máx. | Loss Final |
|---|---|---|---|
| P25 (24,99 s) | 73,7% | 73,8% | 0,7526 |
| P50 (47,45 s) | 76,6% | 76,7% | 0,6777 |
| P75 (69,95 s) | 77,7% | 78,1% | 0,6511 |
| **Sem timeout (95 s)** | **79,1%** | **79,2%** | **0,6311** |

### 2.2 Progressão da Acurácia (IID)

| Configuração | Rodada 20 | Rodada 40 | Rodada 60 | Rodada 80 |
|---|---|---|---|---|
| P25 | 56,2% | 66,0% | 70,8% | 73,7% |
| P50 | 60,4% | 70,1% | 74,4% | 76,6% |
| P75 | 64,0% | 72,5% | 75,9% | 77,7% |
| Sem timeout | **67,3%** | **74,5%** | **77,8%** | **79,1%** |

### 2.3 Velocidade de Convergência (IID) — Rodada em que atinge o limiar

| Configuração | 40% | 50% | 60% | 70% |
|---|---|---|---|---|
| P25 | R10 (287s) | R16 (465s) | R28 (803s) | R58 (1648s) |
| P50 | R6 (307s) | R11 (560s) | R20 (1018s) | R40 (2037s) |
| P75 | R6 (444s) | R9 (665s) | R16 (1183s) | R33 (2437s) |
| Sem timeout | **R4 (393s)** | **R7 (691s)** | **R12 (1184s)** | **R25 (2471s)** |

### 2.4 Tempo de Execução (IID)

| Configuração | Tempo/rodada (médio) | Tempo total | Ganho vs. sem timeout |
|---|---|---|---|
| P25 | 28,2 s | 37,7 min | −70,4% de tempo |
| P50 | 50,7 s | 67,6 min | −48,8% de tempo |
| P75 | 73,8 s | 98,5 min | −25,4% de tempo |
| Sem timeout | 99,0 s | 132,0 min | — |

---

## 3. Resultados — Cenário Não-IID

### 3.1 Acurácia Final e Máxima

| Configuração | Acurácia R80 | Acurácia Máx. | Loss Final |
|---|---|---|---|
| P25 (24,99 s) | 38,5% | 38,5% | 1,8796 |
| P50 (47,45 s) | 40,4% | 49,5% | 1,6215 |
| P75 (69,95 s) | 41,5% | 52,5% | 1,5789 |
| **Sem timeout (95 s)** | **50,6%** | **51,4%** | **1,3255** |

> **Nota:** No Não-IID, a diferença entre acurácia final e máxima indica instabilidade nas rodadas finais — o modelo oscila em vez de convergir monotonicamente.

### 3.2 Progressão da Acurácia (Não-IID)

| Configuração | Rodada 20 | Rodada 40 | Rodada 60 | Rodada 80 |
|---|---|---|---|---|
| P25 | 18,4% | 19,0% | 21,4% | 38,5% |
| P50 | 15,5% | 35,3% | 32,9% | 40,4% |
| P75 | 29,2% | 33,3% | 38,9% | 41,5% |
| Sem timeout | **30,3%** | **30,7%** | **37,9%** | **50,6%** |

### 3.3 Velocidade de Convergência (Não-IID)

| Configuração | 20% | 30% | 40% | 50% |
|---|---|---|---|---|
| P25 | R25 (704s) | R41 (1151s) | nunca | nunca |
| P50 | R17 (853s) | R26 (1293s) | R50 (2471s) | nunca |
| P75 | R4 (288s) | R28 (2003s) | R43 (3076s) | R78 (5578s) |
| Sem timeout | R6 (577s) | R17 (1639s) | **R47 (4534s)** | **R47 (4534s)** |

### 3.4 Estabilidade nas Últimas 10 Rodadas (Não-IID)

| Configuração | Desvio Padrão | Mín. | Máx. |
|---|---|---|---|
| P25 | 0,0559 | 20,7% | 38,5% |
| P50 | 0,0640 | 27,9% | 49,5% |
| P75 | 0,0644 | 32,0% | 52,5% |
| **Sem timeout** | **0,0374** | **40,4%** | **51,4%** |

---

## 4. Análise do Trade-off: Acurácia × Velocidade

### IID — Acurácia por Hora de Treino

A relação entre acurácia e tempo real de parede revela o trade-off central:

| Configuração | Acurácia R80 | Tempo Total | Acurácia/hora |
|---|---|---|---|
| P25 | 73,7% | 37,7 min | 117,3%/h |
| P50 | 76,6% | 67,6 min | 68,0%/h |
| P75 | 77,7% | 98,5 min | 47,3%/h |
| Sem timeout | 79,1% | 132,0 min | 35,9%/h |

**Observação:** O P25 é 3,3× mais rápido por rodada que o sem-timeout, mas entrega menos 5,4 p.p. de acurácia ao final. O sem-timeout entrega a melhor acurácia absoluta, mas ao custo de 3,5× mais tempo total.

### Ponto de equilíbrio temporal

Para atingir **70% de acurácia (IID)**:
- P25: rodada 58 → **27,5 min**
- P50: rodada 40 → **33,9 min**
- P75: rodada 33 → **40,6 min**
- Sem timeout: rodada 25 → **41,2 min**

Ou seja, a configuração P25 atinge 70% de acurácia em menos tempo absoluto (em rodadas de parede) que as demais, apesar da acurácia final inferior.

---

## 5. Interpretação para Cenários Dinâmicos (Mobile)

A motivação original da análise é entender o comportamento em ambientes com alta variabilidade de disponibilidade de clientes — como dispositivos móveis:

**Timeout agressivo (P25):**
- Rodadas rápidas (~28s), mas descartar 75% dos clientes lentos significa que cada rodada usa poucos clientes participantes.
- Em IID, isso é suficiente para boa convergência (73,7% final), mas mais lento em termos de rodadas até convergência.
- Em Não-IID, o desempenho é severamente degradado (38,5% final, nunca atinge 40%).
- **Indicado quando:** latência importa mais que acurácia, ou clientes lentos têm dados redundantes.

**Timeout moderado (P50):**
- Equilíbrio razoável: 76,6% IID e 40,4% Não-IID, com metade do tempo do sem-timeout.
- **Indicado quando:** há restrição de tempo mas dados heterogêneos estão presentes.

**Sem timeout (95s):**
- Melhor acurácia em ambos os cenários (79,1% IID / 50,6% Não-IID).
- Em IID, a estabilidade final é excelente (std=0,0010).
- Em Não-IID, ainda há oscilações, mas a convergência é superior.
- **Indicado quando:** todos os clientes têm dados relevantes e o tempo de treino não é crítico.

**Impacto no Não-IID (cenário crítico para mobile):**
O Não-IID simula a heterogeneidade real dos dados de usuários móveis (cada dispositivo gera dados de contexto diferente). Nesse cenário, excluir clientes lentos (que podem ser justamente os que têm dados raros) prejudica significativamente a generalização do modelo. A diferença entre P25 (38,5%) e sem-timeout (50,6%) é de **+12,1 p.p.**, muito mais pronunciada que no IID (+5,4 p.p.).

---

## 6. Conclusões

1. **O timeout tem impacto significativo e mensurável** tanto em acurácia quanto em tempo de treinamento no cenário síncrono com CIFAR-10.

2. **No IID**, o impacto é moderado: mesmo o timeout mais agressivo (P25) converge para 73,7%, uma redução de 5,4 p.p. em relação ao sem-timeout, mas com 3,5× menos tempo total.

3. **No Não-IID**, o impacto é grave: o P25 nunca atinge 40% de acurácia, e o sem-timeout é o único que ultrapassa 50%. A heterogeneidade dos dados amplia o dano causado por timeouts curtos.

4. **Para cenários móveis** (alta variabilidade de conectividade e dados heterogêneos), o timeout deve ser escolhido com cuidado: um timeout muito curto pode excluir sistematicamente clientes com dados sub-representados, prejudicando a generalização.

5. **A simulação de Monte Carlo para definir o timeout via percentis** se mostrou uma abordagem adequada para explorar esse espaço de forma principiada, permitindo comparações controladas.

---

## 7. Parâmetros da Simulação

| Parâmetro | Valor |
|---|---|
| Dataset | CIFAR-10 |
| Número de clientes | 40 |
| Rodadas de treinamento | 80 |
| Épocas locais | 1 |
| Batch size | 32 |
| Tempo de conexão | U[0, 5] s |
| Tempo de treino | U[0, 90] s |
| Percentis avaliados | P25, P50, P75, Sem timeout |
| Seed | 42 |
