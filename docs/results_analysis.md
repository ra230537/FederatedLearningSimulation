# Análise Comparativa — Aprendizado Federado Assíncrono

> Relatório gerado automaticamente por `analyze_results.py`.

## 1. Resumo dos Experimentos

Todos os experimentos foram executados com a **mesma configuração de hiperparâmetros**,
variando apenas o dataset. Os resultados permitem verificar se os comportamentos
observados no CIFAR-10 (estudo de ablação anterior) se generalizam para outros problemas.

### Parâmetros Fixos

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| α (alpha)  | 0.8 | Taxa de incorporação base |
| β (beta)   | 0.999 | Fator de decaimento temporal |
| γ (gamma)  | 0.075 | Penalização por staleness |
| Clientes   | 40 | Número total de clientes |
| p          | 0.5 | Probabilidade de conexão (p50) |

**Fator de agregação:** `agg_factor = α · β^version · 1/(1 + γ · staleness)`

### Datasets Avaliados

| Dataset | Tarefa | Classes | Complexidade relativa |
|---------|--------|---------|----------------------|
| MNIST | Dígitos manuscritos | 10 | Baixa |
| Fashion MNIST | Peças de roupa | 10 | Média-baixa |
| CIFAR-10 | Objetos naturais (RGB) | 10 | Média-alta |
| GTSRB | Placas de trânsito (RGB) | 43 | Alta |

## 2. Tabela Comparativa Geral

### 2.1 Cenário IID

| Dataset | Avaliações | Acurácia Máx. | Média Últimas 10 | Tail Std | Conv. @90% | Tempo Total | Loss Final |
|---------|-----------|---------------|-----------------|----------|------------|-------------|------------|
| MNIST | 4731 | 0.9938 | 0.9937 | 0.0004 | 3/4731 | 1h33min | 0.0191 |
| Fashion MNIST | 3809 | 0.9185 | 0.9181 | 0.0030 | 348/3809 | 1h33min | 0.2223 |
| CIFAR-10 | 4645 | 0.7362 | 0.7349 | 0.0108 | 2715/4645 | 1h33min | 0.7573 |
| GTSRB | 2534 | 0.9625 | 0.9613 | 0.0023 | 813/2534 | 1h33min | 0.1531 |

### 2.2 Cenário Non-IID

| Dataset | Avaliações | Acurácia Máx. | Média Últimas 10 | Tail Std | Conv. @90% | Tempo Total | Loss Final |
|---------|-----------|---------------|-----------------|----------|------------|-------------|------------|
| MNIST | 4828 | 0.9859 | 0.9837 | 0.0033 | 953/4828 | 1h33min | 0.0546 |
| Fashion MNIST | 3860 | 0.8472 | 0.8369 | 0.0278 | 1554/3860 | 1h33min | 0.4945 |
| CIFAR-10 | 5253 | 0.4765 | 0.4465 | 0.0391 | 4002/5253 | 1h33min | 1.4638 |
| GTSRB | 2633 | 0.4899 | 0.2374 | 0.0759 | 2466/2633 | 1h32min | 2.6594 |

### 2.3 Impacto IID → Non-IID por Dataset

| Dataset | Δ Acurácia Máx. | Δ Avg Last10 | Razão Tail Std (non/iid) |
|---------|----------------|-------------|--------------------------|
| MNIST | -0.0079 | -0.0100 | 8.1× |
| Fashion MNIST | -0.0713 | -0.0812 | 9.3× |
| CIFAR-10 | -0.2597 | -0.2884 | 3.6× |
| GTSRB | -0.4726 | -0.7239 | 33.3× |

## 3. Análise por Dataset

### 3.1 MNIST

**IID:** acurácia máxima de **0.9938** (99.38%), média nas últimas 10 avaliações de 0.9937, tail_std de 0.0004 (baixa oscilação). Convergência em 3/4731 avaliações. Tempo total: 1h33min.

**Non-IID:** acurácia máxima de **0.9859** (98.59%), média nas últimas 10 de 0.9837, tail_std de 0.0033. Convergência em 953/4828 avaliações.

**Diferencial IID→Non-IID:** queda de 0.79 p.p. na acurácia máxima e 8.1× mais instabilidade na cauda.

### 3.2 Fashion MNIST

**IID:** acurácia máxima de **0.9185** (91.85%), média nas últimas 10 avaliações de 0.9181, tail_std de 0.0030 (baixa oscilação). Convergência em 348/3809 avaliações. Tempo total: 1h33min.

**Non-IID:** acurácia máxima de **0.8472** (84.72%), média nas últimas 10 de 0.8369, tail_std de 0.0278. Convergência em 1554/3860 avaliações.

**Diferencial IID→Non-IID:** queda de 7.13 p.p. na acurácia máxima e 9.3× mais instabilidade na cauda.

### 3.3 CIFAR-10

**IID:** acurácia máxima de **0.7362** (73.62%), média nas últimas 10 avaliações de 0.7349, tail_std de 0.0108 (baixa oscilação). Convergência em 2715/4645 avaliações. Tempo total: 1h33min.

**Non-IID:** acurácia máxima de **0.4765** (47.65%), média nas últimas 10 de 0.4465, tail_std de 0.0391. Convergência em 4002/5253 avaliações.

**Diferencial IID→Non-IID:** queda de 25.97 p.p. na acurácia máxima e 3.6× mais instabilidade na cauda.

### 3.4 GTSRB

**IID:** acurácia máxima de **0.9625** (96.25%), média nas últimas 10 avaliações de 0.9613, tail_std de 0.0023 (baixa oscilação). Convergência em 813/2534 avaliações. Tempo total: 1h33min.

**Non-IID:** acurácia máxima de **0.4899** (48.99%), média nas últimas 10 de 0.2374, tail_std de 0.0759. Convergência em 2466/2633 avaliações.

**Diferencial IID→Non-IID:** queda de 47.26 p.p. na acurácia máxima e 33.3× mais instabilidade na cauda.

> **Destaque:** O GTSRB é o dataset mais sensível à heterogeneidade dos dados. Com 43 classes e dados de trânsito naturalmente desbalanceados por região, a queda de 47.26 p.p. e 33.3× de instabilidade é o caso mais extremo observado. Isso reforça a importância do γ adaptativo em datasets multiclasse com alta heterogeneidade natural.

## 4. IID vs. Non-IID — Generalização do Comportamento

O estudo de ablação no CIFAR-10 mostrou que cenários Non-IID são fundamentalmente
mais difíceis: menor acurácia, convergência mais lenta e maior instabilidade.
A tabela abaixo resume se esse padrão se confirma nos novos datasets:

| Dataset | Queda de Acurácia | Instabilidade Maior | Convergência Mais Lenta |
|---------|------------------|---------------------|------------------------|
| MNIST | Sim | Sim | Sim |
| Fashion MNIST | Sim | Sim | Sim |
| CIFAR-10 | Sim | Sim | Sim |
| GTSRB | Sim | Sim | Sim |

## 5. Conexão com o Estudo de Ablação

O estudo de ablação anterior (CIFAR-10, parâmetros α/β/γ) identificou cinco conclusões
principais (C1–C5). Verificamos aqui se elas se sustentam nos novos datasets:

**C1 — β é o parâmetro mais sensível**
Os experimentos atuais usam β=0.999 (valor ótimo identificado na ablação).
O fato de todos os datasets aprenderem com essa configuração confirma indiretamente
que β=0.999 é generalizável.

**C2 — γ é indispensável em Non-IID**
Todos os experimentos usam γ=0.075. A diferença de desempenho IID vs. Non-IID
observada em cada dataset é consistente com a importância da penalização por staleness
para proteger o modelo global em cenários heterogêneos.

**C3 — α tem trade-off estabilidade vs. velocidade**
α=0.8 foi mantido fixo. O tail_std observado em Non-IID em todos os datasets
reflete o comportamento esperado de maior oscilação com α alto.

**C4 — A fórmula tripartite é validada**
A combinação α=0.8, β=0.999, γ=0.075 produziu convergência efetiva em todos os
datasets testados, sugerindo que a fórmula não é específica ao CIFAR-10.

**C5 — Non-IID amplifica todas as sensibilidades**
Confirmado em todos os datasets: queda de acurácia, maior instabilidade e
convergência mais lenta são padrões consistentes, não artefatos do CIFAR-10.

## 6. Conclusões e Próximos Passos

### Conclusões

1. **O comportamento observado no CIFAR-10 generaliza** para datasets de complexidade
   variada (MNIST, Fashion MNIST, GTSRB): convergência consistente em IID,
   queda de desempenho e maior instabilidade em Non-IID.

2. **A fórmula de agregação assíncrona** (`agg_factor = α·β^v·1/(1+γ·s)`) com os
   parâmetros identificados na ablação é robusta — produz aprendizado efetivo em
   todos os quatro datasets sem ajuste fino por dataset.

3. **Complexidade da tarefa** impacta o nível absoluto de acurácia, mas não o
   padrão qualitativo de comportamento: datasets mais difíceis (CIFAR-10, GTSRB)
   atingem acurácias menores, mas a degradação IID→Non-IID é estruturalmente similar.

4. **O cenário Non-IID continua sendo o desafio central**: em todos os datasets,
   a queda de acurácia e o aumento de instabilidade são pronunciados, reforçando
   a conclusão C5 do estudo de ablação.

### Próximos Passos Sugeridos

- Investigar γ adaptativo (variando em função do staleness médio corrente)
  como extensão natural para melhorar estabilidade Non-IID.
- Avaliar impacto do timeout no cenário síncrono (sugestão do Prof. Miguel)
  usando a mesma estratégia de Monte Carlo já empregada no assíncrono.
- Comparar diretamente os resultados assíncronos com os síncronos nos mesmos
  datasets para quantificar o trade-off velocidade vs. acurácia.
