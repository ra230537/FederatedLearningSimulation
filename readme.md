# Simulador de Aprendizado Federado

Simulador de aprendizado federado com abordagens **sincrona** e **assincrona** sobre o dataset CIFAR-10. Desenvolvido como parte de Iniciacao Cientifica FAPESP na UNICAMP, sob orientacao dos professores Luiz Fernando Bittencourt (UNICAMP) e Miguel Elias M. Campista (UFRJ).

## Estrutura do Projeto

```
├── synchronous/          # FL sincrono
│   ├── main.py           # Ponto de entrada
│   ├── server.py         # Agregacao FedAvg por rodada
│   ├── client.py         # Treinamento local
│   ├── constants.py      # Parametros de configuracao
│   └── monte_carlo.py    # Calculo de timeout via Monte Carlo
├── asynchronous/         # FL assincrono
│   ├── main.py           # Ponto de entrada (com args de linha de comando)
│   ├── server.py         # Agregacao on-the-fly com staleness-aware
│   ├── client.py         # Loop continuo de treinamento/envio
│   ├── constants.py      # Parametros de configuracao
│   └── monte_carlo.py    # Calculo de timeout via Monte Carlo
├── utils/
│   ├── models.py         # Arquitetura CNN para CIFAR-10
│   ├── data_split.py     # Distribuicao IID e Non-IID
│   └── plot_accuracy.py  # Graficos com EMA smoothing e boxplots
├── ablation_study.py     # Varredura de parametros (async)
├── plot_ablation.py      # Graficos comparativos do ablation study
└── output-cifar-10/      # Resultados (JSONs e PNGs)
```

## Modelo

CNN sequencial para CIFAR-10 (32x32x3, 10 classes):

```
Conv2D(32) -> Conv2D(32) -> MaxPool -> Dropout(0.25)
Conv2D(64) -> Conv2D(64) -> MaxPool -> Dropout(0.25)
Flatten -> Dense(512) -> Dropout(0.5) -> Dense(10, softmax)
```

Otimizador: Adam. Loss: Sparse Categorical Crossentropy.

## Distribuicao de Dados

- **IID:** Distribuicao Dirichlet entre os clientes.
- **Non-IID:** Cada cliente recebe no maximo 3 classes. A proporcao de dados por classe e definida via Dirichlet. Todas as 10 classes sao garantidas no sistema.

## Abordagem Sincrona

O servidor coordena rodadas: distribui pesos -> clientes treinam em paralelo (threads) -> servidor espera timeout -> agrega via FedAvg ponderado pelo tamanho do dataset de cada cliente.

O timeout por rodada e calculado via simulacao de Monte Carlo (1M amostras) para os percentis 25%, 50% e 75%.

### Parametros (`synchronous/constants.py`)

| Parametro | Valor | Descricao |
|-----------|:-----:|-----------|
| `NUM_CLIENTS` | 40 | Clientes participantes |
| `NUM_UPDATES` | 80 | Rodadas de treinamento |
| `TIMEOUT` | 8s | Timeout fixo por rodada |
| `LOCAL_EPOCHS` | 1 | Epocas locais por rodada |
| `BATCH_SIZE` | 32 | Tamanho do batch |
| `MIN/MAX_CONNECTION_TIME` | 0/5s | Latencia de conexao simulada |
| `MIN/MAX_TRAIN_TIME` | 0/10s | Latencia de treinamento simulada |

### Execucao

```bash
cd synchronous
python main.py                  # IID, todos os percentis (p25, p50, p75)
python main.py --non-iid        # Non-IID, todos os percentis
python main.py --percentile 50  # Apenas p50
```

## Abordagem Assincrona

Clientes treinam continuamente e enviam atualizacoes sem esperar rodadas. O servidor agrega on-the-fly com penalizacao por staleness:

```
agg_factor = base_alpha * decay^version * 1/(1 + tardiness_sensitivity * staleness)
```

O timeout global e calculado via Monte Carlo, multiplicado pelo numero de atualizacoes.

### Parametros (`asynchronous/constants.py`)

| Parametro | Valor | Descricao |
|-----------|:-----:|-----------|
| `NUM_CLIENTS` | 40 | Clientes participantes |
| `NUM_UPDATES` | 80 | Atualizacoes do modelo |
| `LOCAL_EPOCHS` | 1 | Epocas locais |
| `BATCH_SIZE` | 32 | Tamanho do batch |
| `BASE_ALPHA` | 0.8 | Taxa de aprendizado inicial da agregacao |
| `DECAY_OF_BASE_ALPHA` | 0.999 | Decaimento exponencial |
| `TARDINESS_SENSITIVITY` | 0.075 | Penalizacao por staleness |
| `MIN/MAX_CONNECTION_TIME` | 0/5s | Latencia de conexao simulada |
| `MIN/MAX_TRAIN_TIME` | 0/90s | Latencia de treinamento simulada |

### Execucao

```bash
cd asynchronous
python main.py                                             # IID, todos os percentis (p25, p50, p75)
python main.py --non-iid                                   # Non-IID, todos os percentis
python main.py --base-alpha 0.5 --tardiness-sensivity 0.1  # Parametros customizados
python main.py --percentile 50                             # Apenas p50
```

## Ablation Study

O `ablation_study.py` executa um estudo de ablacao com **variacao isolada** (one-at-a-time) no cenario assincrono. Cada parametro e variado individualmente enquanto os outros ficam no valor padrao (alpha=0.8, decay=0.999, tardiness=0.075):

- `base_alpha`: [0.3, 0.5, 0.8]
- `decay_of_base_alpha`: [0.999, 0.99, 0.95]
- `tardiness_sensivity`: [0.0, 0.075, 0.5]
- Distribuicao: [IID, Non-IID]

Total: 14 experimentos unicos. Usa um unico percentil (default p50) para reduzir tempo de execucao.

```bash
python ablation_study.py                       # default: 40 updates, p50
python ablation_study.py --num-updates 80      # mais updates (mais lento, melhor convergencia)
python ablation_study.py --percentile 75       # usar p75 em vez de p50
```

### Graficos do Ablation

O `plot_ablation.py` gera graficos comparativos dos resultados:

```bash
python plot_ablation.py                                    # todos os parametros, IID
python plot_ablation.py --distribution all --vary all      # todos os parametros, IID + Non-IID
python plot_ablation.py --vary base_alpha --distribution iid  # so base_alpha, IID
python plot_ablation.py --mode grid --distribution iid     # modo grid (facetas por alpha)
```

## Visualizacao

O modulo `utils/plot_accuracy.py` gera 3 tipos de graficos:

1. **Overlay suavizado:** curvas EMA (alpha=0.1) sobrepostas por percentil.
2. **Subplots individuais:** um subplot por percentil com bandas de confianca EMA.
3. **Boxplots por faixa:** distribuicao estatistica da acuracia em faixas temporais.

O eixo X usa "rodadas" no sincrono e "atualizacoes" no assincrono.

## Saida

Resultados salvos em `output-cifar-10/`:
- JSONs com tuplas (loss, accuracy, time) por percentil
- PNGs dos graficos gerados
