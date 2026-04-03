# Simulador de Aprendizado Federado

Simulador de aprendizado federado com abordagens sincronas e assincronas, com suporte a multiplos datasets e modelos CNN.

O projeto foi desenvolvido no contexto de iniciacao cientifica FAPESP na UNICAMP.

## O que este repositorio faz

- Simulacao de FL sincronizado (FedAvg por rodada)
- Simulacao de FL assincrono (agregacao on-the-fly com staleness)
- Particionamento IID e Non-IID
- Timeout por percentis via Monte Carlo
- Geracao automatica de graficos de acuracia
- Estudo de ablacao para parametros do servidor assincrono

## Datasets suportados

Os scripts principais aceitam o argumento --dataset com as opcoes:

- cifar10
- mnist
- fashion_mnist
- gtsrb

Diretorio de saida por dataset:

- cifar10 -> output-cifar-10
- mnist -> output-mnist
- fashion_mnist -> output-fashion-mnist
- gtsrb -> output-gtsrb

## Estrutura do projeto

```
.
├── src/
│   ├── synchronous/
│   │   ├── main.py
│   │   ├── server.py
│   │   ├── client.py
│   │   ├── constants.py
│   │   └── monte_carlo.py
│   ├── asynchronous/
│   │   ├── main.py
│   │   ├── server.py
│   │   ├── client.py
│   │   ├── constants.py
│   │   └── monte_carlo.py
│   └── utils/
│       ├── data_loader.py
│       ├── data_split.py
│       ├── models.py
│       └── plot_accuracy.py
├── experiments/
│   ├── ablation_study.py
│   └── plot_ablation.py
└── output-*/
```

## Requisitos

Python 3.10+ recomendado.

Pacotes principais:

- tensorflow
- numpy
- matplotlib
- scipy
- pillow

Arquivo de dependencias:

- requirements.txt

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Instalacao rapida com venv:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Guia de comandos curtos:

- receitas_rapidas.md

## Como executar

Todos os comandos abaixo assumem execucao na raiz do repositorio.

### 1) Simulacao sincrona

Comportamento:

- Se nenhum flag de distribuicao for passado, roda IID e Non-IID
- Por padrao roda percentis 25, 50 e 75
- Opcionalmente roda um cenario extra sem timeout com --no-timeout

Comandos:

```bash
python src/synchronous/main.py
python src/synchronous/main.py --dataset mnist
python src/synchronous/main.py --dataset fashion_mnist --non-iid
python src/synchronous/main.py --dataset gtsrb --iid --percentile 50
python src/synchronous/main.py --dataset cifar10 --no-timeout
```

Argumentos disponiveis em src/synchronous/main.py:

- --dataset {cifar10,mnist,fashion_mnist,gtsrb}
- --iid
- --non-iid
- --percentile INT
- --no-timeout

### 2) Simulacao assincrona

Comportamento:

- Se nenhum flag de distribuicao for passado, roda IID e Non-IID
- Por padrao roda percentis 25, 50 e 75
- Permite ajustar hiperparametros da agregacao assincrona

Comandos:

```bash
python src/asynchronous/main.py
python src/asynchronous/main.py --dataset mnist --percentile 50
python src/asynchronous/main.py --dataset gtsrb --non-iid --num-updates 40
python src/asynchronous/main.py --base-alpha 0.5 --decay-of-base-alpha 0.99 --tardiness-sensivity 0.1
python src/asynchronous/main.py --output-prefix experimento_teste
```

Argumentos disponiveis em src/asynchronous/main.py:

- --num-clients INT
- --num-updates INT
- --epochs INT
- --batch-size INT
- --dataset {cifar10,mnist,fashion_mnist,gtsrb}
- --iid
- --non-iid
- --percentile INT
- --base-alpha FLOAT
- --decay-of-base-alpha FLOAT
- --tardiness-sensivity FLOAT
- --output-prefix STR

### 3) Estudo de ablacao

Script: experiments/ablation_study.py

O estudo varia um parametro por vez no cenario assincrono e executa IID e Non-IID.

Valores considerados:

- base_alpha: 0.3, 0.5, 0.8
- decay_of_base_alpha: 0.999, 0.99, 0.95
- tardiness_sensivity: 0.0, 0.075, 0.5

Comandos:

```bash
python experiments/ablation_study.py
python experiments/ablation_study.py --num-updates 80 --percentile 75
python experiments/ablation_study.py --num-clients 20 --epochs 1 --batch-size 32
```

Importante:

- Atualmente o experiments/ablation_study.py chama src/asynchronous/main.py sem --dataset,
  portanto usa cifar10 por padrao.
- Saidas de ablacao sao gravadas em output-cifar-10.

### 4) Gerar graficos

Graficos de acuracia (gerais):

```bash
python -m utils.plot_accuracy --output-dir output-cifar-10
python -m utils.plot_accuracy --output-dir output-mnist --non-iid --x-label atualizacoes
```

Graficos do estudo de ablacao:

```bash
python experiments/plot_ablation.py --distribution iid --vary all
python experiments/plot_ablation.py --distribution all --vary all --percentile 50
python experiments/plot_ablation.py --mode grid --distribution iid --prefix async --percentile 50
```

## Benchmark estimado (Monte Carlo)

Para estimar tempo sem rodar treino completo, use os mesmos modulos Monte Carlo do projeto.

### Sincrono

No sincrono, o modulo retorna timeout por rodada.
Uma estimativa simples de tempo total e:

- tempo_total_estimado ~= timeout_por_rodada * NUM_UPDATES

Comando:

```bash
python -c "from src.synchronous.monte_carlo import get_percentiles_timeout;from src.synchronous.constants import MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,MIN_TRAIN_TIME,MAX_TRAIN_TIME,NUM_UPDATES;ps=[25,50,75];ts=get_percentiles_timeout(ps,MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,MIN_TRAIN_TIME,MAX_TRAIN_TIME);print({f'p{p}':{'timeout_por_rodada_s':float(t),'tempo_total_estimado_s':float(t*NUM_UPDATES)} for p,t in zip(ps,ts)})"
```

### Assincrono

No assincrono, o modulo ja retorna timeout total para NUM_UPDATES.

Comando:

```bash
python -c "from src.asynchronous.monte_carlo import get_percentiles_timeout;from src.asynchronous.constants import MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,MIN_TRAIN_TIME,MAX_TRAIN_TIME,NUM_UPDATES;ps=[25,50,75];ts=get_percentiles_timeout(ps,NUM_UPDATES,MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,MIN_TRAIN_TIME,MAX_TRAIN_TIME);print({f'p{p}':{'tempo_total_estimado_s':float(t)} for p,t in zip(ps,ts)})"
```

Essas estimativas sao boas para planejamento de execucao e comparacao entre cenarios.

## Formato das saidas

Os resultados de treino sao salvos em JSON, por exemplo:

- accuracy_data_iid.json
- accuracy_data_non_iid.json
- accuracy_data_iid_experimento_teste.json

Cada entrada contem:

- loss
- accuracy
- time

As imagens PNG de graficos tambem sao salvas no mesmo diretorio de saida.

## Modelos usados

Factory centralizada em utils/models.py:

- cnn_cifar10
- cnn_mnist
- cnn_fashion_mnist
- cnn_gtsrb

O mapeamento dataset -> modelo e feito em utils/data_loader.py.

## Boas praticas para manter o repositorio usavel

- Evite versionar resultados massivos novos em output-*.
- Prefira usar --percentile 50 em testes rapidos.
- Reduza --num-updates durante debug inicial.
- Use --output-prefix para separar experimentos sem sobrescrever arquivos.
- Mantenha dados e artefatos grandes fora do controle de versao quando possivel.

## Troubleshooting rapido

- Erro de memoria/tempo: reduza --num-updates e/ou --num-clients.
- Erro com GTSRB: confira permissao de escrita em data/gtsrb para download/extracao.
- Sem graficos: confirme se os JSONs estao no diretorio passado em --output-dir.
