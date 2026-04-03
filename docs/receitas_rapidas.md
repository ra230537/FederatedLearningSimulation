# Receitas Rapidas

Comandos curtos para executar cenarios comuns sem precisar lembrar todos os argumentos.

## Preparacao

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Rodar simulacao sincrona

```bash
# CIFAR-10, IID + Non-IID, percentis 25/50/75
python synchronous/main.py

# MNIST, apenas IID
python synchronous/main.py --dataset mnist --iid

# GTSRB, apenas Non-IID, p50
python synchronous/main.py --dataset gtsrb --non-iid --percentile 50

# CIFAR-10 com cenario sem timeout
python synchronous/main.py --dataset cifar10 --no-timeout
```

## Rodar simulacao assincrona

```bash
# Padrao
python asynchronous/main.py

# Fashion-MNIST, Non-IID, p50
python asynchronous/main.py --dataset fashion_mnist --non-iid --percentile 50

# Ajuste de hiperparametros assincronos
python asynchronous/main.py --base-alpha 0.5 --decay-of-base-alpha 0.99 --tardiness-sensivity 0.1

# Menos atualizacoes para teste rapido
python asynchronous/main.py --num-updates 20 --percentile 50
```

## Estudo de ablacao

```bash
# Default (40 updates, p50)
python ablation_study.py

# Mais atualizacoes
python ablation_study.py --num-updates 80 --percentile 50
```

## Gerar graficos

```bash
# Graficos gerais (precisa dos JSONs no diretorio)
python -m utils.plot_accuracy --output-dir output-cifar-10

# Graficos gerais Non-IID para MNIST
python -m utils.plot_accuracy --output-dir output-mnist --non-iid --x-label atualizacoes

# Graficos de ablacao (IID e Non-IID)
python plot_ablation.py --distribution all --vary all --percentile 50
```

## Benchmark estimado via Monte Carlo (sem rodar treino)

Estes comandos calculam tempo estimado com base nas distribuicoes usadas no codigo.

```bash
python -c "from synchronous.monte_carlo import get_percentiles_timeout;from synchronous.constants import MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,MIN_TRAIN_TIME,MAX_TRAIN_TIME,NUM_UPDATES;ps=[25,50,75];ts=get_percentiles_timeout(ps,MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,MIN_TRAIN_TIME,MAX_TRAIN_TIME);print({f'p{p}':{'timeout_por_rodada_s':float(t),'tempo_total_estimado_s':float(t*NUM_UPDATES)} for p,t in zip(ps,ts)})"
```

```bash
python -c "from asynchronous.monte_carlo import get_percentiles_timeout;from asynchronous.constants import MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,MIN_TRAIN_TIME,MAX_TRAIN_TIME,NUM_UPDATES;ps=[25,50,75];ts=get_percentiles_timeout(ps,NUM_UPDATES,MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,MIN_TRAIN_TIME,MAX_TRAIN_TIME);print({f'p{p}':{'tempo_total_estimado_s':float(t)} for p,t in zip(ps,ts)})"
```

Observacao:

- Sincrono: o timeout do modulo e por rodada.
- Assincrono: o timeout do modulo ja retorna total para NUM_UPDATES.
