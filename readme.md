# Simulação de Aprendizado Federado Síncrono

Simulação de aprendizado federado síncrono sobre o dataset MNIST. O servidor coordena múltiplos clientes que treinam localmente e enviam seus pesos, que são então agregados usando FedAvg ponderado pelo tamanho do dataset de cada cliente.

## Arquitetura

- **Modelo global:** rede densa com camada oculta de 128 neurônios (ReLU) e saída softmax de 10 classes, com entrada de 784 atributos (imagens 28x28 achatadas).
- **Agregação:** média federada (FedAvg) ponderada pelo tamanho do dataset de cada cliente.
- **Divisão dos dados:** distribuição Dirichlet entre os clientes, garantindo heterogeneidade nos dados locais.
- **Simulação de latência:** cada cliente possui um atraso aleatório de conexão (0,1–5 s) somado a um atraso de treinamento (0.1 s–90 s).

## Parâmetros principais (`main.py`)

| Parâmetro      | Valor padrão | Descrição                              |
|----------------|:------------:|----------------------------------------|
| `num_clients`  | 40           | Número de clientes participantes       |
| `round_num`    | 80           | Número de rodadas de treinamento       |
| `timeout`      | 8 s          | Tempo máximo de espera por rodada      |
| `epochs`       | 1            | Épocas de treinamento local por rodada |
| `batch_size`   | 32           | Tamanho do batch local                 |

## Critérios de encerramento de rodada

O servidor suporta dois modos de espera pelos clientes em cada rodada, controlados pelos parâmetros `is_percentage_boundary` e `percentage_boundary` no construtor de `Server`:

1. **Timeout fixo** (`is_percentage_boundary=False`): aguarda até `timeout` segundos. É o comportamento padrão quando os dois parâmetros opcionais são omitidos.

2. **Boundary percentual** (`is_percentage_boundary=True`): aguarda até que uma fração `percentage_boundary` dos clientes conclua o treinamento. Clientes que não terminarem dentro desse critério são descartados na rodada.

## Simulação de múltiplos boundaries

O `main.py` itera sobre diferentes valores de boundary usando:

```python
for boundary in range(25, 100, 25):  # 25%, 50%, 75%
```

Para testar um único valor, ajuste o range para `range(x, x + z, z)` e defina `x` como o percentual desejado (em inteiro de 0 a 100). Caso `is_percentage_boundary` seja `False`, o valor de boundary é irrelevante.

## Saída

Ao final, é gerado o gráfico `output/accuracy.png` com a acurácia do modelo global ao longo do tempo de treinamento para cada boundary simulado.
