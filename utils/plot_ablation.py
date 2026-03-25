import glob
import json
import os

import matplotlib

os.environ["QT_QPA_PLATFORM"] = (
    "offscreen"  # Garante matplotlib sem UI interativo no server/roteiro
)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def exponential_moving_average(values, alpha=0.1):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return values
    result = np.empty(n)
    result[0] = values[0]
    for i in range(1, n):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def plot_ablation_comparison(
    output_dir="output-cifar-10", prefix="sync", percentile="50", alpha=0.1
):
    """
    Lê os JSONs gerados no ablation_study (que contém o prefixo solicitado)
    e plota um gráfico comparativo agrupado pelo percentil especificado.
    """
    # Ex: procura por output-cifar-10/accuracy_data_*_sync_*.json
    pattern = os.path.join(output_dir, f"accuracy_data_*_{prefix}*.json")
    files = glob.glob(pattern)

    if not files:
        print(f"[{prefix.upper()}] Nenhum arquivo encontrado com o padrão: {pattern}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(files))))

    for idx, filepath in enumerate(sorted(files)):
        filename = os.path.basename(filepath)

        # Nome mais limpo pro label (tira o pedaço da frente)
        # Ex: "accuracy_data_iid_sync_E1_B32_C40.json" -> "iid_sync_E1_B32_C40"
        label_name = filename.replace("accuracy_data_", "").replace(".json", "")

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Erro lendo {filepath}: {e}")
            continue

        if percentile not in data:
            continue

        points = sorted(data[percentile], key=lambda x: x["time"])
        if not points:
            continue

        acc = np.array([p["accuracy"] for p in points])
        smoothed = exponential_moving_average(acc, alpha)
        x_axis = np.arange(1, len(acc) + 1)

        # Plota linha transparente original e a EMA destacada
        ax.plot(x_axis, acc, alpha=0.2, linewidth=1, color=colors[idx % len(colors)])
        ax.plot(
            x_axis,
            smoothed,
            label=label_name,
            linewidth=2,
            color=colors[idx % len(colors)],
        )

    ax.set_xlabel("Rodadas / Atualizações", fontsize=12)
    ax.set_ylabel("Acurácia do Modelo (%)", fontsize=12)
    ax.set_title(
        f"Ablação ({prefix.upper()}) - Percentil de Latência Clientes: {percentile}%",
        fontsize=13,
    )

    # Colocar a legenda fora do gráfico para não tampar as linhas
    ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(
        output_dir, f"ablation_comparison_{prefix}_p{percentile}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Gráfico de comparação ({prefix.upper()}) gerado e salvo em: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gera gráficos de ablação comparativos"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output-cifar-10", help="Pasta com JSONs"
    )
    parser.add_argument(
        "--prefix", type=str, default="sync", help="'sync' ou 'async' etc."
    )
    parser.add_argument(
        "--percentile", type=str, default="50", help="Percentil de conexão, ex: '50'"
    )
    args = parser.parse_args()

    plot_ablation_comparison(args.output_dir, args.prefix, args.percentile)
