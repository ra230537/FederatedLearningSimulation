# plot_accuracy.py - Módulo unificado para gráficos de acurácia
#
# Pode ser usado tanto pelo modo síncrono quanto assíncrono.
# Uso standalone: python -m utils.plot_accuracy --output-dir <dir> [--non-iid] [--alpha 0.1]
#
# Gera 3 visualizações:
#   1) Curvas suavizadas sobrepostas (EMA - Média Móvel Exponencial)
#   2) Subplots individuais com banda de confiança (EMA do desvio)
#   3) Boxplot por faixas para comparação estatística

import argparse
import json
import os

import numpy as np


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

# ---------------------------------------------------------------------------
# Funções de suavização
# ---------------------------------------------------------------------------


def exponential_moving_average(values, alpha=0.1):
    """Suavização por Média Móvel Exponencial (EMA).

    Aplica a fórmula recorrente:
        s_0 = x_0
        s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    Diferente da média móvel com janela fixa, a EMA:
      - Não possui descontinuidade em nenhum ponto;
      - Pondera exponencialmente todos os dados anteriores;
      - Reage suavemente a variações locais.

    Parâmetros:
        values : array-like de valores a suavizar.
        alpha  : fator de suavização (0 < alpha <= 1).
                 Menor → curva mais suave.
                 Maior → mais fiel aos dados originais.

    Retorna np.ndarray do mesmo tamanho que `values`.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    result = np.empty(n)
    result[0] = values[0]
    for i in range(1, n):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def ema_confidence_bands(values, smoothed, alpha=0.1):
    """Calcula bandas de confiança suavizadas via EMA do desvio absoluto.

    Em vez de min/max em janela fixa (que gera bandas irregulares e
    com descontinuidade), suaviza o desvio absoluto entre os dados
    originais e a curva EMA, produzindo bandas simétricas e contínuas.

    Retorna (lower, upper) como np.ndarrays do mesmo tamanho que `values`.
    """
    deviation = np.abs(values - smoothed)
    smooth_dev = exponential_moving_average(deviation, alpha)
    return smoothed - smooth_dev, smoothed + smooth_dev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLORS = {"25": "#1f77b4", "50": "#ff7f0e", "75": "#2ca02c", "include_no_timeout": "#d62728"}


def normalize_label(label):
    """Normaliza labels como '25.0' para '25'."""
    try:
        return str(int(float(label)))
    except ValueError:
        return label


def get_color(label):
    return COLORS.get(normalize_label(label), "#9467bd")


def sort_key(label):
    """Ordena labels numéricos primeiro, 'include_no_timeout' por último."""
    try:
        return float(label)
    except ValueError:
        return float("inf")


def get_display_label(label):
    """Retorna o texto de exibição para a legenda do gráfico."""
    normalized = normalize_label(label)
    if normalized == "include_no_timeout":
        return "Sem timeout (100%)"
    return f"{normalized}% conexão"


def load_data(output_dir, is_non_iid):
    """Carrega dados de acurácia do JSON.

    Parâmetros:
        output_dir : diretório onde estão os arquivos JSON.
        is_non_iid : se True, carrega dados non-IID.
    """
    filename = "accuracy_data_non_iid.json" if is_non_iid else "accuracy_data_iid.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------


def plot_smoothed_overlay(
    data, output_dir, is_non_iid, alpha=0.1, x_label="Número de atualizações"
):
    """Gráfico 1: Todas as curvas suavizadas sobrepostas."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, entries in sorted(data.items(), key=lambda x: sort_key(x[0])):
        points = sorted(entries, key=lambda x: x["time"])
        acc = np.array([p["accuracy"] for p in points])
        smoothed = exponential_moving_average(acc, alpha)
        x_axis = np.arange(1, len(acc) + 1)

        color = get_color(label)
        display = get_display_label(label)
        ax.plot(
            x_axis,
            smoothed,
            label=f"{display} (EMA, α={alpha})",
            linewidth=2,
            color=color,
        )
        ax.plot(x_axis, acc, alpha=0.35, linewidth=0.5, color=color)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Acurácia do modelo", fontsize=12)
    ax.set_title(
        "Curvas suavizadas — comparação por percentual de conexão", fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    suffix = "non_iid" if is_non_iid else "iid"
    path = os.path.join(output_dir, f"accuracy_{suffix}.png")
    fig.savefig(path, dpi=150)
    print(f"Gráfico salvo: {path}")
    plt.close(fig)


def plot_individual_bands(
    data, output_dir, is_non_iid, alpha=0.1, x_label="Atualizações"
):
    """Gráfico 2: Subplots individuais com banda de confiança (EMA)."""
    plt = _get_plt()
    labels = sorted(data.keys(), key=sort_key)
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5), sharey=True)

    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        points = sorted(data[label], key=lambda x: x["time"])
        acc = np.array([p["accuracy"] for p in points])
        smoothed = exponential_moving_average(acc, alpha)
        lo, hi = ema_confidence_bands(acc, smoothed, alpha)
        x_axis = np.arange(1, len(acc) + 1)

        color = get_color(label)
        display = get_display_label(label)
        ax.fill_between(
            x_axis, lo, hi, alpha=0.2, color=color, label="Banda de confiança"
        )
        ax.plot(x_axis, smoothed, linewidth=2, color=color, label=f"EMA (α={alpha})")

        ax.set_title(display, fontsize=12)
        ax.set_xlabel(x_label, fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Acurácia do modelo", fontsize=12)
    fig.suptitle(
        "Acurácia por percentual de conexão — com banda de variação",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    suffix = "non_iid" if is_non_iid else "iid"
    path = os.path.join(output_dir, f"accuracy_bands_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Gráfico salvo: {path}")
    plt.close(fig)


def plot_boxplot_by_range(
    data, output_dir, is_non_iid, n_bins=8, x_label="Faixa de atualizações"
):
    """Gráfico 3: Boxplot por faixas para comparação estatística."""
    plt = _get_plt()
    labels = sorted(data.keys(), key=sort_key)
    max_items = max(len(data[label]) for label in labels)
    bin_edges = np.linspace(0, max_items, n_bins + 1, dtype=int)

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5), sharey=True)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        points = sorted(data[label], key=lambda x: x["time"])
        acc = np.array([p["accuracy"] for p in points])

        box_data = []
        tick_labels = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if lo < len(acc):
                chunk = acc[lo : min(hi, len(acc))]
                if len(chunk) > 0:
                    box_data.append(chunk)
                    tick_labels.append(f"{lo}-{min(hi, len(acc))}")

        color = get_color(label)
        display = get_display_label(label)
        bp = ax.boxplot(box_data, patch_artist=True, tick_labels=tick_labels)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_title(display, fontsize=12)
        ax.set_xlabel(x_label, fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Acurácia do modelo", fontsize=12)
    fig.suptitle("Distribuição da acurácia por faixa", fontsize=13, y=1.02)
    fig.tight_layout()

    suffix = "non_iid" if is_non_iid else "iid"
    path = os.path.join(output_dir, f"accuracy_boxplot_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Gráfico salvo: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Função de conveniência
# ---------------------------------------------------------------------------


def generate_all_plots(
    output_dir, is_non_iid, alpha=0.1, n_bins=8, x_label="atualizações"
):
    """Carrega os dados e gera todos os 3 gráficos.

    Parâmetros:
        output_dir : diretório com os JSONs e onde salvar os PNGs.
        is_non_iid : se True, usa dados non-IID.
        alpha      : fator de suavização EMA (default: 0.1).
        n_bins     : número de faixas para o boxplot (default: 8).
        x_label    : termo usado no eixo X (ex.: 'rodadas', 'atualizações').
    """
    data = load_data(output_dir, is_non_iid)

    print("=== Gráfico 1: Curvas suavizadas sobrepostas ===")
    plot_smoothed_overlay(
        data, output_dir, is_non_iid, alpha=alpha, x_label=f"Número de {x_label}"
    )

    print("\n=== Gráfico 2: Subplots com banda de confiança ===")
    plot_individual_bands(
        data, output_dir, is_non_iid, alpha=alpha, x_label=x_label.capitalize()
    )

    print("\n=== Gráfico 3: Boxplot por faixa ===")
    plot_boxplot_by_range(
        data, output_dir, is_non_iid, n_bins=n_bins, x_label=f"Faixa de {x_label}"
    )


# ---------------------------------------------------------------------------
# Execução standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gráficos aprimorados de acurácia (módulo unificado)"
    )
    parser.add_argument("--non-iid", action="store_true", help="Usar dados non-IID")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Fator de suavização EMA (default: 0.1). Menor = mais suave.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=8,
        help="Número de faixas para o boxplot (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Diretório com os dados JSON e onde salvar gráficos",
    )
    parser.add_argument(
        "--x-label",
        type=str,
        default="atualizações",
        help="Termo do eixo X (default: atualizações)",
    )
    args = parser.parse_args()

    generate_all_plots(
        args.output_dir,
        args.non_iid,
        alpha=args.alpha,
        n_bins=args.bins,
        x_label=args.x_label,
    )
