# plot_accuracy_enhanced.py - Gráficos aprimorados para análise de acurácia (síncrono)
# Uso: python plot_accuracy_enhanced.py [--non-iid] [--window 50]
#
# Gera 3 visualizações:
#   1) Curvas suavizadas sobrepostas (média móvel)
#   2) Subplots individuais com banda de confiança (min/max na janela)
#   3) Boxplot por faixas de rodadas para comparação estatística

import json
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def moving_average(values, window):
    """Calcula a média móvel com janela expansiva no início.
    
    Para os primeiros pontos (i < window), usa todos os dados disponíveis
    até ali (janela crescente). A partir do ponto `window`, usa a janela fixa.
    Retorna um array do mesmo tamanho que `values`.
    """
    n = len(values)
    result = np.empty(n)
    cumsum = np.cumsum(values)
    for i in range(n):
        if i < window:
            result[i] = cumsum[i] / (i + 1)
        else:
            result[i] = (cumsum[i] - cumsum[i - window]) / window
    return result


def moving_min_max(values, window):
    """Calcula o mínimo e máximo em uma janela deslizante (expansiva no início).
    
    Retorna arrays do mesmo tamanho que `values`.
    """
    n = len(values)
    mins, maxs = np.empty(n), np.empty(n)
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        mins[i] = np.min(chunk)
        maxs[i] = np.max(chunk)
    return mins, maxs


def load_data(is_non_iid):
    filename = 'accuracy_data_non_iid.json' if is_non_iid else 'accuracy_data_iid.json'
    with open(f'output-cifar-10/{filename}', 'r') as f:
        return json.load(f)


def normalize_label(label):
    """Normaliza labels como '25.0' para '25'."""
    try:
        return str(int(float(label)))
    except ValueError:
        return label


def get_color(label):
    colors = {'25': '#1f77b4', '50': '#ff7f0e', '75': '#2ca02c'}
    return colors.get(normalize_label(label), None)


def plot_smoothed_overlay(data, window, is_non_iid):
    """Gráfico 1: Todas as curvas suavizadas sobrepostas."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, entries in sorted(data.items(), key=lambda x: float(x[0])):
        points = sorted(entries, key=lambda x: x['time'])
        acc = np.array([p['accuracy'] for p in points])
        smoothed = moving_average(acc, window)
        x_axis = np.arange(1, len(acc) + 1)

        color = get_color(label)
        display = normalize_label(label)
        # Linha suavizada (destaque)
        ax.plot(x_axis, smoothed, label=f'{display}% conexão (MM, w={window})',
                linewidth=2, color=color)
        # Dados originais com transparência
        ax.plot(x_axis, acc, alpha=0.15, linewidth=0.5, color=color)

    ax.set_xlabel('Número de rodadas', fontsize=12)
    ax.set_ylabel('Acurácia do modelo', fontsize=12)
    ax.set_title('Curvas suavizadas — comparação por percentual de conexão', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    suffix = 'non_iid' if is_non_iid else 'iid'
    path = f'output-cifar-10/accuracy_{suffix}.png'
    fig.savefig(path, dpi=150)
    print(f'Gráfico salvo: {path}')
    plt.close(fig)


def plot_individual_bands(data, window, is_non_iid):
    """Gráfico 2: Subplots individuais com banda de confiança (min/max)."""
    labels = sorted(data.keys(), key=lambda x: float(x))
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5), sharey=True)

    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        points = sorted(data[label], key=lambda x: x['time'])
        acc = np.array([p['accuracy'] for p in points])
        smoothed = moving_average(acc, window)
        lo, hi = moving_min_max(acc, window)
        x_axis = np.arange(1, len(acc) + 1)

        color = get_color(label)
        display = normalize_label(label)
        ax.fill_between(x_axis, lo, hi, alpha=0.2, color=color, label='Faixa min/max')
        ax.plot(x_axis, smoothed, linewidth=2, color=color, label=f'Média móvel (w={window})')

        ax.set_title(f'{display}% conexão', fontsize=12)
        ax.set_xlabel('Rodadas', fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Acurácia do modelo', fontsize=12)
    fig.suptitle('Acurácia por percentual de conexão — com banda de variação', fontsize=13, y=1.02)
    fig.tight_layout()

    suffix = 'non_iid' if is_non_iid else 'iid'
    path = f'output-cifar-10/accuracy_bands_{suffix}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Gráfico salvo: {path}')
    plt.close(fig)


def plot_boxplot_by_range(data, is_non_iid, n_bins=8):
    """Gráfico 3: Boxplot por faixas de rodadas para comparação estatística."""
    labels = sorted(data.keys(), key=lambda x: float(x))

    # Achar o maior número de rodadas entre todas as séries
    max_rounds = max(len(data[l]) for l in labels)
    bin_edges = np.linspace(0, max_rounds, n_bins + 1, dtype=int)

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5), sharey=True)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        points = sorted(data[label], key=lambda x: x['time'])
        acc = np.array([p['accuracy'] for p in points])

        box_data = []
        tick_labels = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if lo < len(acc):
                chunk = acc[lo:min(hi, len(acc))]
                if len(chunk) > 0:
                    box_data.append(chunk)
                    tick_labels.append(f'{lo}-{min(hi, len(acc))}')

        color = get_color(label)
        display = normalize_label(label)
        bp = ax.boxplot(box_data, patch_artist=True, tick_labels=tick_labels)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_title(f'{display}% conexão', fontsize=12)
        ax.set_xlabel('Faixa de rodadas', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('Acurácia do modelo', fontsize=12)
    fig.suptitle('Distribuição da acurácia por faixa de rodadas', fontsize=13, y=1.02)
    fig.tight_layout()

    suffix = 'non_iid' if is_non_iid else 'iid'
    path = f'output-cifar-10/accuracy_boxplot_{suffix}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Gráfico salvo: {path}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gráficos aprimorados de acurácia (síncrono)')
    parser.add_argument('--non-iid', action='store_true', help='Usar dados non-IID')
    parser.add_argument('--window', type=int, default=50,
                        help='Tamanho da janela para média móvel (default: 50)')
    parser.add_argument('--bins', type=int, default=8,
                        help='Número de faixas para o boxplot (default: 8)')
    args = parser.parse_args()

    data = load_data(args.non_iid)

    print('=== Gráfico 1: Curvas suavizadas sobrepostas ===')
    plot_smoothed_overlay(data, args.window, args.non_iid)

    print('\n=== Gráfico 2: Subplots com banda de confiança ===')
    plot_individual_bands(data, args.window, args.non_iid)

    print('\n=== Gráfico 3: Boxplot por faixa de rodadas ===')
    plot_boxplot_by_range(data, args.non_iid, args.bins)
