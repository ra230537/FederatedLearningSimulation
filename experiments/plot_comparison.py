# experiments/compare_sync_async.py
import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.ema import exponential_moving_average

# ---------------------------------------------------------------------------
# Resolução de caminhos (independente de CWD)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

_DATASET_OUTPUT = {
    "cifar10":       "output-cifar-10",
    "mnist":         "output-mnist",
    "fashion_mnist": "output-fashion-mnist",
    "gtsrb":         "output-gtsrb",
}

def _resolve_dirs(dataset, sync_dir_override=None, async_dir_override=None):
    """Retorna (sync_dir, async_dir) como caminhos absolutos."""
    out = _DATASET_OUTPUT[dataset]
    sync = sync_dir_override or os.path.join(_PROJECT_ROOT, "src", "synchronous", out)
    async_ = async_dir_override or os.path.join(_PROJECT_ROOT, "src", "asynchronous", out)
    return sync, async_

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    """Carrega JSON de `path`. Retorna None se o arquivo não existir."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def normalize_keys(data):
    """Converte chaves como '25.0' para '25'. Ignora chaves não-numéricas."""
    result = {}
    for k, v in data.items():
        try:
            result[str(int(float(k)))] = v
        except (ValueError, TypeError):
            pass
    return result


def compute_metrics(entries):
    """Calcula métricas sobre uma lista de {'loss', 'accuracy', 'time'}.

    Parâmetros:
        entries : lista não-vazia de dicts com chaves 'accuracy' e 'time'

    Levanta:
        ValueError se entries for vazio.

    Retorna dict com:
        max_acc    : float  — pico de acurácia
        avg_last10 : float  — média dos últimos min(10, n) pontos
        tempo_min  : float  — tempo total em minutos
    """
    if not entries:
        raise ValueError("compute_metrics requer lista não-vazia de entries")
    accs = [e["accuracy"] for e in entries]
    n = len(accs)
    return {
        "max_acc":    max(accs),
        "avg_last10": float(np.mean(accs[-min(10, n):])),
        "tempo_min":  entries[-1]["time"] / 60.0,
    }


# ---------------------------------------------------------------------------
# Plotagem de comparação
# ---------------------------------------------------------------------------

_PERCENTILE_LABELS = {
    "25": "p25 — timeout apertado",
    "50": "p50 — timeout médio",
    "75": "p75 — timeout folgado",
}

_COLOR_SYNC  = "#1f77b4"
_COLOR_ASYNC = "#ff7f0e"


def plot_comparison(sync_data, async_data, dist_label, dataset, output_dir, alpha=0.1):
    """Gera PNG com 3 subplots (p25/p50/p75) comparando sync vs async.

    Parâmetros:
        sync_data   : dict normalizado {percentil_str: [entries]}; pode ser None
        async_data  : idem para async
        dist_label  : "IID" ou "Non-IID"
        dataset     : str usado no título (ex: "cifar10")
        output_dir  : diretório onde o PNG será salvo
        alpha       : fator EMA
    """
    percentiles = ["25", "50", "75"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, p in zip(axes, percentiles):
        subplot_title = _PERCENTILE_LABELS.get(p, f"p{p}")
        ax.set_title(subplot_title, fontsize=11)
        ax.set_xlabel("Tempo (s)", fontsize=10)
        ax.grid(True, alpha=0.3)

        has_any = False

        # Curva síncrona
        if sync_data and p in sync_data:
            entries = sorted(sync_data[p], key=lambda e: e["time"])
            times = [e["time"] for e in entries]
            accs  = [e["accuracy"] for e in entries]
            smoothed = exponential_moving_average(accs, alpha)
            ax.plot(times, accs, color=_COLOR_SYNC, alpha=0.35, linewidth=0.5)
            ax.plot(times, smoothed, color=_COLOR_SYNC, linewidth=2,
                    label="Síncrono")
            has_any = True

        # Curva assíncrona
        if async_data and p in async_data:
            entries = sorted(async_data[p], key=lambda e: e["time"])
            times = [e["time"] for e in entries]
            accs  = [e["accuracy"] for e in entries]
            smoothed = exponential_moving_average(accs, alpha)
            ax.plot(times, accs, color=_COLOR_ASYNC, alpha=0.35, linewidth=0.5)
            ax.plot(times, smoothed, color=_COLOR_ASYNC, linewidth=2,
                    label="Assíncrono")
            has_any = True

        if not has_any:
            ax.text(0.5, 0.5, "Dados ausentes", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=10)

    axes[0].set_ylabel("Acurácia", fontsize=10)
    axes[0].legend(fontsize=9, loc="lower right")

    fig.suptitle(
        f"Sync vs Async — {dataset.upper()} {dist_label} — Acurácia × Tempo",
        fontsize=13,
    )
    fig.tight_layout()

    suffix = "non_iid" if dist_label == "Non-IID" else "iid"
    out_path = os.path.join(output_dir, f"comparison_{suffix}.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Impressão de tabela
# ---------------------------------------------------------------------------

def print_table(sync_data, async_data, dist_label, dataset):
    """Imprime tabela comparativa de métricas no terminal."""
    SEP = "-" * 62
    print(f"\nDataset: {dataset} | Distribuicao: {dist_label}")
    print(SEP)
    print(f"{'Percentil':<10} | {'Modo':<12} | {'max_acc':>7} | {'avg_last10':>10} | {'tempo total':>11}")
    print(SEP)
    for p in ["25", "50", "75"]:
        for label, data in [("Sincrono", sync_data), ("Assincrono", async_data)]:
            if data is None or p not in data:
                print(f"p{p:<9} | {label:<12} | {'N/A':>7} | {'N/A':>10} | {'N/A':>11}")
                continue
            m = compute_metrics(data[p])
            print(
                f"p{p:<9} | {label:<12} | "
                f"{m['max_acc']*100:>6.1f}% | "
                f"{m['avg_last10']*100:>9.1f}% | "
                f"{m['tempo_min']:>9.1f} min"
            )
    print(SEP)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comparação Sync vs Async — acurácia × tempo"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=list(_DATASET_OUTPUT.keys()),
        help="Dataset a comparar (default: cifar10)",
    )
    parser.add_argument("--iid",     action="store_true", help="Apenas IID")
    parser.add_argument("--non-iid", action="store_true", help="Apenas Non-IID")
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Fator de suavização EMA (default: 0.1)",
    )
    parser.add_argument("--sync-dir",  type=str, default=None,
                        help="Override do diretório sync")
    parser.add_argument("--async-dir", type=str, default=None,
                        help="Override do diretório async")
    args = parser.parse_args()

    sync_dir, async_dir = _resolve_dirs(
        args.dataset, args.sync_dir, args.async_dir
    )

    # Decide quais distribuições processar
    if args.iid:
        distributions = [("iid", "IID")]
    elif args.non_iid:
        distributions = [("non_iid", "Non-IID")]
    else:
        distributions = [("iid", "IID"), ("non_iid", "Non-IID")]

    for suffix, label in distributions:
        sync_path  = os.path.join(sync_dir,  f"accuracy_data_{suffix}.json")
        async_path = os.path.join(async_dir, f"accuracy_data_{suffix}.json")

        sync_raw  = load_json(sync_path)
        async_raw = load_json(async_path)

        if sync_raw is None and async_raw is None:
            print(f"[AVISO] Nenhum dado encontrado para {label} "
                  f"(sync: {sync_path}, async: {async_path}) — pulando.")
            continue

        if sync_raw is None:
            print(f"[AVISO] Dados sync ausentes para {label}: {sync_path}")
        if async_raw is None:
            print(f"[AVISO] Dados async ausentes para {label}: {async_path}")

        sync_data  = normalize_keys(sync_raw)  if sync_raw  else None
        async_data = normalize_keys(async_raw) if async_raw else None

        print_table(sync_data, async_data, label, args.dataset)
        plot_comparison(
            sync_data, async_data, label,
            args.dataset, _SCRIPT_DIR, args.alpha,
        )


if __name__ == "__main__":
    main()
