# plot_accuracy.py - Gera gráficos de acurácia a partir dos dados JSON salvos
# Uso: python plot_accuracy.py [--non-iid]

import json
import matplotlib.pyplot as plt
import argparse


def plot_from_json(is_non_iid):
    accuracy_data_name = 'accuracy_data_non_iid.json' if is_non_iid else 'accuracy_data_iid.json'
    with open(f'output/{accuracy_data_name}', 'r') as f:
        data = json.load(f)

    all_accuracies = []
    for label, entries in data.items():
        points = sorted(entries, key=lambda x: x['time'])
        accuracy_axis = [p['accuracy'] for p in points]
        updates_axis = list(range(1, len(points) + 1))
        all_accuracies.extend(accuracy_axis)
        plt.plot(updates_axis, accuracy_axis, label=f'{label}%')

    plt.xlabel('Número de atualizações (updates)')
    plt.ylabel('Acurácia do modelo')
    # plt.xlim(0, 1000)
    if is_non_iid:
        min_acc = max(0, min(all_accuracies) - 0.05)
        max_acc = min(1, max(all_accuracies) + 0.05)
        plt.ylim(min_acc, max_acc)
    else:
        plt.ylim(0.9, 1)
    plt.legend()

    output_name = 'accuracy_non_iid.png' if is_non_iid else 'accuracy_iid.png'
    plt.savefig(f'output/{output_name}')
    print(f'Gráfico salvo em output/{output_name}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera gráfico de acurácia a partir dos dados JSON')
    parser.add_argument('--non-iid', action='store_true', help='Usar dados non-IID')
    args = parser.parse_args()
    plot_from_json(args.non_iid)
