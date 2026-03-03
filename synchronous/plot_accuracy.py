# plot_accuracy.py - Gera gráficos de acurácia a partir dos dados JSON salvos
# Uso: python plot_accuracy.py [--non-iid]

import json
import matplotlib.pyplot as plt
import argparse


def plot_from_json(is_non_iid):
    accuracy_data_name = 'accuracy_data_non_iid.json' if is_non_iid else 'accuracy_data_iid.json'
    with open(f'output-cifar-10/{accuracy_data_name}', 'r') as f:
        data = json.load(f)

    all_accuracies = []
    for label, entries in data.items():
        points = sorted(entries, key=lambda x: x['time'])
        accuracy_axis = [p['accuracy'] for p in points]
        rounds_axis = list(range(1, len(points) + 1))
        all_accuracies.extend(accuracy_axis)
        plt.plot(rounds_axis, accuracy_axis, label=f'{label}%')

    plt.xlabel('Número de rodadas')
    plt.ylabel('Acurácia do modelo')
    plt.legend()

    output_name = 'accuracy_non_iid.png' if is_non_iid else 'accuracy_iid.png'
    plt.savefig(f'output-cifar-10/{output_name}')
    print(f'Gráfico salvo em output-cifar-10/{output_name}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera gráfico de acurácia a partir dos dados JSON')
    parser.add_argument('--non-iid', action='store_true', help='Usar dados non-IID')
    args = parser.parse_args()
    plot_from_json(args.non_iid)
