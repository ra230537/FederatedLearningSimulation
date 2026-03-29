# Main.py

import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import argparse
import json

import numpy as np
import tensorflow as tf
from client import Client
from constants import (
    BATCH_SIZE,
    LOCAL_EPOCHS,
    MAX_CONNECTION_TIME,
    MAX_TRAIN_TIME,
    MIN_CONNECTION_TIME,
    MIN_TRAIN_TIME,
    NUM_CLIENTS,
    NUM_UPDATES,
    PERCENTILE_LIST,
)
from monte_carlo import get_percentiles_timeout
from server import Server

from utils.data_loader import get_dataset_info, load_dataset
from utils.data_split import split_iid_data, split_non_iid_data
from utils.plot_accuracy import generate_all_plots

np.random.seed(42)
tf.random.set_seed(42)


def main(
    num_clients,
    num_updates,
    epochs,
    batch_size,
    is_non_iid,
    base_alpha=0.8,
    decay_of_base_alpha=0.999,
    tardiness_sensivity=0.075,
    dataset="cifar10",
    output_prefix="",
    single_percentile=None,
):
    accuracy_history = []
    percentile_list = [single_percentile] if single_percentile else PERCENTILE_LIST
    number_of_clients = num_clients
    number_of_updates = num_updates
    local_epochs = epochs
    batch_size = batch_size

    dataset_info = get_dataset_info(dataset)
    training_data, testing_data = load_dataset(dataset)
    if is_non_iid:
        print("Usando dados não IID")
        training_data_clients = split_non_iid_data(training_data, number_of_clients, dataset_info["num_classes"])
    else:
        print("Usando dados IID")
        training_data_clients = split_iid_data(training_data, number_of_clients, dataset_info["num_classes"])
    percentiles_timeout = get_percentiles_timeout(
        percentile_list,
        number_of_updates,
        MIN_CONNECTION_TIME,
        MAX_CONNECTION_TIME,
        MIN_TRAIN_TIME,
        MAX_TRAIN_TIME,
    )
    for i in range(len(percentiles_timeout)):
        timeout = percentiles_timeout[i]
        percentile = percentile_list[i]
        print(f"Timeout definido para {percentile}%: {timeout}")
        clients = [
            Client(training_data_clients[i], i + 1) for i in range(number_of_clients)
        ]

        server = Server(
            clients,
            number_of_clients,
            number_of_updates,
            timeout,
            local_epochs,
            batch_size,
            testing_data,
            dataset_info["model"],
            base_alpha,
            decay_of_base_alpha,
            tardiness_sensivity,
        )

        server.setup_clients()
        local_history = server.start_training()
        accuracy_history.append(local_history)
    # Salvar dados em arquivo JSON para gerar gráficos depois
    data = {}
    for i, percentile in enumerate(percentile_list):
        data[str(percentile)] = [
            {"loss": p[0], "accuracy": p[1], "time": p[2]} for p in accuracy_history[i]
        ]

    tipo_dist = "non_iid" if is_non_iid else "iid"
    prefix_str = f"_{output_prefix}" if output_prefix else ""
    accuracy_data_name = f"accuracy_data_{tipo_dist}{prefix_str}.json"

    output_dir = dataset_info["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{accuracy_data_name}", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Dados salvos em {output_dir}/{accuracy_data_name}")

    if not output_prefix:
        generate_all_plots(
            output_dir, is_non_iid, alpha=0.1, x_label="atualizações"
        )

    tf.keras.backend.clear_session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--num-updates", type=int, default=NUM_UPDATES)
    parser.add_argument("--epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--non-iid", action="store_true", help="Roda apenas não-IID")
    parser.add_argument("--iid", action="store_true", help="Roda apenas IID")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "fashion_mnist", "gtsrb"], help="Dataset a usar (default: cifar10)")
    parser.add_argument("--base-alpha", type=float, default=0.8)
    parser.add_argument("--decay-of-base-alpha", type=float, default=0.999)
    parser.add_argument("--tardiness-sensivity", type=float, default=0.075)
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--percentile", type=int, default=None, help="Percentil unico (ex: 50). Padrao: todos de PERCENTILE_LIST")
    args = parser.parse_args()

    if args.non_iid:
        distributions = [True]
    elif args.iid:
        distributions = [False]
    else:
        distributions = [False, True]

    for is_non_iid in distributions:
        main(
            num_clients=args.num_clients,
            num_updates=args.num_updates,
            epochs=args.epochs,
            batch_size=args.batch_size,
            is_non_iid=is_non_iid,
            dataset=args.dataset,
            base_alpha=args.base_alpha,
            decay_of_base_alpha=args.decay_of_base_alpha,
            tardiness_sensivity=args.tardiness_sensivity,
            output_prefix=args.output_prefix,
            single_percentile=args.percentile,
        )
