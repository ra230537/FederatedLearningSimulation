# Main.py

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.plot_accuracy import generate_all_plots
from utils.data_split import split_iid_data, split_non_iid_data
from utils.data_loader import get_dataset_info, load_dataset
from server import Server
from monte_carlo import get_percentiles_timeout
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
    TIMEOUT,
)
from client import Client
import tensorflow as tf
import numpy as np
import json
import argparse



np.random.seed(42)
tf.random.set_seed(42)


def main(num_clients, round_num, epochs, batch_size, is_non_iid, dataset="cifar10", output_prefix="", single_percentile=None, include_no_timeout=False):
    accuracy_history = []
    number_of_clients = num_clients
    percentile_list = [
        single_percentile] if single_percentile else PERCENTILE_LIST
    local_epochs = epochs
    batch_size = batch_size

    dataset_info = get_dataset_info(dataset)
    training_data, testing_data = load_dataset(dataset)
    if is_non_iid:
        print("Usando dados não IID")
        training_data_clients = split_non_iid_data(
            training_data, number_of_clients, dataset_info["num_classes"])
    else:
        print("Usando dados IID")
        training_data_clients = split_iid_data(
            training_data, number_of_clients, dataset_info["num_classes"])
    percentiles_timeout = get_percentiles_timeout(
        percentile_list,
        MIN_CONNECTION_TIME,
        MAX_CONNECTION_TIME,
        MIN_TRAIN_TIME,
        MAX_TRAIN_TIME,
    )

    runs = [(str(p), t) for p, t in zip(percentile_list, percentiles_timeout)]
    if include_no_timeout:
        runs.append(("include_no_timeout", MAX_CONNECTION_TIME + MAX_TRAIN_TIME))

    for label, run_timeout in runs:
        print(f"\nTimeout definido para '{label}': {run_timeout:.2f}s")
        clients = [
            Client(training_data_clients[i], i + 1) for i in range(number_of_clients)
        ]

        server = Server(
            clients,
            number_of_clients,
            round_num,
            run_timeout,
            local_epochs,
            batch_size,
            testing_data,
            dataset_info["model"],
        )

        server.setup_clients()
        local_history = server.start_training()
        accuracy_history.append(local_history)

    # Salvar dados em arquivo JSON para gerar gráficos depois
    data = {}
    for i, (label, _) in enumerate(runs):
        data[label] = [
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
        generate_all_plots(output_dir, is_non_iid,
                           alpha=0.1, x_label="rodadas")

    tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-iid", action="store_true",
                        help="Roda apenas não-IID")
    parser.add_argument("--iid", action="store_true", help="Roda apenas IID")
    parser.add_argument("--include-no-timeout", action="store_true",
                        help="Adiciona cenário sem timeout (T=MAX_CONNECTION+MAX_TRAIN)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=[
                        "cifar10", "mnist", "fashion_mnist", "gtsrb"], help="Dataset a usar (default: cifar10)")
    parser.add_argument("--percentile", type=int, default=None,
                        help="Percentil unico (ex: 50). Padrao: todos de PERCENTILE_LIST")
    args = parser.parse_args()
    num_clients = NUM_CLIENTS
    round_num = NUM_UPDATES
    timeout = TIMEOUT
    epochs = LOCAL_EPOCHS
    batch_size = BATCH_SIZE

    if args.non_iid:
        distributions = [True]
    elif args.iid:
        distributions = [False]
    else:
        distributions = [False, True]

    for is_non_iid in distributions:
        main(num_clients, round_num, epochs, batch_size, is_non_iid,
             dataset=args.dataset, single_percentile=args.percentile, include_no_timeout=args.include_no_timeout)
