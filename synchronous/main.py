#Main.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import numpy as np
from client import Client
from server import Server
import json
from constants import *
import argparse
from utils.data_split import split_non_iid_data, split_iid_data
from utils.plot_accuracy import generate_all_plots
np.random.seed(42)
tf.random.set_seed(42)

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, test_data


def main(num_clients, round_num, timeout, epochs, batch_size, is_non_iid):
    number_of_clients = num_clients
    number_of_rounds = round_num
    timeout = timeout
    local_epochs = epochs
    batch_size = batch_size

    training_data, testing_data = load_data()
    if is_non_iid:
        print("Usando dados não IID")
        training_data_clients = split_non_iid_data(training_data, number_of_clients)
    else:
        print("Usando dados IID")
        training_data_clients = split_iid_data(training_data, number_of_clients)
    boundary_list = []
    accuracy_history = []
    for boundary in PERCENTILE_LIST:
        boundary = boundary / 100
        print(f'Percentual atual: {boundary * 100}%')
        boundary_list.append(boundary)
        clients = [Client(training_data_clients[i], i+1) for i in range(number_of_clients)]

        server = Server(clients, number_of_clients, number_of_rounds, timeout, local_epochs, batch_size, testing_data, True, boundary)

        server.create_model()
        server.setup_clients()
        local_history = server.start_training()
        accuracy_history.append(local_history)
    # Salvar dados em arquivo JSON para gerar gráficos depois
    data = {}
    for i, boundary in enumerate(boundary_list):
        data[str(boundary * 100)] = [{'loss': p[0], 'accuracy': p[1], 'time': p[2]} for p in accuracy_history[i]]
    accuracy_data_name = 'accuracy_data_non_iid.json' if is_non_iid else 'accuracy_data_iid.json'
    with open(f'output-cifar-10/{accuracy_data_name}', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Dados salvos em output-cifar-10/{accuracy_data_name}')

    generate_all_plots('output-cifar-10', is_non_iid, alpha=0.1, x_label='rodadas')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-iid", action="store_true", help="Distribution of data")
    args = parser.parse_args()
    num_clients = NUM_CLIENTS
    round_num = NUM_UPDATES
    timeout = TIMEOUT
    epochs = LOCAL_EPOCHS
    batch_size = BATCH_SIZE

    main(num_clients, round_num, timeout, epochs, batch_size, args.non_iid)