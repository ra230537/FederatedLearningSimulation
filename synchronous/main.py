#Main.py

import tensorflow as tf
import numpy as np
from client import Client
from server import Server
import matplotlib.pyplot as plt
import numpy as np
import json
from constants import *

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, test_data

def split_data_random(train_data, num_clients):
    percentages = np.random.dirichlet(alpha=np.ones(num_clients), size=1)[0]  

    shuffled_data = train_data.shuffle(buffer_size=60000)
    dataset_size = shuffled_data.cardinality().numpy()

    split_sizes = (dataset_size * percentages).astype(int)  

    split_datasets = []
    start = 0
    for size in split_sizes:
        split_datasets.append(shuffled_data.skip(start).take(size))
        start += size

    return split_datasets

def main(num_clients, round_num, timeout, epochs, batch_size):

    number_of_clients = num_clients
    number_of_rounds = round_num
    timeout = timeout
    local_epochs = epochs
    batch_size = batch_size

    training_data, testing_data = load_data()
    training_data_clients = split_data_random(training_data, number_of_clients)
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
    with open('output/accuracy_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print('Dados salvos em output/accuracy_data.json')
    for i, boundary in enumerate(boundary_list):
        points = sorted(accuracy_history[i], key=lambda x: x[2])
        accuracy_axis = [p[1] for p in points]
        time_axis = [p[2] for p in points]
        plt.plot(time_axis, accuracy_axis, label=f'{boundary * 100}%')
    plt.xlabel('Tempo de treinamento')
    plt.ylabel('Acurácia do modelo')
    plt.xlim(0, 1000)
    plt.ylim(0.9, 1)
    plt.legend()
    plt.savefig('output/accuracy.png')


if __name__ == "__main__":
    num_clients = NUM_CLIENTS
    round_num = NUM_UPDATES
    timeout = TIMEOUT
    epochs = LOCAL_EPOCHS
    batch_size = BATCH_SIZE

    main(num_clients, round_num, timeout, epochs, batch_size)