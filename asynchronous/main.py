#Main.py

import tensorflow as tf
import numpy as np
from client import Client
from scipy.stats import uniform
from server import Server
import matplotlib.pyplot as plt

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
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
        split_datasets.append(shuffled_data.skip(start) .take(size))
        start += size

    return split_datasets

def get_percentiles_timeout(percentile_list, num_updates):
    connection_dist = uniform(loc=0.1, scale=5)
    train_dist = uniform(loc=0.1, scale=5)
    connection_samples = connection_dist.rvs(1_000_000)
    train_samples = train_dist.rvs(1_000_000)
    sum_samples = connection_samples+train_samples
    # Used the value specified in the paper
    timeout = np.percentile(sum_samples, percentile_list) * num_updates
    print(f'Os timeouts são {timeout}')
    return timeout

def main(num_clients, num_updates, epochs, batch_size):
    accuracy_history = []
    percentile_list = [25, 50, 75]
    number_of_clients = num_clients
    number_of_updates = num_updates
    local_epochs = epochs
    batch_size = batch_size

    training_data, testing_data = load_data()
    training_data_clients = split_data_random(training_data, number_of_clients)
    percentiles_timeout = get_percentiles_timeout(percentile_list, number_of_updates)
    for i in range(len(percentiles_timeout)):
        timeout = percentiles_timeout[i]
        percentile = percentile_list[i]
        print(f'Timeout definido para {percentile}%: {timeout}')
        clients = [Client(training_data_clients[i], i+1) for i in range(number_of_clients)]

        server = Server(clients, number_of_clients, number_of_updates, timeout, local_epochs, batch_size, testing_data)

        server.create_model()
        server.setup_clients()
        local_history = server.start_training()
        accuracy_history.append(local_history)
    print(f'Quantidade de pontos na primeira execução: {len(accuracy_history[0])}')
    print(f'Quantidade de pontos na segunda execução: {len(accuracy_history[1])}')
    print(f'Quantidade de pontos na terceira execução: {len(accuracy_history[2])}')
    for i, percentile in enumerate(percentile_list):
        points = sorted(accuracy_history[i], key=lambda x: x[2])
        accuracy_axis = [p[1] for p in points]
        time_axis = [p[2] for p in points]
        plt.plot(time_axis, accuracy_axis, label=f'{percentile}%')
    plt.xlabel('Tempo de treinamento')
    plt.ylabel('Acurácia do modelo')
    plt.legend()
    plt.savefig('output/accuracy.png')



if __name__ == "__main__":

    
    # print(percentile_samples)
    num_clients = 5
    num_updates = 5
    # timeout = 20
    epochs = 1
    batch_size = 32

    main(num_clients, num_updates, epochs, batch_size)