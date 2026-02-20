#Main.py

import tensorflow as tf
import numpy as np
from client import Client
from scipy.stats import triang
from server import Server

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
        split_datasets.append(shuffled_data.skip(start).take(size))
        start += size

    return split_datasets

def main(num_clients, num_updates, timeout, epochs, batch_size):

    number_of_clients = num_clients
    number_of_updates = num_updates
    timeout = timeout
    local_epochs = epochs
    batch_size = batch_size

    training_data, testing_data = load_data()
    training_data_clients = split_data_random(training_data, number_of_clients)

    clients = [Client(training_data_clients[i], i+1) for i in range(number_of_clients)]

    server = Server(clients, number_of_clients, number_of_updates, timeout, local_epochs, batch_size, testing_data)

    server.create_model()
    server.setup_clients()
    server.start_training()


if __name__ == "__main__":
    min_time_value = 0.1
    max_time_value = 5
    min_value = min_time_value*2
    max_value = max_time_value*2
    moda = (min_time_value + max_time_value)/2+(min_time_value+max_time_value)/2
    c = (moda - min_value)/(max_value - min_value)
    timeout_sync = triang.ppf(0.50, c, loc=min_value, scale=max_value-min_value)
    num_clients = 5
    num_updates = 5
    print(f'timeout definido para 50%: {timeout_sync}')
    # timeout = 20
    epochs = 1
    batch_size = 32

    main(num_clients, num_updates, timeout_sync, epochs, batch_size)