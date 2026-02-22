#Main.py

import tensorflow as tf
import numpy as np
from client import Client
from scipy.stats import uniform
from server import Server
import matplotlib.pyplot as plt
import json
from constants import *
import math
import argparse
np.random.seed(42)
tf.random.set_seed(42)


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

'''
@train_data: Dados de treinamento
@num_clients: quantidade de clientes
'''
def split_non_iid_data(train_data, num_clients):
    max_classes = 3
    x = []
    y = []
    for i, j in train_data:
        x.append(i.numpy())
        y.append(j.numpy())
    x = np.array(x)
    y = np.array(y)
    
    allowed_clients_by_classes = get_allowed_clients_by_classes(num_clients, max_classes)
    
    class_rows_mapping, clients_number_by_classes = calculate_sample_allocations(allowed_clients_by_classes, y, num_clients)

    samples_x_by_client, samples_y_by_client = get_samples_by_clients(clients_number_by_classes, class_rows_mapping, num_clients, x, y)
    
    client_datasets = []
    for client_idx in range(num_clients):
        #Cria [[x1, x4, x19, x23, x42, ...]]
        client_x = np.concatenate(samples_x_by_client[client_idx])
        client_y = np.concatenate(samples_y_by_client[client_idx])
        dataset = tf.data.Dataset.from_tensor_slices((client_x, client_y))
        client_datasets.append(dataset)
    return client_datasets

def get_allowed_clients_by_classes(num_clients, max_classes):
    # Indica quais clientes vão ter cada classe, cada cliente pode ter no maximo max_classes, porém cada classe vai ter vários clientes (indefinido)
    # {1:[1, 3, 5, 32], 2:[1, 6, 32, ..], ...10:[...]}
    allowed_clients_by_classes = {i: [] for i in range(10)}
    for client in range(num_clients):
        qty_classes = np.random.randint(1, max_classes + 1)
        chosen_classes = np.random.choice(10, qty_classes, replace=False)
        for _class in chosen_classes:
            allowed_clients_by_classes[_class].append(client)
            
    # Agora precisamos garantir que nenhuma classe ficou sem cliente
    for _class in range(10):
        if (len(allowed_clients_by_classes[_class])==0):
            #Atribuo essa classe a algum cliente
            random_client = np.random.randint(num_clients)
            allowed_clients_by_classes[_class].append(random_client)
    return allowed_clients_by_classes

def calculate_sample_allocations(allowed_clients_by_classes, y, num_clients):
    # Um mapeamento que indica o percentual que cada cliente possui de dados da classe (soma 100% para cada classe)
    # {class: [0.1,0.02,...]}
    classes_by_clients_percentage = {i: np.zeros(num_clients) for i in range(10)}
    # Um mapeamento que indica os indices que essa classe apareceu
    # {class: [1, 3, 5, 7}
    class_rows_mapping = {}
    # Indica quantos dados cada cliente vai ter.
    clients_number_by_classes = {}
    for _class in range(10):
        valid_clients = allowed_clients_by_classes[_class]
        dirichlet_result = np.random.dirichlet(alpha=np.ones(len(valid_clients)), size=1)[0]  # tamanho: valid_clients [0.2, 0.3, ...]
        for idx, client_idx in enumerate(valid_clients):
            classes_by_clients_percentage[_class][client_idx] = dirichlet_result[idx]
        class_rows_mapping[_class] = np.where(y == _class)[0]
        np.random.shuffle(class_rows_mapping[_class])
    # Agora eu preciso saber quantos dados cada classe tem
    for _class in range(10):
        for client_percentage in classes_by_clients_percentage[_class]:
            if _class not in clients_number_by_classes:
                clients_number_by_classes[_class] = []
            class_size = len(class_rows_mapping[_class])
            clients_number_by_classes[_class].append(client_percentage*class_size)
    return class_rows_mapping, clients_number_by_classes

def get_samples_by_clients(clients_number_by_classes, class_rows_mapping, num_clients, x, y):
    # {1: [x1, x2, x3, ...], 2: [x13, x25, x399, ...]}
    samples_x_by_client = {i: [] for i in range(num_clients)}
    samples_y_by_client = {i: [] for i in range(num_clients)}
    for _class in range(10):
        current_idx = 0
        for client_idx in range(num_clients):
            # Obtém quantos dados esse cliente vai precisar para essa classe
            qty_samples = math.floor(clients_number_by_classes[_class][client_idx])
            
            # Caso seja o ultimo cliente ele vai pegar os ultimos dados
            if (client_idx == num_clients - 1):
                sample_list = class_rows_mapping[_class][current_idx:]
            # Caso não seja o ultimo ele vai buscar exatamente os dados que falamos com base no mapeamento
            else:
                sample_list = class_rows_mapping[_class][current_idx:current_idx+qty_samples]

            # Obtem os exemplares
            x_samples = x[sample_list]
            y_samples = y[sample_list]

            # Cria vários arrays de dado por cliente, um para cada classe
            # Cria [1:[x1, x4, x19, ...],[x3, x42, ...]]
            samples_x_by_client[client_idx].append(x_samples)
            samples_y_by_client[client_idx].append(y_samples)
            # Atualiza o indice atual
            current_idx+=qty_samples
    return samples_x_by_client, samples_y_by_client

def get_percentiles_timeout(percentile_list, num_updates):
    connection_dist = uniform(loc=MIN_CONNECTION_TIME, scale=MAX_CONNECTION_TIME-MIN_CONNECTION_TIME)
    train_dist = uniform(loc=MIN_TRAIN_TIME, scale=MAX_TRAIN_TIME-MIN_TRAIN_TIME)
    connection_samples = connection_dist.rvs(1_000_000)
    train_samples = train_dist.rvs(1_000_000)
    sum_samples = connection_samples+train_samples
    # Used the value specified in the paper
    timeout = np.percentile(sum_samples, percentile_list) * num_updates
    print(f'Os timeouts são {timeout}')
    return timeout

def main(num_clients, num_updates, epochs, batch_size, is_non_iid):
    accuracy_history = []
    percentile_list = PERCENTILE_LIST
    number_of_clients = num_clients
    number_of_updates = num_updates
    local_epochs = epochs
    batch_size = batch_size

    training_data, testing_data = load_data()
    if is_non_iid:
        print("Usando dados não IID")
        training_data_clients = split_non_iid_data(training_data, number_of_clients)
    else:
        print("Usando dados IID")
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
    # Salvar dados em arquivo JSON para gerar gráficos depois
    data = {}
    for i, percentile in enumerate(percentile_list):
        data[str(percentile)] = [{'loss': p[0], 'accuracy': p[1], 'time': p[2]} for p in accuracy_history[i]]

    accuracy_data_name = 'accuracy_data_non_iid.json' if is_non_iid else 'accuracy_data_iid.json'
    with open(f'output/{accuracy_data_name}', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Dados salvos em output/{accuracy_data_name}')
    for i, percentile in enumerate(percentile_list):
        points = sorted(accuracy_history[i], key=lambda x: x[2])
        accuracy_axis = [p[1] for p in points]
        time_axis = [p[2] for p in points]
        plt.plot(time_axis, accuracy_axis, label=f'{percentile}%')
    plt.xlabel('Tempo de treinamento')
    plt.ylabel('Acurácia do modelo')
    plt.xlim(0, 1000)
    plt.ylim(0.9, 1)
    plt.legend()
    if is_non_iid:
        plt.savefig('output/accuracy_non_iid.png')
    else:
        plt.savefig('output/accuracy_iid.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--non-iid', action='store_true', help='Distribution of data')
    args = parser.parse_args()
    num_clients = NUM_CLIENTS
    num_updates = NUM_UPDATES
    # timeout = 20
    epochs = LOCAL_EPOCHS
    batch_size = BATCH_SIZE

    main(num_clients, num_updates, epochs, batch_size, args.non_iid)