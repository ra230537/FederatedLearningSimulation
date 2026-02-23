import numpy as np
import tensorflow as tf
import math
np.random.seed(42)
tf.random.set_seed(42)

def split_iid_data(train_data, num_clients):
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