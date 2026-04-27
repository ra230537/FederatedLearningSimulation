import numpy as np
import math

np.random.seed(42)


def split_iid_data(train_data, num_clients, num_classes=10):
    x, y = train_data
    n = len(x)
    percentages = np.random.dirichlet(alpha=np.ones(num_clients), size=1)[0]
    split_sizes = (n * percentages).astype(int)
    split_sizes[-1] += n - split_sizes.sum()

    indices = np.random.permutation(n)
    shuffled_x = x[indices]
    shuffled_y = y[indices]

    split_datasets = []
    start = 0
    for size in split_sizes:
        split_datasets.append((shuffled_x[start : start + size], shuffled_y[start : start + size]))
        start += size

    return split_datasets


def split_non_iid_data(train_data, num_clients, num_classes=10):
    max_classes = 3
    x, y = train_data

    allowed_clients_by_classes = get_allowed_clients_by_classes(num_clients, max_classes, num_classes)

    sample_idx_by_classes, clients_size_by_classes = calculate_sample_allocations(
        allowed_clients_by_classes, y, num_clients, num_classes
    )

    samples_x_by_client, samples_y_by_client = get_samples_by_clients(
        clients_size_by_classes, sample_idx_by_classes, num_clients, x, y, num_classes
    )

    client_datasets = []
    for client_idx in range(num_clients):
        client_x = np.concatenate(samples_x_by_client[client_idx])
        client_y = np.concatenate(samples_y_by_client[client_idx])
        shuffle_idx = np.random.permutation(len(client_x))
        client_x = client_x[shuffle_idx]
        client_y = client_y[shuffle_idx]
        client_datasets.append((client_x, client_y))
    return client_datasets


def get_allowed_clients_by_classes(num_clients, max_classes, num_classes=10):
    allowed_clients_by_classes = {i: [] for i in range(num_classes)}
    for client in range(num_clients):
        qty_classes = np.random.randint(1, max_classes + 1)
        chosen_classes = np.random.choice(num_classes, qty_classes, replace=False)
        for _class in chosen_classes:
            allowed_clients_by_classes[_class].append(client)

    for _class in range(num_classes):
        if len(allowed_clients_by_classes[_class]) == 0:
            random_client = np.random.randint(num_clients)
            allowed_clients_by_classes[_class].append(random_client)
    return allowed_clients_by_classes


def calculate_sample_allocations(allowed_clients_by_classes, y, num_clients, num_classes=10):
    classes_by_clients_percentage = {i: np.zeros(num_clients) for i in range(num_classes)}
    sample_idx_by_classes = {}
    clients_size_by_classes = {}
    for _class in range(num_classes):
        valid_clients = allowed_clients_by_classes[_class]
        dirichlet_result = np.random.dirichlet(alpha=np.ones(len(valid_clients)), size=1)[0]
        for idx, client_idx in enumerate(valid_clients):
            classes_by_clients_percentage[_class][client_idx] = dirichlet_result[idx]
        sample_idx_by_classes[_class] = np.where(y == _class)[0]
        np.random.shuffle(sample_idx_by_classes[_class])

    for _class in range(num_classes):
        for client_percentage in classes_by_clients_percentage[_class]:
            if _class not in clients_size_by_classes:
                clients_size_by_classes[_class] = []
            class_size = len(sample_idx_by_classes[_class])
            clients_size_by_classes[_class].append(client_percentage * class_size)
    return sample_idx_by_classes, clients_size_by_classes


def get_samples_by_clients(clients_size_by_classes, sample_idx_by_classes, num_clients, x, y, num_classes=10):
    samples_x_by_client = {i: [] for i in range(num_clients)}
    samples_y_by_client = {i: [] for i in range(num_clients)}
    for _class in range(num_classes):
        current_idx = 0
        for client_idx in range(num_clients):
            qty_samples = math.floor(clients_size_by_classes[_class][client_idx])
            if client_idx == num_clients - 1:
                sample_list = sample_idx_by_classes[_class][current_idx:]
            else:
                sample_list = sample_idx_by_classes[_class][current_idx : current_idx + qty_samples]

            x_samples = x[sample_list]
            y_samples = y[sample_list]

            samples_x_by_client[client_idx].append(x_samples)
            samples_y_by_client[client_idx].append(y_samples)
            current_idx += qty_samples
    return samples_x_by_client, samples_y_by_client
