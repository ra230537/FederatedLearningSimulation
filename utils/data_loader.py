import numpy as np
import tensorflow as tf

DATASET_INFO = {
    "cifar10": {
        "num_classes": 10,
        "input_shape": (32, 32, 3),
        "model": "cnn_cifar10",
        "output_dir": "output-cifar-10",
    },
    "mnist": {
        "num_classes": 10,
        "input_shape": (28, 28, 1),
        "model": "cnn_mnist",
        "output_dir": "output-mnist",
    },
    "fashion_mnist": {
        "num_classes": 10,
        "input_shape": (28, 28, 1),
        "model": "cnn_fashion_mnist",
        "output_dir": "output-fashion-mnist",
    },
    "gtsrb": {
        "num_classes": 43,
        "input_shape": (32, 32, 3),
        "model": "cnn_gtsrb",
        "output_dir": "output-gtsrb",
    },
}

VALID_DATASETS = list(DATASET_INFO.keys())

'''
Recebe o nome do dataset
Retorna suas classes, seu shape, nome do dicionario de models.py e o nome de sua pasta
'''
def get_dataset_info(dataset_name: str) -> dict:
    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"Dataset inválido '{dataset_name}'. Opções: {VALID_DATASETS}"
        )
    return DATASET_INFO[dataset_name]

'''
Recebe o nome do dataset
retorna seus dados
'''
def load_dataset(dataset_name: str):
    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"Dataset inválido '{dataset_name}'. Opções: {VALID_DATASETS}"
        )
    loaders = {
        "cifar10": _load_cifar10,
        "mnist": _load_mnist,
        "fashion_mnist": _load_fashion_mnist,
        "gtsrb": _load_gtsrb,
    }
    return loaders[dataset_name]()


def _load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, test_data


def _load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis].astype("float32") / 255.0
    x_test = x_test[..., np.newaxis].astype("float32") / 255.0
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, test_data


def _load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[..., np.newaxis].astype("float32") / 255.0
    x_test = x_test[..., np.newaxis].astype("float32") / 255.0
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, test_data


def _load_gtsrb():
    import os
    import urllib.request
    import zipfile

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "gtsrb")
    train_dir = os.path.join(data_dir, "GTSRB", "Final_Training", "Images")
    test_dir = os.path.join(data_dir, "GTSRB", "Final_Test", "Images")

    train_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    test_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
    test_gt_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"

    os.makedirs(data_dir, exist_ok=True)

    def download_and_extract(url, dest):
        zip_path = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(zip_path):
            print(f"Baixando {os.path.basename(url)}...")
            urllib.request.urlretrieve(url, zip_path)
        print(f"Extraindo {os.path.basename(url)}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)

    if not os.path.exists(train_dir):
        download_and_extract(train_url, data_dir)
    if not os.path.exists(test_dir):
        download_and_extract(test_url, data_dir)
    if not os.path.exists(os.path.join(data_dir, "GT-final_test.csv")):
        download_and_extract(test_gt_url, data_dir)

    import csv
    from PIL import Image

    def load_ppm_image(path):
        img = Image.open(path).resize((32, 32))
        return np.array(img, dtype="float32") / 255.0

    # Treino: cada subpasta (00000-00042) é uma classe com CSV de anotações
    train_images, train_labels = [], []
    for class_id in range(43):
        class_dir = os.path.join(train_dir, f"{class_id:05d}")
        csv_file = os.path.join(class_dir, f"GT-{class_id:05d}.csv")
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                img_path = os.path.join(class_dir, row["Filename"])
                train_images.append(load_ppm_image(img_path))
                train_labels.append(class_id)

    # Teste: imagens soltas em uma pasta, labels vêm do CSV
    test_csv = os.path.join(data_dir, "GT-final_test.csv")
    test_images, test_labels = [], []
    with open(test_csv, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            img_path = os.path.join(test_dir, row["Filename"])
            test_images.append(load_ppm_image(img_path))
            test_labels.append(int(row["ClassId"]))

    x_train = np.array(train_images)
    y_train = np.array(train_labels)
    x_test = np.array(test_images)
    y_test = np.array(test_labels)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, test_data
