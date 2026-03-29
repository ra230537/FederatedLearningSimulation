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
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        raise ImportError(
            "O dataset GTSRB requer tensorflow_datasets. "
            "Instale com: pip install tensorflow-datasets"
        )

    def preprocess(sample):
        image = tf.image.resize(sample["image"], [32, 32])
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(sample["label"], tf.int64)
        return image, label

    ds_train = tfds.load("gtsrb", split="train", shuffle_files=False)
    ds_test = tfds.load("gtsrb", split="test", shuffle_files=False)

    train_images, train_labels = [], []
    for img, lbl in ds_train.map(preprocess):
        train_images.append(img.numpy())
        train_labels.append(lbl.numpy())

    test_images, test_labels = [], []
    for img, lbl in ds_test.map(preprocess):
        test_images.append(img.numpy())
        test_labels.append(lbl.numpy())

    x_train = np.array(train_images)
    y_train = np.array(train_labels)
    x_test = np.array(test_images)
    y_test = np.array(test_labels)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, test_data
