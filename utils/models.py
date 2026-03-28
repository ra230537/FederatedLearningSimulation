import tensorflow as tf


def cnn_cifar10() -> tf.keras.Model:
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(32, 32, 3),  # pyright: ignore[reportCallIssue]
            ),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def cnn_mnist() -> tf.keras.Model:
    # Input: 28x28x1 (grayscale), 10 classes
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(28, 28, 1),  # pyright: ignore[reportCallIssue]
            ),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def cnn_fashion_mnist() -> tf.keras.Model:
    # Input: 28x28x1 (grayscale), 10 classes
    # Slightly deeper than MNIST to handle more complex textures
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(28, 28, 1),  # pyright: ignore[reportCallIssue]
            ),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def get_model(model_name) -> tf.keras.Model:
    options = {
        "cnn": cnn_cifar10,
        "cnn_cifar10": cnn_cifar10,
        "cnn_mnist": cnn_mnist,
        "cnn_fashion_mnist": cnn_fashion_mnist,
    }
    if model_name not in options:
        raise ValueError(
            f"Modelo inválido '{model_name}'. Opções: {list(options.keys())}"
        )
    return options[model_name]()
