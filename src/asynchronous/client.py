#Client.py
import tensorflow as tf
from constants import *


class Client:
    def __init__(
        self,
        dataset,
        client_id,
        train_time_range,
        speed_tier_name="fast",
    ):
        self.local_model = None
        self.dataset = dataset
        self.client_id = client_id
        self.train_time_range = train_time_range
        self.speed_tier_name = speed_tier_name
        self.completed_updates = 0

    def setup_client(self, model):
        self.local_model = tf.keras.models.clone_model(model)
        self.local_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

    def perform_fit(self, base_weights, local_epochs, batch_size):
        """Executa um round de treinamento local (sem sleeps, sem threading).
        Retorna os pesos atualizados."""
        self.local_model.set_weights(base_weights)
        self.local_model.fit(
            self.dataset.batch(batch_size), epochs=local_epochs, verbose=0
        )
        return self.local_model.get_weights()
