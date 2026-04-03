#Client.py
import time
import tensorflow as tf
import random
import threading
from constants import *
random.seed(42)

# Lock global para impedir multiplos fit() simultaneos no TensorFlow.
fit_lock = threading.Lock()

class Client:
    def __init__(self, dataset, client_id):
        self.local_model = None
        self.dataset = dataset
        self.client_id = client_id
        self._state_lock = threading.Lock()
        self._is_training = False
        self._has_fresh_update = False

    def setup_client(self, model):
        self.local_model = tf.keras.models.clone_model(model)
        self.local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, local_epochs, batch_size):
        with self._state_lock:
            if self._is_training:
                self._has_fresh_update = False
                print(f"Cliente {self.client_id} ainda estava treinando; treino desta rodada foi ignorado.")
                return
            self._is_training = True

        start_training_time = time.time()
        with self._state_lock:
            self._has_fresh_update = False
        try:
            connection_delay = random.uniform(MIN_CONNECTION_TIME, MAX_CONNECTION_TIME)
            train_delay = random.uniform(MIN_TRAIN_TIME, MAX_TRAIN_TIME)
            time.sleep(connection_delay)
            with fit_lock:
                fit_start = time.time()
                self.local_model.fit(self.dataset.batch(batch_size), epochs=local_epochs, verbose=0)
                fit_time = time.time() - fit_start
            # Desconta o tempo real do fit do delay simulado, garantindo que o
            # tempo total de processamento nao ultrapasse MAX_TRAIN_TIME
            remaining_delay = max(0.0, train_delay - fit_time)
            time.sleep(remaining_delay)
            end_training_time = time.time()
            with self._state_lock:
                self._has_fresh_update = True
            print(f"Tempo de Execução do cliente {self.client_id}: {end_training_time-start_training_time:.2f}s (Fit real: {fit_time:.2f}s)")
        finally:
            with self._state_lock:
                self._is_training = False

    def get_dataset_size(self):
        size = self.dataset.cardinality().numpy()
        return size
    
    def set_model_weights(self, weights):
        with self._state_lock:
            if self._is_training:
                return False
            self.local_model.set_weights(weights)
            self._has_fresh_update = False
            return True
    
    def get_model_weights(self):
        return self.local_model.get_weights()

    def has_fresh_update(self):
        with self._state_lock:
            return self._has_fresh_update