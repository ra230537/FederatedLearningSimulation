#Client.py
import time
import tensorflow as tf
import random as rn
from constants import *
rn.seed(42)

class Client:
    def __init__(self, dataset, client_id):
        self.local_model = None
        self.dataset = dataset
        self.client_id = client_id

    def setup_client(self, model):
        self.local_model = tf.keras.models.clone_model(model)
        self.local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, local_epochs, batch_size):
        start_training_time = time.time()
        connection_delay = random.uniform(MIN_CONNECTION_TIME, MAX_CONNECTION_TIME)
        train_delay = random.uniform(MIN_TRAIN_TIME, MAX_TRAIN_TIME)
        total_delay = connection_delay + train_delay
        time.sleep(total_delay)
        self.local_model.fit(self.dataset.batch(batch_size), epochs=local_epochs, verbose=0)
        end_training_time = time.time()
        total_time = end_training_time - start_training_time
        print(f"[Cliente {self.client_id}] Finalizado em {total_time:.2f}s (Delay: {total_delay:.2f}s)")
        return self.local_model.get_weights()

    def train_multiple(self, number_of_updates, local_epochs, batch_size, server):
        for update_round in range(number_of_updates):
            start_time = time.time()
            if server.event.is_set():
                print(f'[TIMEOUT] Parando a execução do cliente {self.client_id} durante a atualização {update_round + 1}')
                return
            self.local_model.set_weights(server.get_model_weights())
            server_version = server.version
            updated_weights = self.train(local_epochs, batch_size)
            updated_params = updated_weights, server_version, start_time
            print(f"Atualização {update_round + 1} do cliente {self.client_id} com a versão {server_version} do servidor")
            server.aggregate_update(self, updated_params, update_round) 