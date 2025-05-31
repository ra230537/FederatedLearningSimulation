#Server.py

import tensorflow as tf
import threading
import time

class Server:
    def __init__(self, clients, num_clients, num_updates, timeout, local_epochs, batch_size, testing_data):
        self.clients = clients
        self.number_of_clients = num_clients
        self.number_of_updates = num_updates
        self.timeout = timeout
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.global_model = None
        self.testing_data = testing_data

    def create_model(self):
        self.global_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def setup_clients(self):
        for client in self.clients:
            client.setup_client(self.global_model)

    def get_model_weights(self):
        return self.global_model.get_weights()
    
    def aggregate_update(self, client, updated_weights, round):

        print(f"Agregando atualização {round + 1} de {client}.")
        global_weights = self.global_model.get_weights()

        for i in range(len(global_weights)):
            global_weights[i] = (global_weights[i] + updated_weights[i])/2

        self.global_model.set_weights(global_weights)

    def evaluate(self):
        self.global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        loss, accuracy = self.global_model.evaluate(self.testing_data.batch(self.batch_size), verbose=0)
        return loss, accuracy

    def start_training(self):

        threads = []
        for client in self.clients:
            thread = threading.Thread(
                target= client.train_multiple,
                args= (self.number_of_updates, self.local_epochs, self.batch_size, self)
            )
            threads.append((client, thread))
            thread.start()

        s=time.time()

        while time.time() - s < self.timeout:
            done_training = all(not t.is_alive() for _, t in threads)
            if done_training:
                break
            time.sleep(0.5) 

        for client_id, thread in threads:
            if thread.is_alive():
                print(f"Cliente {client_id} excedeu o tempo limite.")

        loss, accuracy = self.evaluate()

        print(f"Treinamento federado assíncrono concluído.")
        print(f"Perda final do modelo global: {loss:.4f}")
        print(f"Acurácia final do modelo global: {accuracy:.4f}")


