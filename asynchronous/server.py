#Server.py

import tensorflow as tf
import threading
import time
import matplotlib.pyplot as plt

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
        self.accuracy_history = []
        self.start_time = time.time()

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

        print(f"Agregando atualização {round + 1} do cliente {client.client_id}.")
        global_weights = self.global_model.get_weights()

        for i in range(len(global_weights)):
            global_weights[i] = (global_weights[i] + updated_weights[i])/2
        loss, accuracy, time_stamp = self.evaluate()
        self.accuracy_history.append((loss, accuracy, time_stamp))
        self.global_model.set_weights(global_weights)

    def evaluate(self):
        self.global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        loss, accuracy = self.global_model.evaluate(self.testing_data.batch(self.batch_size), verbose=0)
        now = time.time()
        return loss, accuracy, now - self.start_time

    def start_training(self):

        threads = []
        for client in self.clients:
            thread = threading.Thread(
                target= client.train_multiple,
                args= (self.number_of_updates, self.local_epochs, self.batch_size, self)
            )
            threads.append((client, thread))
            thread.start()


        while time.time() - self.start_time < self.timeout:
            done_training = all(not t.is_alive() for _, t in threads)
            if done_training:
                break
            time.sleep(0.5) 

        for client, thread in threads:
            if thread.is_alive():
                print(f"Cliente {client.client_id} excedeu o tempo limite.")

        loss, accuracy, now = self.evaluate()
        accuracy_axis = [self.accuracy_history[i][1] for i in range(len(self.accuracy_history))]
        time_axis = [self.accuracy_history[i][2] for i in range(len(self.accuracy_history))]
        print(f"Dados historicos: {accuracy_axis}")
        print(f"Treinamento federado assíncrono concluído.")
        print(f"Perda final do modelo global: {loss:.4f}")
        print(f"Acurácia final do modelo global: {accuracy:.4f}")
        plt.plot(time_axis, accuracy_axis)
        plt.savefig('output/accuracy.png')


