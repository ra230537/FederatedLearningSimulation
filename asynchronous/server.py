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
        self.version = 0
        self.BASE_ALPHA = 0.8
        self.DECAY_OF_BASE_ALPHA = 0.999
        self.TARDINESS_SENSITIVITY = 0.075

        self.lock = threading.Lock()
        self.event = threading.Event()

    def create_model(self):
        self.global_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def setup_clients(self):
        for client in self.clients:
            client.setup_client(self.global_model)

    def get_model_weights(self):
        return self.global_model.get_weights()
    
    def aggregate_update(self, client, updated_params, round):
        # Garante que nós não vamos agregar nada depois do timeout
        if self.event.is_set():
            return
        updated_weights, client_version, start_training_time = updated_params

        print(f"Agregando atualização {round + 1} do cliente {client.client_id}.")

        # Evitar concorrência nas operações
        with self.lock:
            global_weights = self.global_model.get_weights()
            self.update_global_weights(global_weights, updated_weights, client_version, start_training_time)
            
            loss, accuracy, time_stamp = self.evaluate()
            self.accuracy_history.append((loss, accuracy, time_stamp))
            self.global_model.set_weights(global_weights)

            # Atualizar versão do servidor
            self.version += 1

    def evaluate(self):
        
        loss, accuracy = self.global_model.evaluate(self.testing_data.batch(self.batch_size), verbose=0)
        now = time.time()
        return loss, accuracy, now - self.start_time

    def update_global_weights(self, global_weights, updated_weights, client_version, start_training_time):
        delay = time.time() - start_training_time
        staleness = self.version - client_version
        agg_factor = self.get_aggregation_factor(staleness, delay)
        for i in range(len(global_weights)):
            global_weights[i] = global_weights[i]*(1-agg_factor) + agg_factor*updated_weights[i]

    def get_aggregation_factor(self, staleness, delay):
        return self.BASE_ALPHA*(self.DECAY_OF_BASE_ALPHA**staleness)*(1/(1+self.TARDINESS_SENSITIVITY*delay))

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
        # Make the flag true and we put a condition to stop the training if it is true.
        self.event.set()

        loss, accuracy, now = self.evaluate()
        accuracy_axis = [self.accuracy_history[i][1] for i in range(len(self.accuracy_history))]
        time_axis = [self.accuracy_history[i][2] for i in range(len(self.accuracy_history))]
        print(f"Dados historicos: {accuracy_axis}")
        print(f"Treinamento federado assíncrono concluído.")
        print(f"Perda final do modelo global: {loss:.4f}")
        print(f"Acurácia final do modelo global: {accuracy:.4f}")
        plt.plot(time_axis, accuracy_axis)
        plt.savefig('output/accuracy.png')


