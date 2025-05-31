#Server.py

import tensorflow as tf
import threading
import time

class Server:
    def __init__(self, clients, num_clients, round_num, timeout, local_epochs, batch_size, testing_data):
        self.clients = clients
        self.number_of_clients = num_clients
        self.number_of_rounds = round_num
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
    
    def aggregate_round(self, client_weights, client_sizes, round):

        print(f"Gerando novo modelo global. Rodada {round + 1}.")
        total_size = sum(client_sizes)
        weighted_weights = [
            tf.add_n([
                client_weights[i][weight_idx] * (client_sizes[i] / total_size)
                for i in range(len(client_weights))
            ])
            for weight_idx in range(len(client_weights[0]))
        ]
        self.global_model.set_weights(weighted_weights)

    def evaluate(self):
        self.global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        loss, accuracy = self.global_model.evaluate(self.testing_data.batch(self.batch_size), verbose=0)
        return loss, accuracy

    def distribute_weights(self):
        global_weights = self.global_model.get_weights()
        for client in self.clients:
            client.set_model_weights(global_weights)

    def train_clients(self, round):

        client_weights = []
        client_sizes = []
        threads = []

        for client in self.clients:
            thread = threading.Thread(
                target= client.train,
                args= (self.local_epochs, self.batch_size)
            )
            threads.append((client, thread))
            thread.start()

        if round == 0:
            time.sleep(1)

        s=time.time()

        while time.time() - s < self.timeout:
            done_training = all(not t.is_alive() for _, t in threads)
            if done_training:
                break
            time.sleep(0.5)  

        for client, thread in threads:
            if thread.is_alive():
                print(f"Cliente {client} excedeu o tempo limite na rodada {round}.")
            else:
                client_weights.append(client.get_model_weights())
                client_sizes.append(client.get_dataset_size())

        return client_weights, client_sizes

    def start_training(self):
        for round_num in range(self.number_of_rounds):
            print(f"\nRodada {round_num + 1}")
            self.distribute_weights()
            client_weights, client_sizes = self.train_clients(round_num)
            self.aggregate_round(client_weights, client_sizes, round_num)
        print("Treinamento concluído. Novo modelo global gerado.")
        loss, accuracy = self.evaluate()
        print(f"Avaliação do modelo global - Perda: {loss:.4f}, Acurácia: {accuracy:.4f}")
            


