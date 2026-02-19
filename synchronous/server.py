#Server.py

import tensorflow as tf
import threading
import time

class Server:
    def __init__(self, clients, num_clients, round_num, timeout, local_epochs, batch_size, testing_data, is_percentage_boundary = False, percentage_boundary = 1):
        self.clients = clients
        self.number_of_clients = num_clients
        self.number_of_rounds = round_num
        self.start_time = time.time()
        self.timeout = timeout
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.global_model = None
        self.testing_data = testing_data
        self.accuracy_history = []
        self.is_percentage_boundary = is_percentage_boundary
        self.percentage_boundary = percentage_boundary

    def create_model(self):
        self.global_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def setup_clients(self):
        for client in self.clients:
            client.setup_client(self.global_model)
    
    def aggregate_round(self, client_weights, client_sizes, round_num):
        if len(client_weights) == 0:
            print(f'Não houve resposta de nenhum cliente, o modelo não foi modificado')
            return
        print(f"Gerando novo modelo global. Rodada {round_num + 1}.")
        total_size = sum(client_sizes)
        weighted_weights = [
            tf.add_n([
                client_weights[i][weight_idx] * (client_sizes[i] / total_size)
                for i in range(len(client_weights))
            ])
            for weight_idx in range(len(client_weights[0]))
        ]
        loss, accuracy, time_stamp = self.evaluate()
        self.accuracy_history.append((loss, accuracy, time_stamp))
        self.global_model.set_weights(weighted_weights)

    def evaluate(self):
        self.global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        loss, accuracy = self.global_model.evaluate(self.testing_data.batch(self.batch_size), verbose=0)
        now = time.time()
        return loss, accuracy, now - self.start_time

    def distribute_weights(self):
        global_weights = self.global_model.get_weights()
        for client in self.clients:
            client.set_model_weights(global_weights)

    def train_clients(self, round_num):

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

        if round_num == 0:
            time.sleep(1)

        round_start_time = time.time()
        count_done_training = 0
        while self.should_round_running(round_start_time, count_done_training):
            count_done_training = sum(not t.is_alive() for _, t in threads)
            if count_done_training == self.number_of_clients:
                break
            time.sleep(0.001)  

        for client, thread in threads:
            if thread.is_alive():
                print(f"Cliente {client.client_id} excedeu o tempo limite na rodada {round_num}.")
            else:
                client_weights.append(client.get_model_weights())
                client_sizes.append(client.get_dataset_size())
        print(f'Percentual de clientes na rodada {round_num+1}: {100*len(client_weights)/self.number_of_clients}%')
        return client_weights, client_sizes

    def get_timeout(self):
        return self.timeout
    
    def should_round_running(self, round_start_time ,count_done_training):
        if self.is_percentage_boundary:
            return count_done_training/self.number_of_clients < self.percentage_boundary
        return time.time() - round_start_time < self.get_timeout()

    def start_training(self):
        for round_num in range(self.number_of_rounds):
            print(f"\nRodada {round_num + 1}")
            self.distribute_weights()
            client_weights, client_sizes = self.train_clients(round_num)
            self.aggregate_round(client_weights, client_sizes, round_num)
        print("Treinamento concluído. Novo modelo global gerado.")
        loss, accuracy,_ = self.evaluate()

        
        # print(f"Dados historicos: {accuracy_axis}")
        print(f"Treinamento federado síncrono concluído.")
        print(f"Perda final do modelo global: {loss:.4f}")
        print(f"Acurácia final do modelo global: {accuracy:.4f}")
        return self.accuracy_history
            


