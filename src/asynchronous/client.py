#Client.py
import time
import tensorflow as tf
import random
import threading
from constants import *
random.seed(42)

# Lock global para impedir que múltiplas threads executem fit() simultaneamente no TF
fit_lock = threading.Lock()

class Client:
    def __init__(self, dataset, client_id, train_time_range, speed_tier_name="fast"):
        self.local_model = None
        self.dataset = dataset
        self.client_id = client_id
        self.train_time_range = train_time_range
        self.speed_tier_name = speed_tier_name

    def setup_client(self, model):
        self.local_model = tf.keras.models.clone_model(model)
        self.local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, local_epochs, batch_size):
        # Obtem os tempos de conexão e treinamento simulados
        start_training_time = time.time()
        connection_delay = random.uniform(MIN_CONNECTION_TIME, MAX_CONNECTION_TIME)
        train_delay = random.uniform(*self.train_time_range)
        time.sleep(connection_delay)

        # Usa o lock para garantir que apenas uma thread execute fit() por vez, evitando erros do TensorFlow
        with fit_lock:
            fit_start = time.time()
            self.local_model.fit(self.dataset.batch(batch_size), epochs=local_epochs, verbose=0)
            fit_time = time.time() - fit_start
        # Desconta o tempo real do fit do delay simulado, garantindo que o
        # tempo total de processamento nao ultrapasse o teto do tier do cliente
        remaining_delay = max(0.0, train_delay - fit_time)
        time.sleep(remaining_delay)
        end_training_time = time.time()
        total_time = end_training_time - start_training_time
        total_delay = connection_delay + train_delay
        print(f"[Cliente {self.client_id} | {self.speed_tier_name}] Finalizado em {total_time:.2f}s (Delay simulado: {total_delay:.2f}s | Fit real: {fit_time:.2f}s)")
        return self.local_model.get_weights()

    def train_multiple(self, number_of_updates, local_epochs, batch_size, server):
        for update_round in range(number_of_updates):
            start_time = time.time()
            # Verifica se o timeout do servidor já foi atingido antes de iniciar o treinamento
            if server.event.is_set():
                print(f'[TIMEOUT] Parando a execução do cliente {self.client_id} durante a atualização {update_round + 1}')
                return
            # Atualiza os pesos do modelo local com os pesos do servidor antes de treinar
            self.local_model.set_weights(server.get_model_weights())
            server_version = server.version
            
            # Obtem os pesos atualizados após o treinamento e envia para o servidor
            updated_weights = self.train(local_epochs, batch_size)
            updated_params = updated_weights, server_version, start_time
            print(f"Atualização {update_round + 1} do cliente {self.client_id} com a versão {server_version} do servidor")
            server.aggregate_update(self, updated_params, update_round) 