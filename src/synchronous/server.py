# server.py

import time
import heapq

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

from constants import (
    MAX_CONNECTION_TIME,
    MIN_CONNECTION_TIME,
    MIN_TRAIN_TIME,
    MAX_TRAIN_TIME,
    SIMULATION_SEED,
)

from utils.models import get_model, get_device, get_model_weights, set_model_weights


class Server:
    def __init__(
        self,
        clients,
        num_clients,
        round_num,
        timeout,
        local_epochs,
        batch_size,
        testing_data,
        model_name,
    ):
        self.clients = clients
        self.number_of_clients = num_clients
        self.number_of_rounds = round_num
        self.start_time = 0
        self.timeout = timeout
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.global_model = get_model(model_name)
        self.testing_data = testing_data  # tuple (x_test, y_test)
        self.accuracy_history = []

    def setup_clients(self):
        for client in self.clients:
            client.setup_client(self.global_model)

    def aggregate_round(self, client_weights, client_sizes, round_num):
        if len(client_weights) == 0:
            print("Não houve resposta de nenhum cliente, o modelo não foi modificado")
            return
        print(f"Gerando novo modelo global. Rodada {round_num + 1}.")
        total_size = sum(client_sizes)
        weighted_weights = []
        for weight_idx in range(len(client_weights[0])):
            aggregated = torch.zeros_like(client_weights[0][weight_idx])
            for i in range(len(client_weights)):
                aggregated += client_weights[i][weight_idx] * (client_sizes[i] / total_size)
            weighted_weights.append(aggregated)
        set_model_weights(self.global_model, weighted_weights)
        loss, accuracy, time_stamp = self.evaluate()
        self.accuracy_history.append((loss, accuracy, time_stamp))

    def evaluate(self):
        device = get_device()
        self.global_model.eval()
        x, y = self.testing_data
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=self.batch_size)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = self.global_model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        now = time.time()
        return avg_loss, accuracy, now - self.start_time

    def distribute_weights(self):
        global_weights = get_model_weights(self.global_model)
        for client in self.clients:
            client.set_model_weights(global_weights)

    def _sample_round_duration(self, rng):
        connection = rng.uniform(MIN_CONNECTION_TIME, MAX_CONNECTION_TIME)
        train = rng.uniform(MIN_TRAIN_TIME,MAX_TRAIN_TIME)
        return connection + train

    def train_clients(self, round_num):
        client_weights = []
        client_sizes = []
        pq = []

        for idx,client in enumerate(self.clients):
            duration = self._sample_round_duration(self.rng)
            heapq.heappush(pq,(duration,idx))

        while pq:
            finish_time,client_idx = heapq.heappop(pq)
            client = self.clients[client_idx]

            if finish_time > self.timeout:
                late_ids = [client.client_id]
                late_ids.extend(self.clients[ci].client_id for _, ci in pq)
                for cid in late_ids:
                    print(f"Cliente {cid} excedeu o tempo limite na rodada {round_num}.")
                break

            client.train(self.local_epochs, self.batch_size,finish_time)
            client_weights.append(client.get_model_weights())
            client_sizes.append(client.get_dataset_size())

        print(
            f"Percentual de clientes na rodada {round_num + 1}: {100 * len(client_weights) / self.number_of_clients}%"
        )
        return client_weights, client_sizes

    def start_training(self):
        """Simulacao por eventos discretos: O tempo de cada cliente eh amostrado
        em cada rodada para estimar se haverá timeout ou não (sem threads, sem sleeps, sem lock). 
        ."""
        self.start_time = time.time()
        self.rng = random.Random(SIMULATION_SEED)

        for round_num in range(self.number_of_rounds):
            print(f"\nRodada {round_num + 1}")
            self.distribute_weights()
            client_weights, client_sizes = self.train_clients(round_num)
            self.aggregate_round(client_weights, client_sizes, round_num)
        print("Treinamento concluído. Novo modelo global gerado.")
        loss, accuracy, _ = self.evaluate()

        print("Treinamento federado síncrono concluído.")
        print(f"Perda final do modelo global: {loss:.4f}")
        print(f"Acurácia final do modelo global: {accuracy:.4f}")
        return self.accuracy_history
