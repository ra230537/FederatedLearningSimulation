# server.py

import threading
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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

    def train_clients(self, round_num):
        client_weights = []
        client_sizes = []
        threads = []
        stop_event = threading.Event()

        for client in self.clients:
            thread = threading.Thread(
                target=client.train, args=(self.local_epochs, self.batch_size, stop_event)
            )
            threads.append((client, thread))
            thread.start()

        if round_num == 0:
            time.sleep(1)

        round_start_time = time.time()
        while time.time() - round_start_time < self.timeout:
            if all(not t.is_alive() for _, t in threads):
                break
            time.sleep(0.001)

        stop_event.set()
        for client, thread in threads:
            thread.join()
            if client.has_fresh_update():
                client_weights.append(client.get_model_weights())
                client_sizes.append(client.get_dataset_size())
            else:
                print(f"Cliente {client.client_id} excedeu o tempo limite na rodada {round_num}.")

        print(
            f"Percentual de clientes na rodada {round_num + 1}: {100 * len(client_weights) / self.number_of_clients}%"
        )
        return client_weights, client_sizes

    def start_training(self):
        self.start_time = time.time()
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
