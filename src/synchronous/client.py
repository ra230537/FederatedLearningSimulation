# client.py
import time
import random
import threading
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from constants import *

random.seed(42)


class Client:
    def __init__(self, dataset, client_id):
        self.local_model = None
        self.dataset = dataset  # tuple (x, y)
        self.client_id = client_id
        self._has_fresh_update = False

    def setup_client(self, model):
        from utils.models import get_device, get_model_weights, set_model_weights
        self.local_model = type(model)().to(get_device())
        set_model_weights(self.local_model, get_model_weights(model))

    def train(self, local_epochs, batch_size, stop_event):
        if stop_event.is_set():
            return

        self._has_fresh_update = False
        start_training_time = time.time()
        connection_delay = random.uniform(MIN_CONNECTION_TIME, MAX_CONNECTION_TIME)
        train_delay = random.uniform(MIN_TRAIN_TIME, MAX_TRAIN_TIME)
        time.sleep(connection_delay)

        fit_start = time.time()
        self._fit(local_epochs, batch_size)
        fit_time = time.time() - fit_start

        remaining_delay = max(0.0, train_delay - fit_time)
        time.sleep(remaining_delay)

        end_training_time = time.time()
        print(f"Tempo de Execução do cliente {self.client_id}: {end_training_time-start_training_time:.2f}s (Fit real: {fit_time:.2f}s)")
        if stop_event.is_set():
            return
        self._has_fresh_update = True

    def _fit(self, local_epochs, batch_size):
        from utils.models import get_device
        device = get_device()
        x, y = self.dataset
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.local_model.train()
        optimizer = torch.optim.Adam(self.local_model.parameters())
        criterion = nn.CrossEntropyLoss()
        for _ in range(local_epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.local_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def get_dataset_size(self):
        return len(self.dataset[0])

    def set_model_weights(self, weights):
        from utils.models import set_model_weights
        set_model_weights(self.local_model, weights)
        self._has_fresh_update = False

    def get_model_weights(self):
        from utils.models import get_model_weights
        return get_model_weights(self.local_model)

    def has_fresh_update(self):
        return self._has_fresh_update
