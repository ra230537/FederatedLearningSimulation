#Client.py
import time

class Client:
    def __init__(self, dataset):
        self.local_model = None
        self.dataset = dataset

    def setup_client(self, model):
        self.local_model = model
        self.local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, local_epochs, batch_size):
        self.local_model.fit(self.dataset.batch(batch_size), epochs=local_epochs, verbose=0)
        return self.local_model.get_weights()

    def train_multiple(self, number_of_updates, local_epochs, batch_size, server):
        for update_round in range(number_of_updates):
            print(f"Atualização {update_round + 1} de {self}")
            self.local_model.set_weights(server.get_model_weights())
            updated_weights = self.train(local_epochs, batch_size)
            server.aggregate_update(self, updated_weights, update_round) 