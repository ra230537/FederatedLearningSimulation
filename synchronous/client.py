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
        s = time.time()
        self.local_model.fit(self.dataset.batch(batch_size), epochs=local_epochs, verbose=0)
        e = time.time()
        print(f"Tempo de Execução de {self}: {e-s:.2f}")

    def get_dataset_size(self):
        size = self.dataset.cardinality().numpy()
        return size
    
    def set_model_weights(self, weights):
        self.local_model.set_weights(weights)
    
    def get_model_weights(self):
        return self.local_model.get_weights()