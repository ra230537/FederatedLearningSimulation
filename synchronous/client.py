#Client.py
import time
import tensorflow as tf
import random

class Client:
    def __init__(self, dataset, client_id):
        self.local_model = None
        self.dataset = dataset
        self.client_id = client_id

    def setup_client(self, model):
        self.local_model = tf.keras.models.clone_model(model)
        self.local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, local_epochs, batch_size):
        start_training_time = time.time()
        connection_delay = random.uniform(0.1, 5)
        train_delay = random.uniform(0.1, 90)
        total_delay = connection_delay + train_delay
        time.sleep(total_delay)
        self.local_model.fit(self.dataset.batch(batch_size), epochs=local_epochs, verbose=0)
        end_training_time = time.time()
        print(f"Tempo de Execução do cliente {self.client_id}: {end_training_time-start_training_time:.2f}s")

    def get_dataset_size(self):
        size = self.dataset.cardinality().numpy()
        return size
    
    def set_model_weights(self, weights):
        self.local_model.set_weights(weights)
    
    def get_model_weights(self):
        return self.local_model.get_weights()