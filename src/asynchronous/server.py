# Server.py

import heapq
import random
import time

from constants import (
    ACCURACY_STABILITY_DELTA,
    ACCURACY_STABILITY_PATIENCE,
    MAX_CONNECTION_TIME,
    MIN_CONNECTION_TIME,
    SIMULATION_SEED,
    STABILITY_EMA_ALPHA,
    STABILITY_EVAL_EVERY,
    STABILITY_MIN_ROUNDS,
)
from utils.models import get_model


class Server:
    def __init__(
        self,
        clients,
        num_clients,
        num_updates,
        timeout,
        local_epochs,
        batch_size,
        testing_data,
        model_name,
        base_alpha,
        decay_of_base_alpha,
        tardiness_sensivity,
        stop_on_stability=False,
        target_accuracy=None,
        stability_delta=ACCURACY_STABILITY_DELTA,
        stability_patience=ACCURACY_STABILITY_PATIENCE,
        stability_ema_alpha=STABILITY_EMA_ALPHA,
        stability_eval_every=STABILITY_EVAL_EVERY,
        stability_min_rounds=STABILITY_MIN_ROUNDS,
    ):
        self.clients = clients
        self.number_of_clients = num_clients
        self.number_of_updates = num_updates
        self.timeout = timeout
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.global_model = get_model(model_name)
        self.testing_data = testing_data
        self.accuracy_history = []
        self.start_time = 0
        self.version = 0
        self.base_alpha = base_alpha
        self.decay_of_base_alpha = decay_of_base_alpha
        self.tardiness_sensitivity = tardiness_sensivity
        self.stop_on_stability = stop_on_stability
        self.target_accuracy = target_accuracy
        self.stability_delta = stability_delta
        self.stability_patience = stability_patience
        self.stability_ema_alpha = stability_ema_alpha
        self.stability_eval_every = stability_eval_every
        self.stability_min_rounds = stability_min_rounds

        self.global_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def setup_clients(self):
        for client in self.clients:
            client.setup_client(self.global_model)

    def update_global_weights(self, global_weights, updated_weights, client_version):
        staleness = self.version - client_version
        agg_factor = self.get_aggregation_factor(staleness)
        for i in range(len(global_weights)):
            global_weights[i] = (
                global_weights[i] * (1 - agg_factor) + agg_factor * updated_weights[i]
            )

    def get_aggregation_factor(self, staleness):
        return (
            self.base_alpha
            * (self.decay_of_base_alpha**self.version)
            * (1 / (1 + self.tardiness_sensitivity * staleness))
        )

    def _sample_round_duration(self, client, rng):
        connection = rng.uniform(MIN_CONNECTION_TIME, MAX_CONNECTION_TIME)
        train = rng.uniform(*client.train_time_range)
        return connection + train

    def start_training(self):
        """Simulacao por eventos discretos: cada update do cliente eh um
        evento agendado em tempo virtual. O heap garante processamento em
        ordem cronologica (sem threads, sem sleeps, sem lock)."""
        self.start_time = time.time()
        rng = random.Random(SIMULATION_SEED)

        # Heap entries: (virtual_finish_time, seq, client_idx, base_version, base_weights)
        # seq desempata tempos identicos e mantem ordem deterministica.
        pq = []
        seq = 0
        initial_weights = self.global_model.get_weights()
        for idx, client in enumerate(self.clients):
            duration = self._sample_round_duration(client, rng)
            heapq.heappush(pq, (duration, seq, idx, 0, initial_weights))
            seq += 1

        best_accuracy = 0.0
        smoothed_accuracy = None
        patience_counter = 0

        while pq:
            finish_time, _, client_idx, base_version, base_weights = heapq.heappop(pq)
            client = self.clients[client_idx]

            if finish_time > self.timeout:
                # Heap eh ordenado: tudo que sobra tambem excedeu o timeout.
                late_ids = [client.client_id]
                late_ids.extend(self.clients[ci].client_id for _, _, ci, _, _ in pq)
                for cid in late_ids:
                    print(f"Cliente {cid} excedeu o tempo limite virtual.")
                break

            updated_weights = client.perform_fit(
                base_weights, self.local_epochs, self.batch_size
            )

            global_weights = self.global_model.get_weights()
            self.update_global_weights(global_weights, updated_weights, base_version)
            self.global_model.set_weights(global_weights)

            loss, accuracy = self.global_model.evaluate(  # pyright: ignore[reportGeneralTypeIssues]
                self.testing_data.batch(self.batch_size), verbose=0
            )
            self.accuracy_history.append((loss, accuracy, finish_time))

            staleness = self.version - base_version
            self.version += 1
            client.completed_updates += 1

            print(
                f"[t_virtual={finish_time:7.2f}s] Cliente {client.client_id} | "
                f"{client.speed_tier_name} | base_v={base_version} | "
                f"staleness={staleness} | acc={accuracy:.4f}"
            )

            if self.target_accuracy is not None and accuracy >= self.target_accuracy:
                print(
                    f"Acurácia alvo {self.target_accuracy:.4f} atingida: {accuracy:.4f}. "
                    f"Parando treinamento."
                )
                break

            if self.stop_on_stability:
                if self.version % self.stability_eval_every == 0:
                    if smoothed_accuracy is None:
                        smoothed_accuracy = accuracy
                    else:
                        smoothed_accuracy = (
                            self.stability_ema_alpha * accuracy
                            + (1 - self.stability_ema_alpha) * smoothed_accuracy
                        )

                    if smoothed_accuracy > best_accuracy + self.stability_delta:
                        best_accuracy = smoothed_accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if (
                        self.version >= self.stability_min_rounds
                        and patience_counter >= self.stability_patience
                    ):
                        print(
                            f"Acurácia estabilizada (EMA alpha={self.stability_ema_alpha}). "
                            f"Melhor suavizada: {best_accuracy:.4f}, "
                            f"últimas {patience_counter} avaliações sem melhora "
                            f"> {self.stability_delta} (avaliando a cada {self.stability_eval_every} rounds). "
                            f"Parando treinamento."
                        )
                        break

            if client.completed_updates < self.number_of_updates:
                next_duration = self._sample_round_duration(client, rng)
                heapq.heappush(
                    pq,
                    (
                        finish_time + next_duration,
                        seq,
                        client_idx,
                        self.version,
                        self.global_model.get_weights(),
                    ),
                )
                seq += 1

        loss, accuracy = self.global_model.evaluate(  # pyright: ignore[reportGeneralTypeIssues]
            self.testing_data.batch(self.batch_size), verbose=0
        )
        wall_clock = time.time() - self.start_time
        print("Treinamento federado assíncrono (DES) concluído.")
        print(f"Perda final do modelo global: {loss:.4f}")
        print(f"Acurácia final do modelo global: {accuracy:.4f}")
        print(f"Wall-clock real: {wall_clock:.1f}s | Total de agregacoes: {self.version}")
        return self.accuracy_history
