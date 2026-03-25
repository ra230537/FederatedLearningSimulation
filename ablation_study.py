import itertools
import os
import subprocess

# from synchronous.main import main as sync_main
# from asynchronous.main import main as async_main

def run_simulation(cmd):
    # Converte o comando para string para o print original do terminal
    cmd_str = " ".join(cmd)
    
    # Executa o comando e redireciona a saída do processo filho direto pro terminal em tempo real
    print(f"\n[INICIANDO PROCESSO] {cmd_str}")
    
    try:
        resultado = subprocess.run(cmd, check=True)
        print(f"\n[SUCESSO] Processo finalizado: {cmd_str}")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRO] Processo falhou com código {e.returncode}: {cmd_str}")


def run_ablation():
    print("== Iniciando Estudo de Ablação==\n")
    os.makedirs("output-cifar-10", exist_ok=True)

    # Parâmetros:
    NUM_CLIENTS = 40
    ROUNDS_OR_UPDATES = (
        10  # Valor reduzido para testes rápidos, aumente conforme necessário
    )
    epochs = 1
    batch_size = 32
    # TIMEOUT = 8

    # Variáveis:
    base_alpha = [0.3, 0.5, 0.8]
    decay_of_base_alpha = [0.999, 0.99, 0.95]
    tardiness_sensivity = [0.0, 0.075, 0.5]
    distributions = [True, False]  # False = IID, True = Non-IID

    combinations = list(
        itertools.product(
            base_alpha, decay_of_base_alpha, tardiness_sensivity, distributions
        )
    )

    print(
        f"Total de {len(combinations)} experimentos agendados (Assíncronos)."
    )
    
    commands = []

    for (
        base_alpha,
        decay_of_base_alpha,
        tardiness_sensivity,
        is_non_iid,
    ) in combinations:
        suffix = f"A{base_alpha}_B{decay_of_base_alpha}_C{tardiness_sensivity}_D{NUM_CLIENTS}"
        dist_str = "Non-IID" if is_non_iid else "IID"

        # ---------------------------------------------------------
        # 1. Rodar Síncrono
        # ---------------------------------------------------------
        # TODO: Se for rodar os síncronos futuramente, criar comandos e adicionar no commands[]

        # ---------------------------------------------------------
        # 2. Preparar Comando Assíncrono
        # ---------------------------------------------------------
        cmd = [
            "python", "asynchronous/main.py",
            "--num-clients", str(NUM_CLIENTS),
            "--num-updates", str(ROUNDS_OR_UPDATES),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--base-alpha", str(base_alpha),
            "--decay-of-base-alpha", str(decay_of_base_alpha),
            "--tardiness-sensivity", str(tardiness_sensivity),
            "--output-prefix", f"async_{suffix}"
        ]
        if is_non_iid:
            cmd.append("--non-iid")
        
        commands.append(cmd)

    print("\nIniciando testes usando subprocessos (Execução sequencial)")
    
    for cmd in commands:
        run_simulation(cmd)

    print("\n== Estudo de ablação finalizado com sucesso! ==")


if __name__ == "__main__":
    run_ablation()
