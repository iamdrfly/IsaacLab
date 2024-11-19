import os
import subprocess
import time
import threading
import wandb
import logging

PATH_WANDB_FOLDER = "/home/lab/IsaacLab/train_wandb"
# Configura il logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def synchronized_file_check(file_path, timeout=43200, check_interval=60):
    """
    Aspetta che un file venga creato entro un timeout specificato (default: 12 ore).
    """
    elapsed = 0
    logging.info(f"In attesa del file {file_path} (Timeout: {timeout}s)...")
    while not os.path.exists(file_path):
        if elapsed > timeout:
            raise TimeoutError(f"Timeout superato: il file {file_path} non è stato generato.")
        time.sleep(check_interval)
        elapsed += check_interval
        logging.info(f"Attesa del file {file_path}... {elapsed}s trascorsi.")
    logging.info(f"File {file_path} trovato dopo {elapsed}s.")


def execute_isaac_training():
    """
    Esegue il processo di Isaac nel thread principale tramite un comando bash.
    """
    bash_command = (
        "./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_wandb.py "
        "--task pos-grace-rough-direct --headless"
    )
    logging.info("Avvio del training di Isaac...")
    process = subprocess.Popen(bash_command, shell=True)
    process.wait()  # Aspetta la fine del processo
    if process.returncode != 0:
        logging.error("Il processo di Isaac si è interrotto con un errore.")
        raise RuntimeError("Errore durante il training di Isaac.")
    logging.info("Training di Isaac completato.")


def monitor_training_and_log():
    """
    Monitora il completamento del training leggendo `resultisaac.txt`.
    """
    # Controlla che il training sia completo leggendo il file
    result_file = PATH_WANDB_FOLDER + "/resultisaac.txt"
    synchronized_file_check(result_file, timeout=43200)  # Timeout: 12 ore

    # Legge i risultati e logga su WandB
    logging.info(f"Lettura dei risultati da {result_file}...")
    with open(result_file, "r") as f:
        metrics = {line.split('=')[0]: float(line.split('=')[1]) for line in f.readlines()}

    # Logga i risultati su WandB
    logging.info("Logging dei risultati su WandB.")
    wandb.log(metrics)

    # Elimina il file dei risultati per evitare conflitti
    os.remove(result_file)
    logging.info(f"File {result_file} eliminato dopo l'uso.")


def train():
    """
    Funzione principale per il training, gestita da WandB agent.
    """
    # Avvio della run di WandB
    run = wandb.init()

    # Estrazione degli iperparametri
    config = run.config
    params = {
        "position_tracking_reward_scale": config.position_tracking_reward_scale,
        "heading_tracking_reward_scale": config.heading_tracking_reward_scale,
        "joint_vel_reward_scale": config.joint_vel_reward_scale,
        "joint_torque_reward_scale": config.joint_torque_reward_scale,
        "joint_vel_limit_reward_scale": config.joint_vel_limit_reward_scale,
        "joint_torque_limit_reward_scale": config.joint_torque_limit_reward_scale,
        "base_acc_reward_scale": config.base_acc_reward_scale,
        "base_lin_acc_weight": config.base_lin_acc_weight,
        "base_ang_acc_weight": config.base_ang_acc_weight,
        "feet_acc_reward_scale": config.feet_acc_reward_scale,
        # "action_rate_reward_scale":{"min": -0.01*10, "max": -0.01/10.},
        "max_feet_contact_force": config.max_feet_contact_force,
        "feet_contact_force_reward_scale": config.feet_contact_force_reward_scale,
        "wait_time": config.wait_time,
        "dont_wait_reward_scale": config.dont_wait_reward_scale,
        "move_in_direction_reward_scale": config.move_in_direction_reward_scale,
        "stand_min_dist": config.stand_min_dist,
        "stand_min_ang": config.stand_min_ang,
        "stand_at_target_reward_scale": config.stand_at_target_reward_scale,
        "undesired_contact_reward_scale": config.undesired_contact_reward_scale,
        "stumble_reward_scale": config.stumble_reward_scale,
        "feet_termination_force": config.feet_termination_force,
        "termination_reward_scale": config.termination_reward_scale

    }
    # Scrive i parametri in hyperisac.txt
    hyper_file = PATH_WANDB_FOLDER + "/hyperisac.txt"
    with open(hyper_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key}={value}\n")
    logging.info(f"Iperparametri scritti in {hyper_file}.")

    # Avvia il processo di Isaac nel thread principale
    training_thread = threading.Thread(target=execute_isaac_training)
    training_thread.start()

    # Monitora i risultati in un altro thread
    monitor_thread = threading.Thread(target=monitor_training_and_log)
    monitor_thread.start()

    # Aspetta che entrambi i thread completino
    training_thread.join()
    monitor_thread.join()

    # Elimina il file degli iperparametri per evitare conflitti
    if os.path.exists(hyper_file):
        os.remove(hyper_file)
        logging.info(f"File {hyper_file} eliminato dopo l'uso.")

    # Termina la run di WandB
    run.finish()


# Configurazione dello sweep di WandB
sweep_config = {
    "method": "bayes",
    "metric": {"name": "mean_reward", "goal": "maximize"},
    "parameters": {
        "position_tracking_reward_scale": {"min": 8., "max": 15.},
        "heading_tracking_reward_scale": {"min": 4., "max": 6.},
        "joint_vel_reward_scale": {"min": -0.001*10, "max": -0.001/10.},
        "joint_torque_reward_scale": {"min": -0.00001*10, "max": -0.00001/20.},
        "joint_vel_limit_reward_scale": {"min": -1.*10, "max": -1./10.},
        "joint_torque_limit_reward_scale": {"min": -0.2*10, "max": -0.2/10.},
        "base_acc_reward_scale":{"min": -0.001*10, "max": -0.001/10.},
        "base_lin_acc_weight":{"min": 1./10, "max": 1*10.},
        "base_ang_acc_weight":{"min": 0.02/10, "max": 0.02*10.},
        "feet_acc_reward_scale":{"min": -0.002*100, "max": -0.002/100.},
        # "action_rate_reward_scale":{"min": -0.01*10, "max": -0.01/10.},
        "max_feet_contact_force":{"min": 500., "max": 700.},
        "feet_contact_force_reward_scale":{"min": -0.00001*100, "max": -0.00001/100.},
        "wait_time":{"min": 0.05, "max": 0.4},
        "dont_wait_reward_scale":{"min": -3., "max": -0.5},
        "move_in_direction_reward_scale":{"min": 0.5, "max": 2.},
        "stand_min_dist":{"min": 0.2, "max": 0.3},
        "stand_min_ang":{"min": 0.4, "max": 0.6},
        "stand_at_target_reward_scale":{"min": -0.5*10, "max": -0.5/10.},
        "undesired_contact_reward_scale":{"min": -0.9*10, "max": -0.9},
        "stumble_reward_scale":{"min": -3., "max": -0.9},
        "feet_termination_force":{"min": 1300., "max": 1500.},
        "termination_reward_scale":{"min": -250., "max": -200.}

    },
}

# Definiamo lo sweep
sweep_id = wandb.sweep(sweep_config, project="isaaclab-optimization", count=10)

# Avvio dello sweep
wandb.agent(sweep_id, function=train)
