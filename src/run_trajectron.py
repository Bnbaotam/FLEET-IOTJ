#!/usr/bin/env python3
import subprocess
import os
import time
from pathlib import Path

def main():
    # =========================
    # Global variables
    # =========================
    MODEL = "Trajectron"
    STRATEGY = "fedavg"

    SERVER_SCRIPT_PATH = "src/fl_experiments"
    SERVER_SCRIPT = "server_fedavg.py"

    CLIENT_SCRIPT_PATH = "src/fl_experiments"
    CLIENT_SCRIPT = "client.py"

    SERVER_LOG_PATH = f'src/fl_experiments/log/{MODEL}/{STRATEGY}/'
    SERVER_LOG = 'server-log-FedAvg-10-clients_state_v.txt'

    CLIENT_CONFIG_PATH = f'config/TP/{MODEL}/CUIP_10_clients'
    DATA_FRACTION = 1.0

    # Hardcoded server address & port
    SERVER_HOST = "0.0.0.0"       # server binds to all interfaces in container
    SERVER_PORT = 2002
    server_addr = f"127.0.0.1:{SERVER_PORT}"   # clients connect via localhost

    # CLIENT CONFIGS
    clients = [
        ('Broad', 'client1-Broad-FedAvg-state_v.txt'),
        ('Market', 'client2-Market-FedAvg-state_v.txt'),
        ('Pine_Cam1', 'client3-Pine_Cam1-FedAvg-state_v.txt'),
        ('Pine_Cam2', 'client4-Pine_Cam2-FedAvg-state_v.txt'),
        ('Hwy27_Cam1', 'client5-Hwy27_Cam1-FedAvg-state_v.txt'),
        ('Hwy27_Cam2', 'client6-Hwy27_Cam2-FedAvg-state_v.txt'),
        ('Georgia_Cam1', 'client7-Georgia_Cam1-FedAvg-state_v.txt'),
        ('Georgia_Cam2', 'client8-Georgia_Cam2-FedAvg-state_v.txt'),
        ('Lindsay_Cam1', 'client9-Lindsay_Cam1-FedAvg-state_v.txt'),
        ('Lindsay_Cam2', 'client10-Lindsay_Cam2-FedAvg-state_v.txt'),
    ]

    # =========================
    # Ensure working directory
    # =========================
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")

    # Create log directory
    log_dir = Path(SERVER_LOG_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # Kill existing processes and free the server port
    # =========================
    subprocess.run(['pkill', '-f', SERVER_SCRIPT], capture_output=True)
    subprocess.run(['pkill', '-f', CLIENT_SCRIPT], capture_output=True)
    try:
        subprocess.run(f"fuser -k {SERVER_PORT}/tcp", shell=True, capture_output=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not free port {SERVER_PORT}: {e}")
    time.sleep(2)

    # =========================
    # Start server
    # =========================
    server_cmd = [
        'python3', os.path.join(SERVER_SCRIPT_PATH, SERVER_SCRIPT),
        '--host', SERVER_HOST,
        '--port', str(SERVER_PORT)
    ]

    server_log_full = os.path.join(SERVER_LOG_PATH, SERVER_LOG)
    print("🐧 Starting SERVER...")
    with open(server_log_full, 'w') as f:
        proc = subprocess.Popen(server_cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Server PID: {proc.pid}")
    time.sleep(3)

    # =========================
    # Start clients
    # =========================
    for i, (config_name, log_name) in enumerate(clients, 1):
        client_cmd = [
            'python3', os.path.join(CLIENT_SCRIPT_PATH, CLIENT_SCRIPT),
            '--config_file', os.path.join(CLIENT_CONFIG_PATH, config_name + '.yml'),
            '--data_fraction', str(DATA_FRACTION),
            '--server_address', server_addr
        ]

        log_file = os.path.join(SERVER_LOG_PATH, log_name)
        print(f"🤖 [{i}/{len(clients)}] Starting {config_name}...")
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(client_cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"✅ {config_name} PID: {proc.pid}")
        time.sleep(1)

    print("\n🎉 All processes launched!")
    print(f"📋 Logs: {log_dir}")
    print(f"🌐 Server: {server_addr}")
    print("\n🛑 Kill: pkill -f server_FedAvg || pkill -f client")
    print("\n🔍 Monitor: tail -f src/fl_experiments/log/Trajectron/FedAvg/*.txt")

if __name__ == "__main__":
    main()
