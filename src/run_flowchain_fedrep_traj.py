
import subprocess
import os
import time
from pathlib import Path
import subprocess
import os
import time
from pathlib import Path
import socket       


def is_port_free(host: str, port: int, timeout: float = 2.0) -> bool:
    """True if the port is free (can bind to it)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


"""
===========================
Experiment Configuration Guide
===========================

1. Model Selection
------------------
Model choice is controlled via the client configuration file.

Update the following field in the client config:

    MODEL:
        TYPE: <model_name>

Available options:
    - FlowChain:
        TYPE: fastpredNF_CIF_separate_cond
    - MID:
        TYPE: MID
    - Trajectron++:
        TYPE: Trajectron


2. Number of Clients
--------------------
The number of participating clients is defined in the clients list.

Add or remove entries as needed.

Format:
    clients = [
        ('ClientName1', 'client1-log.txt'),
        ('ClientName2', 'client2-log.txt'),
        ...
    ]


3. Client Configuration Path
----------------------------
Ensure all client configuration files are located at:

    CLIENT_CONFIG_PATH


4. Strategy Selection
---------------------
The federated learning strategy is determined by the server script.

Modify the following variable:

    SERVER_SCRIPT = "<server_script_name>.py"

Examples:
    - FedAvg
    - FedCMBA
    - FedRep++
    - FedRepTraj

5. Run Script
-------------
python3 run_flowchain_fedrep_traj.py
"""
        
def main():
    # =========================
    # Global variables
    # =========================
    MODEL = "FlowChain"  # for logging
    STRATEGY = "fedrepTraj" # for logging

    SERVER_SCRIPT_PATH = "src/" # where server script lives
    SERVER_SCRIPT = "server_fedrep_traj.py" # server script file name

    CLIENT_SCRIPT_PATH = "src/" # where client script lives
    CLIENT_SCRIPT = "client_fedrep_traj.py"  # client script file name

    SERVER_LOG_PATH = f'src/fl_experiments/log/{MODEL}/{STRATEGY}/25_clients' # server log will be saved here
    SERVER_LOG = 'server-log-FedRepTraj-FlowChain-25-clients_state_v.txt' # server log file name

    CLIENT_CONFIG_PATH = f'config/TP/{MODEL}/25_clients' # where client config lives
    DATA_FRACTION = 1.0 

    
    SERVER_HOST = "0.0.0.0"       # server binds to all interfaces in container
    SERVER_PORT = 2003
    server_addr = f"127.0.0.1:{SERVER_PORT}"   # clients connect via localhost

    # CLIENT CONFIGS
    clients = [
        ('Broad_2', 'client1-Broad_2-FedRepTraj-FlowChain-state_v.txt'),
        ('Market_1', 'client2-Market_1-FedRepTraj-FlowChain-state_v.txt'),
        ('Pine_1', 'client3-Pine_1-FedRepTraj-FlowChain-state_v.txt'),
        ('Pine_2', 'client4-Pine_2-FedRepTraj-FlowChain-state_v.txt'),
        ('Hwy27_1', 'client5-Hwy27_1-FedRepTraj-FlowChain-state_v.txt'),
        ('Hwy27_2', 'client6-Hwy27_2-FedRepTraj-FlowChain-state_v.txt'),
        ('Georgia_1', 'client7-Georgia_1-FedRepTraj-FlowChain-state_v.txt'),
        ('Georgia_2', 'client8-Georgia_2-FedRepTraj-FlowChain-state_v.txt'),
        ('Lindsay_1', 'client9-Lindsay_1-FedRepTraj-FlowChain-state_v.txt'),
        ('Lindsay_2', 'client10-Lindsay_2-FedRepTraj-FlowChain-state_v.txt'),
        ('C001', 'client11-C001-FedRepTraj-FlowChain-state_v.txt'),
        ('C002', 'client12-C002-FedRepTraj-FlowChain-state_v.txt'),
        ('C003', 'client13-C003-FedRepTraj-FlowChain-state_v.txt'),
        ('C004', 'client14-C004-FedRepTraj-FlowChain-state_v.txt'),
        ('C005', 'client15-C005-FedRepTraj-FlowChain-state_v.txt'),
        ('C028', 'client16-C028-FedRepTraj-FlowChain-state_v.txt'),
        ('C035', 'client17-C035-FedRepTraj-FlowChain-state_v.txt'),
        ('C041', 'client18-C041-FedRepTraj-FlowChain-state_v.txt'),
        ('C042', 'client19-C042-FedRepTraj-FlowChain-state_v.txt'),
        ('C043', 'client20-C043-FedRepTraj-FlowChain-state_v.txt'),
        ('C044', 'client21-C044-FedRepTraj-FlowChain-state_v.txt'),
        ('C045', 'client22-C045-FedRepTraj-FlowChain-state_v.txt'),
        ('C046', 'client23-C046-FedRepTraj-FlowChain-state_v.txt'),
        ('s110_camera_basler_south1_8mm', 'client24-s110_camera_basler_south1_8mm-FedRepTraj-FlowChain-state_v.txt'),
        ('s110_camera_basler_south2_8mm', 'client25-s110_camera_basler_south2_8mm-FedRepTraj-FlowChain-state_v.txt'),
        # Add or discrad to match the number of clients you want to run
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
    # Kill existing server/client processes
    # =========================
    subprocess.run(['pkill', '-f', SERVER_SCRIPT], capture_output=True, text=True)
    subprocess.run(['pkill', '-f', CLIENT_SCRIPT], capture_output=True, text=True)
    # also kill by port (if tools exist)
    subprocess.run(f"pkill -f --port {SERVER_PORT}", shell=True, capture_output=True, text=True)
    time.sleep(3)

    # =========================
    # Wait until the server port is actually free
    # =========================
    print(f"Waiting for port {SERVER_PORT} to be free...")
    while not is_port_free(SERVER_HOST, SERVER_PORT, timeout=2.0):
        print(f"Port {SERVER_PORT} still in use; retrying...")
        # kill again just in case something re‑started
        subprocess.run(['pkill', '-f', SERVER_SCRIPT], capture_output=True, text=True)
        subprocess.run(f"pkill -f --port {SERVER_PORT}", shell=True, capture_output=True, text=True)
        time.sleep(2.0)
    print(f"Port {SERVER_PORT} is now free.")

    # =========================
    # Start server
    # =========================
    server_cmd = [
        'python3', os.path.join(SERVER_SCRIPT_PATH, SERVER_SCRIPT),
        '--host', SERVER_HOST,
        '--port', str(SERVER_PORT)
    ]

    server_log_full = os.path.join(SERVER_LOG_PATH, SERVER_LOG)
    print("Starting SERVER...")
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
        print(f"[{i}/{len(clients)}] Starting {config_name}...")
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(client_cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"{config_name} PID: {proc.pid}")
        time.sleep(1)

    print("\nAll processes launched!")
    print(f"Logs: {log_dir}")
    print(f"Server: {server_addr}")
    print("\nKill: pkill -f server_fedrep_traj || pkill -f client_fedrep_traj")
    # Change the string accroding to the global vaiables
    print(f"\nMonitor: tail -f src/fl_experiments/log/FlowChain/{STRATEGY}/25_clients/*.txt")

if __name__ == "__main__":
    main()
