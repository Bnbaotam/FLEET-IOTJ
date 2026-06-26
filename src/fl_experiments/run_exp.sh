#!/bin/bash

echo "Starting Servers..."

# server port 6060 - ORIN 1
python3 src/fl_experiments/server_fedavgm6060.py > server-log-FedAvgM-MID-2nd.txt 2>&1  &

# server port 7070 - ORIN 2
# python3 src/fl_experiments/server_fedavgm7070.py > server-log-FedAvgM-MID-3rd.txt 2>&1 &

# server port 8080 - ORIN 3
# python3 src/fl_experiments/server_fedavgm8080.py > server-log-FedAvgM-MID-4th.txt 2>&1 &

# server port 9090 - ORIN 4 
# python3 src/fl_experiments/server_fedavgm9090.py > server-log-FedAvgM-MID-5th.txt 2>&1 &


echo "All servers launched in background..."
