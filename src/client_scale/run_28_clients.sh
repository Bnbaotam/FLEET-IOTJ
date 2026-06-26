#!/bin/bash

# This script launches each client on its own CPU core.
# Paste your 28 python commands inside the COMMANDS array below.

# ============================
# 1. PASTE YOUR CLIENT COMMANDS HERE
# ============================
COMMANDS=(
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client1.yml --data_fraction 1.0 > src/client_scale/client1.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client2.yml --data_fraction 1.0 > src/client_scale/client2.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client3.yml --data_fraction 1.0 > src/client_scale/client3.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client4.yml --data_fraction 1.0 > src/client_scale/client4.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client5.yml --data_fraction 1.0 > src/client_scale/client5.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client6.yml --data_fraction 1.0 > src/client_scale/client6.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client7.yml --data_fraction 1.0 > src/client_scale/client7.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client8.yml --data_fraction 1.0 > src/client_scale/client8.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client9.yml --data_fraction 1.0 > src/client_scale/client9.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client10.yml --data_fraction 1.0 > src/client_scale/client10.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client11.yml --data_fraction 1.0 > src/client_scale/client11.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client12.yml --data_fraction 1.0 > src/client_scale/client12.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client13.yml --data_fraction 1.0 > src/client_scale/client13.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client14.yml --data_fraction 1.0 > src/client_scale/client14.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client15.yml --data_fraction 1.0 > src/client_scale/client15.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client16.yml --data_fraction 1.0 > src/client_scale/client16.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client17.yml --data_fraction 1.0 > src/client_scale/client17.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client18.yml --data_fraction 1.0 > src/client_scale/client18.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client19.yml --data_fraction 1.0 > src/client_scale/client19.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client20.yml --data_fraction 1.0 > src/client_scale/client20.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client21.yml --data_fraction 1.0 > src/client_scale/client21.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client22.yml --data_fraction 1.0 > src/client_scale/client22.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client23.yml --data_fraction 1.0 > src/client_scale/client23.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client24.yml --data_fraction 1.0 > src/client_scale/client24.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client25.yml --data_fraction 1.0 > src/client_scale/client25.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client26.yml --data_fraction 1.0 > src/client_scale/client26.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client27.yml --data_fraction 1.0 > src/client_scale/client27.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client28.yml --data_fraction 1.0 > src/client_scale/client28.log 2>&1"

# Add up to 28 total lines here...
)

# ============================
# 2. LAUNCH CLIENTS
# ============================
CPU=0
TOTAL=${#COMMANDS[@]}

echo "Launching $TOTAL clients..."

for ((i=0; i<$TOTAL; i++)); do
    CMD="${COMMANDS[$i]}"
    
    echo "Launching client $((i+1)) on CPU $CPU"
    
    # Run the command pinned to CPU core
    taskset -c $CPU bash -c "$CMD" &
    
    CPU=$((CPU + 1))
done

echo "All clients launched."
