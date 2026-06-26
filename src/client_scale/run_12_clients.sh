#!/bin/bash

# ============================
# 1. CLIENT COMMANDS (12 clients)
# ============================
COMMANDS=(
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client1.yml  --data_fraction 1.0 > client1-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client2.yml  --data_fraction 1.0 > client2-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client3.yml  --data_fraction 1.0 > client3-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client4.yml  --data_fraction 1.0 > client4-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client5.yml  --data_fraction 1.0 > client5-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client6.yml  --data_fraction 1.0 > client6-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client7.yml  --data_fraction 1.0 > client7-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client8.yml  --data_fraction 1.0 > client8-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client9.yml  --data_fraction 1.0 > client9-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client10.yml --data_fraction 1.0 > client10-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client11.yml --data_fraction 1.0 > client11-12Client.log 2>&1"
"python3 src/fl_experiments/client4040.py --config_file config/TP/FlowChain/client12.yml --data_fraction 1.0 > client12-12Client.log 2>&1"
)

TOTAL=${#COMMANDS[@]}

# ============================
# 2. CPU / GPU DETECTION
# ============================

# Count CPUs
CPUS=$(nproc --all)

# Count GPUs (if nvidia GPUs exist)
if command -v nvidia-smi &> /dev/null; then
    GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPUS=0
fi

echo "Detected $CPUS CPU cores"
echo "Detected $GPUS GPUs"

# ============================
# 3. LAUNCH CLIENTS
# ============================

echo "Launching $TOTAL clients..."

CPU=0
GPU=0

for ((i=0; i<$TOTAL; i++)); do
    CMD="${COMMANDS[$i]}"

    echo -n "Client $((i+1)): CPU $CPU"

    # Assign GPU if available (round-robin)
    if [ $GPUS -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$GPU
        echo -n ", GPU $GPU"
        GPU=$(( (GPU + 1) % GPUS ))
    fi
    
    echo ""

    # Launch client pinned to CPU
    taskset -c $CPU bash -c "$CMD" &

    CPU=$((CPU + 1))
done

echo "All 12 clients launched."
