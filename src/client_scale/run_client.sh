#!/bin/bash

CONFIG=$1
LOG=$2

taskset -c $3 python3 src/fl_experiments/client4040.py \
    --config_file "$CONFIG" \
    --data_fraction 1.0 \
    > "$LOG" 2>&1
