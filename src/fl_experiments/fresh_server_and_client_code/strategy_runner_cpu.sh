#!/bin/bash
set -e

########################################
# PYTHON PATH FIX (IMPORTANT)
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

cd "$PROJECT_ROOT"

echo "PYTHONPATH set to:"
echo "$PYTHONPATH"

########################################
# USER CONFIGURATION SECTION
########################################

TOTAL_CPUS=64
BASE_PORT=4000
LOG_ROOT="${SCRIPT_DIR}/logs"

RUN_TS=$(date +"%Y%m%d-%H%M%S")
LOGDIR="${LOG_ROOT}/${RUN_TS}"

mkdir -p "$LOGDIR"

echo "Run timestamp: $RUN_TS"
echo "Creating logs under: $LOGDIR"
echo "Detected TOTAL_CPUS = $TOTAL_CPUS"

########################################
# STRATEGY DEFINITIONS
########################################

STRATEGIES=(
"FedAvg"
# "FedCMBA"
# "FedRepPP"
)

SERVER_SCRIPTS=(
"$PROJECT_ROOT/src/fl_experiments/fresh_server_and_client_code/server_FedAvg.py"
# "$PROJECT_ROOT/src/fl_experiments/fresh_server_and_client_code/server_FedCMBA.py"
# "$PROJECT_ROOT/src/fl_experiments/fresh_server_and_client_code/server_FedRepPP.py"
)

CLIENT_SCRIPTS=(
"$PROJECT_ROOT/src/fl_experiments/fresh_server_and_client_code/client.py"
# "$PROJECT_ROOT/src/fl_experiments/fresh_server_and_client_code/client.py"
# "$PROJECT_ROOT/src/fl_experiments/fresh_server_and_client_code/client.py"
)

NUM_CLIENTS=(
10
# 10
# 10
)

########################################
# CONFIG FILE LISTS
########################################

CONFIGS_FedAvg=(
"$PROJECT_ROOT/config/TP/FlowChain/Broad.yml"
"$PROJECT_ROOT/config/TP/FlowChain/Market.yml"
"$PROJECT_ROOT/config/TP/FlowChain/Pine_Cam1.yml"
"$PROJECT_ROOT/config/TP/FlowChain/Pine_Cam2.yml"
"$PROJECT_ROOT/config/TP/FlowChain/Hwy27_Cam1.yml"
"$PROJECT_ROOT/config/TP/FlowChain/Hwy27_Cam2.yml"
"$PROJECT_ROOT/config/TP/FlowChain/Georgia_Cam1.yml"
"$PROJECT_ROOT/config/TP/FlowChain/Georgia_Cam2.yml"
"$PROJECT_ROOT/config/TP/FlowChain/Lindsay_Cam1.yml"
"$PROJECT_ROOT/config/TP/FlowChain/Lindsay_Cam2.yml"
)

# CONFIGS_FedCMBA=(
# "$PROJECT_ROOT/config/TP/FlowChain/Broad.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Market.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Pine_Cam1.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Pine_Cam2.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Hwy27_Cam1.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Hwy27_Cam2.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Georgia_Cam1.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Georgia_Cam2.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Lindsay_Cam1.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Lindsay_Cam2.yml"
# )

# CONFIGS_FedRepPP=(
# "$PROJECT_ROOT/config/TP/FlowChain/Broad.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Market.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Pine_Cam1.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Pine_Cam2.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Hwy27_Cam1.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Hwy27_Cam2.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Georgia_Cam1.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Georgia_Cam2.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Lindsay_Cam1.yml"
# "$PROJECT_ROOT/config/TP/FlowChain/Lindsay_Cam2.yml"
# )

########################################
# INTERNAL FUNCTION
########################################

run_strategy() {
    local strategy=$1
    local server_script=$2
    local client_script=$3
    local n_clients=$4
    local port=$5
    local strat_idx=$6

    # ✅ CORRECT ARRAY DEREFERENCE
    declare -n config_files="CONFIGS_${strategy}"

    local CPUS_PER_STRATEGY=$((n_clients + 1))
    local CPU_BASE=$((strat_idx * CPUS_PER_STRATEGY))

    local STRAT_LOG_DIR="${LOGDIR}/${strategy}"
    mkdir -p "$STRAT_LOG_DIR"

    ###################################
    # SERVER
    ###################################

    SERVER_LOG="${STRAT_LOG_DIR}/${strategy}_server_${RUN_TS}.log"

    taskset -c "$CPU_BASE" \
      env PYTHONPATH="$PYTHONPATH" \
      python3 "$server_script" --port "$port" \
      > "$SERVER_LOG" 2>&1 &

    sleep 3

    ###################################
    # CLIENTS
    ###################################

    for ((i=1; i<=n_clients; i++)); do
        config_abs="${config_files[$((i-1))]}"
        config="$(realpath --relative-to="$PROJECT_ROOT" "$config_abs")"

        cpu=$((CPU_BASE + i))
        cfg_name="$(basename "${config%.yml}")"
        CLIENT_LOG="${STRAT_LOG_DIR}/${strategy}_client_${i}_${cfg_name}_${RUN_TS}.log"

        taskset -c "$cpu" \
          env PYTHONPATH="$PYTHONPATH" \
          python3 "$client_script" \
          --config_file "$config" \
          --data_fraction 1.0 \
          --server_port "$port" \
          > "$CLIENT_LOG" 2>&1 &
    done
}

########################################
# MAIN
########################################

NUM_STRATS=${#STRATEGIES[@]}
echo "Launching $NUM_STRATS strategies..."

for ((s=0; s<NUM_STRATS; s++)); do
    run_strategy \
        "${STRATEGIES[$s]}" \
        "${SERVER_SCRIPTS[$s]}" \
        "${CLIENT_SCRIPTS[$s]}" \
        "${NUM_CLIENTS[$s]}" \
        "$((BASE_PORT + s))" \
        "$s" &
done

wait

echo "All strategies launched. Logs are under: $LOGDIR"
