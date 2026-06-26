# FLEET: Real-Time Edge-Deployable Personalized Federated Learning for Trajectory Prediction at Urban Intersections

Research codebase for federated trajectory prediction on multi-intersection driving data (CUIP / FLEET), built on [Flower](https://flower.dev/) and PyTorch. The repo supports several **federated aggregation strategies**, multiple prediction models (FlowChain, MID, Trajectron++), and scales from a small local smoke test (8 clients) to full 25-client server runs.

Run all commands from the **project root** unless noted otherwise.

---

## Federated Learning Strategies

Strategy implementations live in `src/custom_strategy/`. Each server script in `src/` or `src/fl_experiments/` wires a strategy into Flower and logs per-client **ADE**, **FDE**, and **inference time** each round.

| Strategy | Idea | Aggregation / behavior | Server entry point(s) | Client notes |
|----------|------|------------------------|----------------------|--------------|
| **FedAvg** | Standard federated averaging | Sample-weighted mean of all client updates | `src/server.py` (≥8 clients), `src/fl_experiments/fresh_server_and_client_code/server_FedAvg.py` (10 clients), `src/fl_experiments/server_fedavg.py` | `src/client.py` or `src/fl_experiments/fresh_server_and_client_code/client.py` |
| **FedCMB / FedCMBA** | Custom **metric-based** aggregation | Weights updates by a combined ADE/FDE metric (harmonic-style score from client metrics); better-performing clients contribute more | `src/server_fedcmba.py`, `…/server_FedCMBA.py` | Same clients as FedAvg |
| **FedRep** | **Representation personalization** | Server aggregates **encoder/backbone** only; each client keeps a local prediction head | `src/server_fedrep.py`, `src/fl_experiments/server_FedRep++.py` (25 clients) | `src/client_fedrep.py` (splits encoder vs head) |
| **FedRep++ / FedRepPP** | FedRep variant used in scaled FlowChain runs | Same `FedRepCustom` backend as FedRep | `…/server_FedRepPP.py`, `src/fl_experiments/server_FedRep++.py` | `src/fl_experiments/client.py` or `client_fedrep.py` depending on orchestrator |
| **FedProx** | Proximal regularization | Flower `FedProx` (`proximal_mu=0.25`) to limit client drift from the global model | `…/server_FedProx.py` | Standard client |
| **FedFPBA** | **Fixed priority** aggregation | Assigns higher weight to a priority client each round; can dispatch specialized global models per client | `…/server_FedFPBA.py`, `src/server_FedFixedPriorityBasedAggregation.py` | Standard client |
| **FedCMBA + FedFPBA** | **Hybrid** strategy | Combines metric-based (FedCMBA) weighting with priority-client logic (FedFPBA) | `…/server_FedCMBA_FedFPBA.py`, `src/server_FedCMBA_FedFPBA.py` | Standard client |
| **FedRepTraj** | FedRep with **alternating head/representation training** | Same representation-only server aggregation as FedRep, but uses a dedicated client that trains the head and encoder in separate local steps each round | `src/server_fedrep_traj.py`, `src/fl_experiments/server_fedrep_traj.py` | **Must** use `src/client_fedrep_traj.py` (not `client_fedrep.py` or `client.py`) |

**Naming notes:** older docs and logs may say **FedCMB**; the implementation class is `FedCMBA`. **FedRep++** and **FedRepPP** refer to the same representation-personalization approach in different scripts.

**Strategy source files:**

```
src/custom_strategy/
  FedCMBA.py                      # metric-based aggregation (FedCMB/A)
  FedRepCustom.py                 # FedRep / FedRep++ representation-only aggregation
  FedFixedPriorityBasedAggregation.py   # FedFPBA
  FedCMBA_FedFPBA.py              # hybrid metric + priority aggregation
```

**Batch comparison (10-client CUIP, timestamped logs):** `strategy_runner_no_cpu.sh` and `strategy_runner_cpu.sh` launch **FedAvg**, **FedCMBA**, and **FedRepPP** side by side under `src/fl_experiments/fresh_server_and_client_code/logs/<timestamp>/`.

---

## Project Structure

```
src/
  server.py, client.py              # Canonical Flower server/client (FedAvg; local smoke tests)
  server_fedcmba.py, server_fedrep.py # FedCMB/A and FedRep servers (8-client minimum)
  client_fedrep.py                    # FedRep client (encoder/head split)
  server_fedrep_traj.py               # FedRepTraj server
  client_fedrep_traj.py               # FedRepTraj client (alternating head/rep training)
  main_centralized.py                 # Single-process (non-FL) training & testing
  run_flowchain.py                    # 25-client FlowChain + FedRep++ orchestrator
  run_flowchain_fedrep_traj.py        # 25-client FlowChain + FedRepTraj orchestrator
  run_trajectron.py                   # 10-client Trajectron + FedAvg orchestrator
  custom_strategy/                  # FL aggregation strategies (see table above)
  fl_experiments/                   # Scaled servers/clients + shell runners
    fresh_server_and_client_code/     # Maintained multi-strategy servers & strategy_runner_*.sh
  data/
    unified_loader.py                 # Loads processed .pkl environments
    TP/processed_data/              # CUIP / FLEET scene pickles ({scene}_train.pkl, …)
  models/                             # FlowChain, MID, Trajectron model code
  mid_cmd_cuip_10_clients.txt         # Reference commands for 10-client MID runs

config/TP/
  FlowChain/25_clients/               # Per-scene YAML for 25-client CUIP FlowChain FL
  Trajectron/CUIP_10_clients/         # 10-client Trajectron configs
  MID/CUIP_10_clients/, CUIP_25_clients/

output/                               # Training outputs (gitignored, local only)
```

---

## Data

Processed trajectories are expected under:

```
src/data/TP/processed_data/
  {scene}_train.pkl
  {scene}_test.pkl
```

**These files are not included in the repository** (see `.gitignore`). Place preprocessed `.pkl` files locally before training, or run the preprocessing scripts under `src/data/TP/` to generate them.

**Current CUIP / FLEET scenes** (25 intersections) include Chattanooga cameras (`Broad_2`, `Market_1`, `Pine_1`, …), CityFlow nodes (`C001`–`C046`, …), and TUMTraf cameras (`s110_camera_basler_south1_8mm`, …). See `src/data/unified_loader.py` for the authoritative scene list.

**Legacy ETH scenes** (`hotel`, `zara1`, `zara2`, `eth`, `univ`, …) use an older layout under `src/data/TP/raw_data_old/` (also not published). Root-level configs such as `config/TP/FlowChain/hotel.yml` target that layout. **For current work, use configs whose `DATA.DATASET_NAME` matches a file in `processed_data/`**, e.g. `config/TP/FlowChain/25_clients/Broad_2.yml`.

`src/default_params.py` sets `DATA.PATH = "./src/data/"` — always run from the project root so paths resolve correctly.

---

## Prerequisites

- Python 3.12 (tested with 3.12.10 via pyenv)
- NVIDIA GPU with CUDA 11.8 (PyTorch cu118 wheels)
- Flower ≥ 1.18.0

### Environment setup

```bash
~/.pyenv/versions/3.12.10/bin/python -m venv .venv
source .venv/bin/activate
pip install -r requirements_vm1.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify core imports:

```bash
python check_imports.py   # expect 15/15
```

`requirements_vm1.txt` is the canonical pinned dependency list. `requirements.txt` is a minimal unpinned reference; `environment_vm1.yml` is an optional conda export for the same stack.

---

## How to Run

| Goal | Entry point |
|------|-------------|
| Local FL smoke test (FedAvg, 8 CUIP clients) | `src/server.py` + `src/client.py` |
| FedCMB/A or FedRep (8 clients) | `src/server_fedcmba.py` or `src/server_fedrep.py` + matching client |
| Centralized (non-FL) training | `src/main_centralized.py` or `run_training.sh` |
| 25-client FlowChain + FedRep++ | `src/run_flowchain.py` (**server recommended** — heavy) |
| 25-client FlowChain + FedRepTraj | `src/run_flowchain_fedrep_traj.py` |
| Trajectron FL (10 clients, FedAvg) | `src/run_trajectron.py` |
| Multi-strategy batch (FedAvg / FedCMBA / FedRepPP) | `src/fl_experiments/fresh_server_and_client_code/strategy_runner_no_cpu.sh` |
| MID 10-client reference commands | `src/mid_cmd_cuip_10_clients.txt` |

Set `PYTHONPATH` when running from subdirectories or orchestrators:

```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

---

### Local FL smoke test (FedAvg, verified)

`src/server.py` requires **at least 8 connected clients**. Use configs under `config/TP/FlowChain/25_clients/` whose `DATASET_NAME` matches `processed_data/`.

**Terminal 1 — server:**

```bash
source .venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
python src/server.py --num_rounds 2 --port 9090
```

**Terminal 2 — eight CUIP clients:**

```bash
source .venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

for scene in Broad_2 Market_1 Pine_1 Pine_2 Hwy27_1 Hwy27_2 Georgia_1 Georgia_2; do
  python src/client.py \
    --config_file "config/TP/FlowChain/25_clients/${scene}.yml" \
    --data_fraction 1.0 \
    --server_address 127.0.0.1:9090 &
done
wait
```

### FedCMB/A (8 clients)

Same client launch as above; start `src/server_fedcmba.py` instead of `src/server.py`.

### FedRep (8 clients)

Start `src/server_fedrep.py`, and launch `src/client_fedrep.py` with the same `25_clients` configs and `--server_address`.

### Centralized training (single GPU, one scene)

```bash
python src/main_centralized.py \
  --config_file config/TP/FlowChain/25_clients/Broad_2.yml \
  --mode train --gpu 0
```

See `run_training.sh` / `run_testing.sh` for additional model/scene examples (some still reference legacy ETH configs).

### FedRepTraj (dedicated client/server)

FedRepTraj uses the same FedRep server aggregation but a **different client** that alternates head-only and representation-only local training each round. Do not mix `client_fedrep_traj.py` with other server scripts, or vice versa.

**Terminal 1 — server:**

```bash
python src/server_fedrep_traj.py --num_rounds 2 --port 9090
```

**Terminal 2 — clients (example: 4 scenes; server minimum is 4 for this script):**

```bash
for scene in Broad_2 Market_1 Pine_1 Pine_2; do
  python src/client_fedrep_traj.py \
    --config_file "config/TP/FlowChain/25_clients/${scene}.yml" \
    --data_fraction 1.0 \
    --server_address 127.0.0.1:9090 &
done
wait
```

**25-client server orchestrator:**

```bash
python src/run_flowchain_fedrep_traj.py
```

### Scaled experiments (server)

These launch many GPU clients and are intended for a **server or workstation with sufficient GPU memory**:

```bash
# 25-client FlowChain + FedRep++
python src/run_flowchain.py

# 25-client FlowChain + FedRepTraj
python src/run_flowchain_fedrep_traj.py

# Multi-strategy batch with timestamped logs
bash src/fl_experiments/fresh_server_and_client_code/strategy_runner_no_cpu.sh
```

---

## Metrics & Evaluation

- **ADE** (Average Displacement Error) and **FDE** (Final Displacement Error) are computed on each client every round.
- **Inference time** (end-to-end prediction latency) is measured and reported alongside ADE/FDE.
- FedCMB/A uses client ADE/FDE to compute per-client aggregation weights.
- Checkpoints from FL clients are saved under `output/config/TP/.../best_model_<uuid>.pth` when a new best local ADE is observed.

---

## Local Artifacts (not committed)

These are kept on disk for experiments but are **gitignored**:

| Path | Contents |
|------|----------|
| `output/` | Model checkpoints, copied configs, metrics |
| `logs/` | Timestamped strategy-runner logs (repo root) |
| `src/fl_experiments/log/`, `src/fl_experiments/logs/` | Orchestrator and experiment logs |
| `src/fl_experiments/fresh_server_and_client_code/logs/` | Batch strategy-runner logs |
| `src/client_scale/*.log` | Scale-test client logs |
| `*.pth`, `*.pt`, `*.ckpt` | Model weights anywhere in the tree |
| `src/data/TP/processed_data*/`, `raw_data*/`, `**/*.pkl` | Processed and raw trajectory data (not published) |
| `.venv/` | Local virtual environment |

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

This is a research collaboration between the **University of Tennessee at Chattanooga (UTC)** and **DENSO Corporation**.

See [LICENSE](./LICENSE) for full details.
