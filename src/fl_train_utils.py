import os
import time
from typing import Dict, List
import numpy as np
import torch
from yacs.config import CfgNode
from copy import deepcopy

from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics

from tqdm import tqdm

def fl_local_train(cfg: CfgNode, model: torch.nn.Module, train_loader, num_epochs: int = 1):
    """Local training loop for FL clients."""
    model.train()
    ###########################################################
    try:
        optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.SOLVER.LR) #changed this line
    except:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    ##########################################################
    loss_list = []

    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, data_dict in pbar:
            # Skip invalid batch
            if data_dict is None:
                print(f"[WARNING] Skipping batch {batch_idx} due to collate failure.")
                continue

            # Move tensors to GPU if needed
            data_dict = {
                k: data_dict[k].cuda() if isinstance(data_dict[k], torch.Tensor) else data_dict[k]
                for k in data_dict
            }

            # Forward and optimize
            optimizer.zero_grad()
            loss_info = model.update(data_dict)
            loss_tensor = loss_info["loss"]
            if isinstance(loss_tensor, torch.Tensor):
                loss_tensor.backward()
            optimizer.step()

            loss_list.append(loss_info)
            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_info.items()})

    # Average losses
    if len(loss_list) == 0:
        return 0.0
    return sum(d["loss"] for d in loss_list) / len(loss_list)


def evaluate_model(cfg: CfgNode, model: torch.nn.Module, data_loader, visualize=False):
    """Run evaluation loop (ADE/FDE) after training rounds."""
    model.eval()
    metrics = Build_Metrics(cfg)
    result_info = {}

    inference_times = []

    with torch.no_grad():
        result_list = []
        for i, data_dict in enumerate(data_loader):
            data_dict = {
                k: data_dict[k].cuda() if isinstance(data_dict[k], torch.Tensor) else data_dict[k]
                for k in data_dict
            }
            dict_list = []
            for _ in range(cfg.TEST.N_TRIAL):
                start = time.time()
                result_dict = model.predict(deepcopy(data_dict), return_prob=False)
                end = time.time()
                inference_times.append(end - start)

                dict_list.append(deepcopy(result_dict))
            dict_list = metrics.denormalize(dict_list)
            result_list.append(deepcopy(metrics(dict_list)))

        d = aggregate(result_list)
        result_info.update({k: d[k] for k in d.keys() if d[k] != 0.0})

    # Add average inference time
    if inference_times:
        avg_time = np.mean(inference_times)
        result_info["inference_time"] = avg_time
        print(f"[Client] Avg inference time per prediction: {avg_time:.6f}s")

        # Auto-detect batch size for FPS calculation
        first_key = next(iter(data_dict))
        if isinstance(data_dict[first_key], torch.Tensor):
            batch_size = data_dict[first_key].shape[0]
        else:
            batch_size = 1  # Fallback if not tensor

        fps = batch_size / avg_time if avg_time > 0 else 0.0
        result_info["fps"] = fps
        print(f"[Client] Estimated FPS: {fps:.2f} (Batch size: {batch_size})")


    model.train()
    return result_info


def aggregate(dict_list: List[Dict]) -> Dict:
    """Helper to average metrics from multiple runs."""
    if "nsample" in dict_list[0]:
        return {
            k: np.sum([d[k] for d in dict_list], axis=0) / np.sum([d["nsample"] for d in dict_list])
            for k in dict_list[0].keys()
        }
    return {k: np.mean([d[k] for d in dict_list], axis=0) for k in dict_list[0].keys()}
