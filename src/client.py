import os
import argparse
import flwr as fl
import torch
import numpy as np
import uuid
from copy import deepcopy

from models.build_model import Build_Model
from data.unified_loader import unified_loader
from utils import load_config
from fl_train_utils import fl_local_train, evaluate_model


class TrajectoryClient(fl.client.NumPyClient):
    def __init__(self, cfg, data_fraction=1.0):
        # Initialize the client with configuration and data loaders
        self.cfg = cfg
        self.model = Build_Model(cfg)
        self.train_loader = unified_loader(
            cfg, rand=True, split="train", data_fraction=data_fraction
        )
        self.val_loader = unified_loader(
            cfg, rand=False, split="test", data_fraction=data_fraction
        )

        # Generate a unique ID for this client to avoid checkpoint collisions
        self.client_id = uuid.uuid4().hex
        # Track the lowest ADE observed so far for checkpointing
        self.best_ade = float("inf")
        # Ensure the output directory exists for saving checkpoints
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state_dict = {
            key: torch.from_numpy(np.asarray(param))
            for key, param in zip(state_dict.keys(), parameters)
        }
        self.model.load_state_dict(new_state_dict, strict=True)

    def get_metrics(self):
        eval_metrics = evaluate_model(
            self.cfg, self.model, self.val_loader, visualize=False
        )
        ade = eval_metrics.get("ade", 0.0)
        fde = eval_metrics.get("fde", 0.0)
        inference_time = eval_metrics.get("inference_time", 0.0)

        print(
            f"[DEBUG] Local evaluation → ADE: {ade:.4f}, FDE: {fde:.4f}, "
            f"Inference time: {inference_time:.6f}s"
        )

        return {
            "ADE": ade,
            "FDE": fde,
            "inference_time": inference_time,
        }

    def fit(self, parameters, config):
        # Load incoming global parameters
        self.set_parameters(parameters)
        # Perform one local training epoch
        loss = fl_local_train(
            self.cfg, model=self.model, train_loader=self.train_loader, num_epochs=1
        )

        # Evaluate on validation set
        metrics = self.get_metrics()
        metrics["loss"] = loss
        metrics["score"] = metrics["ADE"]  # alias for FL

        # Save checkpoint if current ADE is lower than any seen before
        current_ade = metrics["ADE"]
        if current_ade < self.best_ade:
            self.best_ade = current_ade
            # Use client_id to avoid overwriting checkpoints from other clients
            ckpt_filename = f"best_model_{self.client_id}.pth"
            ckpt_path = os.path.join(self.cfg.OUTPUT_DIR, ckpt_filename)
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "ade": current_ade,
                "round": config.get("round", None),
            }, ckpt_path)
            print(f"[Client {self.client_id}] New best ADE {current_ade:.4f}; checkpoint saved to {ckpt_path}")

        # Log metrics for this round
        round_num = config.get("round", "N/A")
        print(
            f"[Client {self.client_id}] Round {round_num} - ADE: {metrics['ADE']:.4f}, "
            f"FDE: {metrics['FDE']:.4f}, Score: {metrics['score']:.4f}, "
            f"Inference time: {metrics['inference_time']:.6f}s"
        )

        # Return updated parameters and metrics to server
        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        # Load parameters and perform evaluation
        self.set_parameters(parameters)

        round_num = config.get("round", "N/A")
        metrics = evaluate_model(self.cfg, self.model, self.val_loader, visualize=self.cfg.TEST.VISUALIZE)

        ade = metrics.get("ade", 0.0)
        fde = metrics.get("fde", 0.0)
        score = metrics.get("score", 0.0)
        inference_time = metrics.get("inference_time", 0.0)

        # Print evaluation results
        print(
            f"[Client {self.client_id}] Round {round_num} - ADE: {ade:.4f}, FDE: {fde:.4f}, "
            f"Score: {score:.4f}, Inference time: {inference_time:.6f}s"
        )

        return float(score), len(self.val_loader.dataset), metrics


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client for FlowChain")
    parser.add_argument("--config_file", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize predictions")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device id")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of training data to use")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train/test/tune (default: train)")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:9090", help="Flower server address (host:port)")
    args = parser.parse_args()

    cfg = load_config(args)
    cfg.defrost()
    cfg.TEST.VISUALIZE = args.visualize
    cfg.DATA.NUM_WORKERS = 0
    cfg.freeze()

    client = TrajectoryClient(cfg, data_fraction=args.data_fraction)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
