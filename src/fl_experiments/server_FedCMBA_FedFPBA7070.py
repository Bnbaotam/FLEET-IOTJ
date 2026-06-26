#server_FedCMBA_FedFPBA.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from datetime import timedelta


import argparse
import flwr as fl
import time
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Union, Optional
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy

# Import the FedCMBA_FedFPBA strategy we want to extend with logging
from custom_strategy.FedCMBA_FedFPBA import FedCMBA_FedFPBA


class LoggingFedCMBA_FedFPBA(FedCMBA_FedFPBA):
    """Enhanced logging wrapper for FedCMBA_FedFPBA strategy."""
    
    def __init__(
        self, 
        *args, 
        log_dir: str = "./logs",
        log_to_file: bool = True,
        visualize_weights: bool = True,
        **kwargs
    ):
        """Initialize with all FedCMBA_FedFPBA parameters plus logging options."""
        super().__init__(*args, **kwargs)
        
        # Logging setup
        self.log_dir = log_dir
        self.log_to_file = log_to_file
        self.visualize_weights = visualize_weights
        self.start_time = time.time()
        self.round_start_times = {}
        self.client_metrics_history = {}
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory if it doesn't exist and logging is enabled
        if self.log_to_file:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file_path = os.path.join(self.log_dir, f"fed_cmba_fpba_{self.timestamp}.log")
            self.metrics_file_path = os.path.join(self.log_dir, f"metrics_{self.timestamp}.json")
            
            # Initialize metrics dictionary
            self.all_metrics = {
                "rounds": {},
                "clients": {},
                "aggregation_weights": {},
                "harmonic_means": {}
            }
        
        # Print initialization message
        print(f"\n{'='*70}")
        print(f"Initializing LoggingFedCMBA_FedFPBA with enhanced logging")
        print(f"Priority weight: {self.priority_weight}")
        print(f"{'='*70}\n")
        
        self.log_message(f"Federated Learning session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"Strategy: FedCMBA_FedFPBA with priority weight: {self.priority_weight}")

    def log_message(self, message: str):
        """Write message to console and log file if enabled."""
        print(message)
        
        if self.log_to_file:
            with open(self.log_file_path, "a") as f:
                f.write(f"{message}\n")

    def save_metrics(self):
        """Save metrics to JSON file if logging is enabled."""
        if self.log_to_file:
            with open(self.metrics_file_path, "w") as f:
                json.dump(self.all_metrics, f, indent=2)

    def configure_fit(self, server_round, parameters, client_manager):
        """Add round information to configuration and log start time."""
        # Record the start time of this round
        self.round_start_times[server_round] = time.time()
        
        # Add round number to config
        config = {"round": server_round}
        if self.on_fit_config_fn is not None:
            config.update(self.on_fit_config_fn(server_round))
        
        # Log which clients are being configured for training
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        
        self.log_message(f"\n[Server - FedCMBA_FedFPBA] Round {server_round} - Configuring {len(fit_ins)} clients for training")
        
        # Track specialized models
        specialized_models_count = 0
        base_models_count = 0
        
        for i, (client, ins) in enumerate(fit_ins):
            client_id = getattr(client, "cid", f"Client-{i+1}")
            
            # Check if this client has a specialized model
            model_type = "specialized" if f"client_{client_id}" in self.global_models else "base"
            if model_type == "specialized":
                specialized_models_count += 1
            else:
                base_models_count += 1
                
            self.log_message(f"  Client {client_id}: Sending {model_type} model parameters")
        
        self.log_message(f"  Summary: {specialized_models_count} specialized models, {base_models_count} base models")
        
        # Return fit instructions with updated config
        return [(client, fl.common.FitIns(parameters=ins.parameters, config=config)) 
                for client, ins in fit_ins]

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Add round information to evaluation configuration."""
        # Add round number to config
        config = {"round": server_round}
        if self.on_evaluate_config_fn is not None:
            config.update(self.on_evaluate_config_fn(server_round))
        
        # Get evaluate instructions from parent class 
        evaluate_ins = super().configure_evaluate(server_round, parameters, client_manager)
        
        self.log_message(f"[Server - FedCMBA_FedFPBA] Round {server_round} - Configuring {len(evaluate_ins)} clients for evaluation")
        
        # Return evaluation instructions with updated config
        return [(client, fl.common.EvaluateIns(parameters=ins.parameters, config=config)) 
                for client, ins in evaluate_ins]

    def format_ascii_bar(self, value, max_length=20):
        """Create an ASCII bar chart for visualizing weights."""
        bar_length = int(value * max_length)
        return "[" + "#" * bar_length + " " * (max_length - bar_length) + "]"

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Log detailed information about the aggregation process."""
        # Calculate round duration
        round_duration = time.time() - self.round_start_times.get(server_round, self.start_time)
        
        # Log training metrics from each client
        self.log_message(f"\n[Server - FedCMBA_FedFPBA] Round {server_round} - Client Training Results:")
        self.log_message(f"  Training round completed in {round_duration:.2f} seconds")
        self.log_message(f"  {len(results)} clients successfully completed training")
        self.log_message(f"  {len(failures)} clients failed during training")
        
        # Store round metrics in our history
        if self.log_to_file:
            self.all_metrics["rounds"][str(server_round)] = {
                "duration": round_duration,
                "successful_clients": len(results),
                "failed_clients": len(failures)
            }
        
        # Extract and log client metrics
        for i, (client, fit_res) in enumerate(results):
            client_id = getattr(client, "cid", f"Client-{i+1}")
            
            # Extract all metrics
            metrics = fit_res.metrics
            loss = metrics.get("loss", None)
            accuracy = metrics.get("accuracy", None)
            examples = metrics.get("num_examples", None)
            fde = metrics.get("FDE", None)
            ade = metrics.get("ADE", None)
            
            # Store metrics history for this client
            if client_id not in self.client_metrics_history:
                self.client_metrics_history[client_id] = []
            
            client_round_metrics = {
                "round": server_round,
                "loss": loss,
                "accuracy": accuracy,
                "examples": examples,
                "FDE": fde,
                "ADE": ade
            }
            
            self.client_metrics_history[client_id].append(client_round_metrics)
            
            # Save to overall metrics if logging is enabled
            if self.log_to_file:
                if client_id not in self.all_metrics["clients"]:
                    self.all_metrics["clients"][client_id] = {}
                self.all_metrics["clients"][client_id][str(server_round)] = client_round_metrics
            
            # Log metrics
            metrics_str = f"  Client {client_id}:"
            if loss is not None:
                metrics_str += f" Loss: {loss:.4f},"
            if accuracy is not None:
                metrics_str += f" Accuracy: {accuracy:.4f},"
            if fde is not None:
                metrics_str += f" FDE: {fde:.4f},"
            if ade is not None:
                metrics_str += f" ADE: {ade:.4f},"
            if examples is not None:
                metrics_str += f" Examples: {examples}"
                
            self.log_message(metrics_str.rstrip(","))
        
        # Let the parent class handle the actual aggregation
        aggregated_results = super().aggregate_fit(server_round, results, failures)
        
        # Store computed harmonic means (extract from super's logging)
        # Save metrics after aggregation
        self.save_metrics()
        
        return aggregated_results

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Log evaluation metrics from each client."""
        # Calculate round duration
        eval_time = time.time() - self.round_start_times.get(server_round, self.start_time)
        
        self.log_message(f"\n[Server - FedCMBA_FedFPBA] Round {server_round} - Client Evaluation Results:")
        self.log_message(f"  Evaluation completed in {eval_time:.2f} seconds")
        self.log_message(f"  {len(results)} clients successfully completed evaluation")
        self.log_message(f"  {len(failures)} clients failed during evaluation")
        
        # Extract and aggregate evaluation metrics
        all_metrics = {}
        
        # Extract and log client evaluation metrics
        for i, (client, eval_res) in enumerate(results):
            client_id = getattr(client, "cid", f"Client-{i+1}")
            
            # Extract metrics
            loss = eval_res.loss
            metrics = eval_res.metrics
            
            # Store metrics for this evaluation round
            client_eval_metrics = {
                "loss": loss,
                **metrics
            }
            
            # Update our metrics storage
            if self.log_to_file:
                if "evaluations" not in self.all_metrics:
                    self.all_metrics["evaluations"] = {}
                if client_id not in self.all_metrics["evaluations"]:
                    self.all_metrics["evaluations"][client_id] = {}
                self.all_metrics["evaluations"][client_id][str(server_round)] = client_eval_metrics
            
            # Generate metrics string
            # metrics_str = f"  Client {client_id} - Loss: {loss:.4f if loss else 'N/A'}"
            metrics_str = (f"  Client {client_id} - Loss: {loss:.4f}" if loss is not None else f"  Client {client_id} - Loss: N/A")

            
            # Add all available metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    metrics_str += f", {metric_name}: {float(metric_value):.4f}"
                    
                    # Aggregate metrics across clients
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(float(metric_value))
                else:
                    metrics_str += f", {metric_name}: {metric_value}"
                
            self.log_message(metrics_str)
        
        # Calculate and log aggregated metrics
        if all_metrics:
            self.log_message("\n  Aggregated Evaluation Metrics:")
            for metric_name, values in all_metrics.items():
                avg_value = sum(values) / len(values)
                self.log_message(f"    Average {metric_name}: {avg_value:.4f}")
        
        # Save metrics after evaluation
        self.save_metrics()
        
        # Let the parent class handle the actual aggregation
        return super().aggregate_evaluate(server_round, results, failures)
    
    def evaluate(
        self, 
        server_round: int, 
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Log server-side evaluation if available."""
        result = super().evaluate(server_round, parameters)
        
        if result is not None:
            loss, metrics = result
            self.log_message(f"\n[Server - FedCMBA_FedFPBA] Round {server_round} - Server-side Evaluation:")
            self.log_message(f"  Loss: {loss:.4f}")
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    self.log_message(f"  {metric_name}: {metric_value:.4f}")
                else:
                    self.log_message(f"  {metric_name}: {metric_value}")
            
            # Store server-side evaluation in metrics
            if self.log_to_file:
                if "server_evaluation" not in self.all_metrics:
                    self.all_metrics["server_evaluation"] = {}
                self.all_metrics["server_evaluation"][str(server_round)] = {
                    "loss": loss,
                    **metrics
                }
                self.save_metrics()
        
        return result
    
    def compute_combined_metric(self, fit_res: Union[FitRes, Tuple]) -> float:
        """Override to log the computed metric."""
        harmonic_mean = super().compute_combined_metric(fit_res)
        
        # Extract client ID if possible
        client_id = None
        if isinstance(fit_res, tuple) and len(fit_res) > 0:
            if hasattr(fit_res[0], "cid"):
                client_id = fit_res[0].cid
        
        # Store the harmonic mean for visualization or tracking
        if client_id and self.log_to_file:
            if client_id not in self.all_metrics["harmonic_means"]:
                self.all_metrics["harmonic_means"][client_id] = {}
            self.all_metrics["harmonic_means"][client_id][str(self.current_round)] = harmonic_mean
        
        return harmonic_mean


def main():
    parser = argparse.ArgumentParser(description="Federated Server with FedCMBA_FedFPBA Strategy")
    parser.add_argument("--num_rounds", type=int, default=20, help="Number of federated training rounds")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7070, help="Server port (default: 8080)")
    parser.add_argument("--priority_weight", type=float, default=0.7, 
                        help="Weight for priority client (default: 0.4)")
    parser.add_argument("--min_clients", type=int, default=4, 
                        help="Minimum number of clients (default: 4)")
    parser.add_argument("--log_dir", type=str, default="./logs", 
                        help="Directory to store logs (default: ./logs)")
    parser.add_argument("--log_to_file", action="store_true", help="Enable logging to file")
    parser.add_argument("--visualize_weights", action="store_true", help="Visualize weights in ASCII")
    args = parser.parse_args()

    # Create the enhanced logging strategy
    strategy = LoggingFedCMBA_FedFPBA(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        priority_weight=args.priority_weight,
        log_dir=args.log_dir,
        log_to_file=args.log_to_file,
        visualize_weights=args.visualize_weights
    )

    # Print startup message
    print(f"\n{'='*80}")
    print(f"Starting Flower server with LoggingFedCMBA_FedFPBA strategy")
    print(f"Server address: {args.host}:{args.port}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Minimum clients: {args.min_clients}")
    print(f"Priority weight: {args.priority_weight}")
    print(f"Log directory: {args.log_dir}")
    print(f"Log to file: {args.log_to_file}")
    print(f"{'='*80}\n")

    start_time = time.time()
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    formatted_time = str(timedelta(seconds=int(elapsed)))
    print(f"Total runtime: {formatted_time} (HH:MM:SS)")


if __name__ == "__main__":
    main()