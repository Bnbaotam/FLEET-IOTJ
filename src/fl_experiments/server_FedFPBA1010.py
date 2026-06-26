import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import flwr as fl
import time
from datetime import datetime
from typing import List, Tuple, Dict, Union, Optional
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy

# Import the FedFPBA strategy we want to extend with logging
from custom_strategy.FedFixedPriorityBasedAggregation import FedFPBA


class LoggingFedFPBA(FedFPBA):
    """Enhanced logging wrapper for FedFPBA strategy."""
    
    def __init__(self, *args, log_dir: str = "./logs", **kwargs):
        """Initialize with all FedFPBA parameters plus logging options."""
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        self.start_time = time.time()
        self.round_start_times = {}
        self.client_metrics_history = {}
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Print initialization message
        print(f"\n{'='*60}")
        print(f"Initializing LoggingFedFPBA with enhanced logging")
        print(f"Priority client fraction: {self.priority_client_fraction}")
        print(f"Base client fraction: {self.base_client_fraction}")
        print(f"{'='*60}\n")

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
        
        print(f"\n[Server - FedFPBA] Round {server_round} - Configuring {len(fit_ins)} clients for training")
        for i, (client, ins) in enumerate(fit_ins):
            client_id = getattr(client, "cid", f"Client-{i+1}")
            
            # Check if this client has a specialized model
            model_type = "specialized" if f"client_{client_id}" in self.global_models else "base"
            print(f"  Client {client_id}: Sending {model_type} model parameters")
        
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
        
        print(f"[Server - FedFPBA] Round {server_round} - Configuring {len(evaluate_ins)} clients for evaluation")
        
        # Return evaluation instructions with updated config
        return [(client, fl.common.EvaluateIns(parameters=ins.parameters, config=config)) 
                for client, ins in evaluate_ins]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Log detailed information about the aggregation process."""
        # Calculate round duration
        round_duration = time.time() - self.round_start_times.get(server_round, self.start_time)
        
        # Log training metrics for each client
        print(f"\n[Server - FedFPBA] Round {server_round} - Client Training Results:")
        print(f"  Training round completed in {round_duration:.2f} seconds")
        print(f"  {len(results)} clients successfully completed training")
        print(f"  {len(failures)} clients failed during training")
        
        for i, (client, fit_res) in enumerate(results):
            client_id = getattr(client, "cid", f"Client-{i+1}")
            
            # Extract and log client metrics
            loss = fit_res.metrics.get("loss", None)
            accuracy = fit_res.metrics.get("accuracy", None)
            examples = fit_res.metrics.get("num_examples", None)
            
            # Store metrics history
            if client_id not in self.client_metrics_history:
                self.client_metrics_history[client_id] = []
            
            self.client_metrics_history[client_id].append({
                "round": server_round,
                "loss": loss,
                "accuracy": accuracy,
                "examples": examples
            })
            
            # print(f"  Client {client_id} - Loss: {loss:.4f if loss else 'N/A'}, "
            #       f"Accuracy: {accuracy:.4f if accuracy else 'N/A'}, "
            #       f"Examples: {examples if examples else 'N/A'}")

            print(f"  Client {client_id} - Loss: {f'{loss:.4f}' if loss is not None else 'N/A'}, "
                  f"Accuracy: {f'{accuracy:.4f}' if accuracy is not None else 'N/A'}, "
                  f"Examples: {examples if examples is not None else 'N/A'}")

        
        # Let the parent class handle the actual aggregation
        aggregated_results = super().aggregate_fit(server_round, results, failures)
        
        # Log the number of specialized models created
        print(f"  Created {len(self.global_models)} specialized models for the next round")
        
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
        
        print(f"\n[Server - FedFPBA] Round {server_round} - Client Evaluation Results:")
        print(f"  Evaluation completed in {eval_time:.2f} seconds")
        print(f"  {len(results)} clients successfully completed evaluation")
        print(f"  {len(failures)} clients failed during evaluation")
        
        # Extract and log client evaluation metrics
        for i, (client, eval_res) in enumerate(results):
            client_id = getattr(client, "cid", f"Client-{i+1}")
            
            # Extract metrics - adapt these to match what your clients return
            loss = eval_res.loss
            accuracy = eval_res.metrics.get("accuracy", None)
            ade = eval_res.metrics.get("ade", None)  # Average Displacement Error
            fde = eval_res.metrics.get("fde", None)  # Final Displacement Error
            score = eval_res.metrics.get("score", None)
            
            # metrics_str = f"  Client {client_id} - Loss: {loss:.4f if loss else 'N/A'}"
            if loss is not None:
                metrics_str = f"  Client {client_id} - Loss: {loss:.4f}"
            else:
                metrics_str = f"  Client {client_id} - Loss: N/A"

            
            if accuracy is not None:
                metrics_str += f", Accuracy: {accuracy:.4f}"
            if ade is not None:
                metrics_str += f", ADE: {ade:.4f}"
            if fde is not None:
                metrics_str += f", FDE: {fde:.4f}"
            if score is not None:
                metrics_str += f", Score: {score:.4f}"
                
            print(metrics_str)
        
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
            print(f"\n[Server - FedFPBA] Round {server_round} - Server-side Evaluation:")
            print(f"  Loss: {loss:.4f}")
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    print(f"  {metric_name}: {metric_value:.4f}")
                else:
                    print(f"  {metric_name}: {metric_value}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Federated Server with FedFPBA Strategy")
    parser.add_argument("--num_rounds", type=int, default=20, help="Number of federated training rounds")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=1111, help="Server port (default: 9090)")
    parser.add_argument("--priority_fraction", type=float, default=0.1, 
                        help="Weight for priority client (default: 0.4)")
    parser.add_argument("--base_fraction", type=float, default=0.9, 
                        help="Weight for non-priority clients (default: 0.2)")
    parser.add_argument("--min_clients", type=int, default=8, 
                        help="Minimum number of clients (default: 4)")
    parser.add_argument("--log_dir", type=str, default="./logs", 
                        help="Directory to store logs (default: ./logs)")
    args = parser.parse_args()

    # Create the enhanced logging strategy
    strategy = LoggingFedFPBA(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        priority_client_fraction=args.priority_fraction,
        base_client_fraction=args.base_fraction,
        log_dir=args.log_dir,
    )

    # Print startup message
    print(f"\n{'='*80}")
    print(f"Starting Flower server with LoggingFedFPBA strategy")
    print(f"Server address: {args.host}:{args.port}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Minimum clients: {args.min_clients}")
    print(f"Priority client fraction: {args.priority_fraction}")
    print(f"Base client fraction: {args.base_fraction}")
    print(f"{'='*80}\n")

    # Start Flower server
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()