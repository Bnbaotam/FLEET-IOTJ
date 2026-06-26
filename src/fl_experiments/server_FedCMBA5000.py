import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from datetime import timedelta

import argparse
import flwr as fl
from custom_strategy.FedCMBA import FedCMBA


class LoggingFedCMBA(FedCMBA):
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Inject the current round number into evaluation config."""
        config = {"round": server_round}
        evaluate_ins = super().configure_evaluate(server_round, parameters, client_manager)
        return [
            (client, fl.common.EvaluateIns(parameters=ins.parameters, config=config))
            for client, ins in evaluate_ins
        ]

    def configure_fit(self, server_round, parameters, client_manager):
        config = {"round": server_round}
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        return [
            (client, fl.common.FitIns(parameters=ins.parameters, config=config))
            for client, ins in fit_ins
        ]

    def aggregate_evaluate(self, server_round, results, failures):
        """Log ADE/FDE metrics for each client."""
        metrics = super().aggregate_evaluate(server_round, results, failures)
        print(f"\n[Server - FedCMB] Round {server_round} - Client Evaluation Results:")
        for client, eval_result in results:
            ade = eval_result.metrics.get("ade", None)
            fde = eval_result.metrics.get("fde", None)
            score = eval_result.metrics.get("score", None)
            print(f"Client {client.cid} - ADE: {ade:.4f}, FDE: {fde:.4f}, Score: {score:.4f}")
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Federated Server with FedCMB Strategy")
    parser.add_argument("--num_rounds", type=int, default=20, help="Number of federated training rounds")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 9090)")
    args = parser.parse_args()

    strategy = LoggingFedCMBA(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
    )

    # Start Flower server
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
