import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from datetime import timedelta

import argparse
import flwr as fl


class LoggingFedAvg(fl.server.strategy.FedAvg):
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Inject the current round number into evaluation config."""
        config = {"round": server_round}
        evaluate_ins = super().configure_evaluate(server_round, parameters, client_manager)
        return [
            (client, fl.common.EvaluateIns(parameters=ins.parameters, config=config))
            for client, ins in evaluate_ins
        ]

    def configure_fit(self, server_round, parameters, client_manager):
        """Inject the current round number into fit config."""
        config = {"round": server_round}
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        return [
            (client, fl.common.FitIns(parameters=ins.parameters, config=config))
            for client, ins in fit_ins
        ]

    def aggregate_evaluate(self, server_round, results, failures):
        """Log ADE/FDE and inference time metrics for each client."""
        metrics = super().aggregate_evaluate(server_round, results, failures)

        print(f"\n[Server] Round {server_round} - Client Evaluation Results:")
        for client, eval_result in results:
            client_metrics = eval_result.metrics
            ade = client_metrics.get("ade", None)
            fde = client_metrics.get("fde", None)
            score = client_metrics.get("score", None)
            inference_time = client_metrics.get("inference_time", None)

            log_line = f"Client {client.cid} - ADE: {ade:.4f}, FDE: {fde:.4f}, Score: {score:.4f}"
            if inference_time is not None:
                log_line += f", Inference time: {inference_time:.6f}s"

            print(log_line)

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Server (FedAvg)")
    parser.add_argument("--num_rounds", type=int, default=20, help="Number of federated training rounds")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=4040, help="Server port (default: 9090)")
    args = parser.parse_args()

    # Initialize FedAvg with round tracking
    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
    )

    # Start Flower server
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
