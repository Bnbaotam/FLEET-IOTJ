import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import flwr as fl
import numpy as np
from custom_strategy.FedRepCustom import FedRepCustom 


class LoggingFedRep(FedRepCustom):
    def configure_fit(self, server_round, parameters, client_manager):
        """Inject the current round number into the fit configuration."""
        config = {"round": server_round}
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        return [
            (client, fl.common.FitIns(parameters=ins.parameters, config=config))
            for client, ins in fit_ins
        ]

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Inject the current round number into the evaluation configuration."""
        config = {"round": server_round}
        evaluate_ins = super().configure_evaluate(server_round, parameters, client_manager)
        return [
            (client, fl.common.EvaluateIns(parameters=ins.parameters, config=config))
            for client, ins in evaluate_ins
        ]

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results and log ADE/FDE/inference time for each client."""
        if not results:
            print(f"[Server - FedRep] Round {server_round} - No evaluation results received.")
            return None, {}

        scores = []
        print(f"\n[Server - FedRep] Round {server_round} - Client Evaluation Results:")
        for client, eval_result in results:
            metrics = eval_result.metrics or {}
            ade = metrics.get("ade", 0.0)
            fde = metrics.get("fde", 0.0)
            score = metrics.get("score", ade)  # Default to ADE if score is missing
            inference_time = metrics.get("inference_time", None)

            log_line = f"Client {client.cid} - ADE: {ade:.4f}, FDE: {fde:.4f}, Score: {score:.4f}"
            if inference_time is not None:
                log_line += f", Inference time: {inference_time:.6f}s"
            print(log_line)

            scores.append(score)

        avg_score = float(np.mean(scores))
        return avg_score, {"avg_score": avg_score}


def main():
    parser = argparse.ArgumentParser(description="Federated Server with FedRep Strategy")
    parser.add_argument("--num_rounds", type=int, default=20, help="Number of federated training rounds")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7070, help="Server port (default: 9090)")
    args = parser.parse_args()

    strategy = LoggingFedRep(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=8,
    )

    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
