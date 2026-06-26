#FedCMBA_FedFPBA.py

import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, Scalar, FitIns, FitRes, EvaluateIns, EvaluateRes, MetricsAggregationFn, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters

class FedCMBA_FedFPBA(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        priority_weight: float = 0.7,  # Priority weight for the targeted client
    ) -> None:
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.priority_weight = priority_weight
        
        # Track current priority client for each round
        self.current_round = 0
        self.global_models = {}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self.current_round = server_round
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Store the default parameters as the base global model
        if parameters is not None:
            self.global_models["base"] = parameters
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        fit_configurations = []
        
        # For each client, determine if we have a specialized model to send
        for i, client in enumerate(clients):
            client_id = getattr(client, "cid", f"Client-{i+1}")
            
            # If we have a specialized model for this client, use it; otherwise use base model
            client_specific_key = f"client_{client_id}"
            client_model = self.global_models.get(client_specific_key, self.global_models.get("base"))
            
            fit_ins = FitIns(client_model, config)
            fit_configurations.append((client, fit_ins))
            
        return fit_configurations

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        return [(client, evaluate_ins) for client in clients]

    def compute_combined_metric(self, fit_res: Union[FitRes, Tuple]) -> float:
        """Calculate the harmonic mean of ADE and FDE metrics"""
        try:
            if isinstance(fit_res, tuple):
                metrics = fit_res[1].metrics if len(fit_res) > 1 else {}
            else:
                metrics = fit_res.metrics

            fde = metrics.get("FDE", 0)
            ade = metrics.get("ADE", 0)

            if isinstance(fde, str) or fde is None:
                fde = 0
            else:
                fde = float(fde)

            if isinstance(ade, str) or ade is None:
                ade = 0
            else:
                ade = float(ade)

            if fde + ade == 0:
                return 1.0  # Default value if both metrics are zero

            # Calculate harmonic mean
            return 2 * (fde * ade) / (fde + ade)

        except Exception as e:
            print(f"[ERROR] Failed to compute combined metric: {e}")
            return 1.0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        # Create multiple global models, each prioritizing a different client
        global_models = {}
        
        print(f"\n===== Round {server_round}: Combined FedCMBA+FedAPBA =====")
        
        for priority_idx, (priority_client, _) in enumerate(results):
            # Get client ID
            priority_client_id = getattr(priority_client, "cid", f"Client-{priority_idx+1}")
            
            # Extract parameters from all clients
            parameters_list = [parameters_to_ndarrays(res.parameters) for (_, res) in results]
            
            # Calculate harmonic means for all clients
            harmonic_means = []
            for _, fit_res in results:
                harmonic_mean = self.compute_combined_metric(fit_res)
                harmonic_means.append(harmonic_mean)
            
            # The inverse of harmonic means (lower error = higher weight)
            inverse_harmonic_means = []
            for hm in harmonic_means:
                if hm > 0:
                    inverse_harmonic_means.append(1.0 / hm)
                else:
                    inverse_harmonic_means.append(1.0)  # Default if harmonic mean is zero
            
            # Assign weights: priority client gets priority_weight, others get proportional weights
            weights = []
            remaining_weight = 1.0 - self.priority_weight
            total_inverse_hm = 0.0
            
            # Calculate total inverse harmonic mean for non-priority clients
            for i, (client, _) in enumerate(results):
                if i != priority_idx:  # Skip the priority client
                    total_inverse_hm += inverse_harmonic_means[i]
            
            # Assign weights to each client
            for i, (client, _) in enumerate(results):
                client_id = getattr(client, "cid", f"Client-{i+1}")
                
                if i == priority_idx:
                    # This client gets priority weighting
                    weight = self.priority_weight
                else:
                    # Other clients get proportional weighting based on inverse harmonic mean
                    if total_inverse_hm > 0:
                        weight = remaining_weight * (inverse_harmonic_means[i] / total_inverse_hm)
                    else:
                        # Equal distribution if all inverse harmonic means are zero
                        non_priority_count = len(results) - 1
                        weight = remaining_weight / non_priority_count if non_priority_count > 0 else 0
                
                weights.append(weight)
            
            # Log the weights and metrics for this model
            print(f"\nGlobal Model for priority client {priority_client_id}:")
            for i, ((client, fit_res), weight, hm) in enumerate(zip(results, weights, harmonic_means)):
                client_id = getattr(client, "cid", f"Client-{i+1}")
                metrics = fit_res.metrics
                fde = metrics.get("FDE", 0)
                ade = metrics.get("ADE", 0)
                
                print(f"  Client {client_id}:")
                print(f"    FDE: {fde}, ADE: {ade}")
                print(f"    Harmonic Mean: {hm:.4f}")
                print(f"    Weight: {weight:.4f} ({weight*100:.1f}%)")
                
                if i == priority_idx:
                    print(f"    [PRIORITY CLIENT]")
            
            # Perform weighted aggregation of parameters
            aggregated_params = []
            for param_idx in range(len(parameters_list[0])):
                # Extract this parameter across all clients
                param_slice = [params[param_idx] for params in parameters_list]
                
                # Weighted average
                weighted_param = np.average(param_slice, axis=0, weights=weights)
                aggregated_params.append(weighted_param)
            
            # Convert to Flower Parameters and store
            client_specific_model = ndarrays_to_parameters(aggregated_params)
            global_models[f"client_{priority_client_id}"] = client_specific_model
            
            # The first model will be our base return value
            if priority_idx == 0:
                base_global_model = client_specific_model
        
        # Store all models for the next round
        self.global_models = global_models
        
        print("=================================================================\n")
        
        # Return the first model as the "official" aggregate, but we'll use the specialized models in configure_fit
        return base_global_model, {"round": server_round}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        total_loss = 0
        num_clients = 0
        
        for client, eval_res in results:
            if eval_res:
                total_loss += eval_res.loss
                num_clients += 1
        
        avg_loss = total_loss / num_clients if num_clients > 0 else None

        return avg_loss, {"avg_loss": avg_loss} if avg_loss is not None else (None, {})

    def evaluate(
        self, 
        server_round: int, 
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            return None
        
        # Convert parameters to NDArrays
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        
        # Call the evaluation function
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        
        if eval_res is None:
            return None
        
        loss, metrics = eval_res
        return loss, metrics