import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, Scalar, FitIns, FitRes, EvaluateIns, EvaluateRes, MetricsAggregationFn, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters

class FedFPBA(fl.server.strategy.Strategy):
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
        priority_client_fraction: float = 0.4,  # Priority weight for targeted client
        base_client_fraction: float = 0.2,      # Base weight for non-priority clients
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
        self.priority_client_fraction = priority_client_fraction
        self.base_client_fraction = base_client_fraction
        
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
        metrics_dict = {}
        
        print(f"\n===== Round {server_round}: Adaptive Priority-Based Aggregation =====")
        
        for priority_idx, (priority_client, _) in enumerate(results):
            # Get client ID
            priority_client_id = getattr(priority_client, "cid", f"Client-{priority_idx+1}")
            
            # Extract parameters from all clients
            parameters_list = [parameters_to_ndarrays(res.parameters) for (_, res) in results]
            
            # Assign weights based on priority
            weights = []
            for i, (client, _) in enumerate(results):
                client_id = getattr(client, "cid", f"Client-{i+1}")
                
                if client_id == priority_client_id:
                    # This client gets priority weighting
                    weight = self.priority_client_fraction
                else:
                    # Other clients get base weighting
                    weight = self.base_client_fraction
                
                weights.append(weight)
                
            # Normalize weights to ensure they sum to 1
            weights = [w / sum(weights) for w in weights]
            
            # Log the weights for this model
            print(f"\nGlobal Model for priority client {priority_client_id}:")
            for i, ((client, _), weight) in enumerate(zip(results, weights)):
                client_id = getattr(client, "cid", f"Client-{i+1}")
                print(f"  Client {client_id}: {weight:.4f} ({weight*100:.0f}%)")
            
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