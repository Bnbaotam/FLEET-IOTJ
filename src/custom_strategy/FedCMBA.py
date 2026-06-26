
import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, Scalar, FitIns, FitRes, EvaluateIns, EvaluateRes, MetricsAggregationFn, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters

class FedCMBA(fl.server.strategy.Strategy):
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
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        return [(client, fit_ins) for client in clients]

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












    ''' Dr. Eric's code 
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Convert initial results to NDArrays for manipulation
        parameters_list = [parameters_to_ndarrays(result.parameters) for (_, result) in results]
        
        # Compute combined weights and metrics
        total_weight = 0
        combined_metrics = []
        
        # Weights will be used for aggregation
        client_weights = []
        
        for result in results:
            combined_metric = self.compute_combined_metric(result)
            client_weights.append(combined_metric)
            total_weight += combined_metric
            combined_metrics.append(combined_metric)
        
        # Normalize client weights
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Perform weighted aggregation of parameters
        aggregated_params = []
        for param_idx in range(len(parameters_list[0])):
            # Extract this parameter across all clients
            param_slice = [params[param_idx] for params in parameters_list]
            
            # Weighted average
            weighted_param = np.average(param_slice, axis=0, weights=normalized_weights)
            aggregated_params.append(weighted_param)
        
        # Convert back to Flower Parameters
        aggregated_parameters = ndarrays_to_parameters(aggregated_params)
        
        return aggregated_parameters, {"combined_metrics": combined_metrics}
    '''

     
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Convert initial results to NDArrays for manipulation
            parameters_list = [parameters_to_ndarrays(result.parameters) for (_, result) in results]
    
            # Compute combined weights and metrics
            total_weight = 0
            combined_metrics = []
    
            # Weights will be used for aggregation
            client_weights = []
    
            print(f"\n===== Round {server_round}: Client Metrics =====")
    
            for idx, (client_proxy, result) in enumerate(results):
                # Extract client ID or use index if not available
                client_id = getattr(client_proxy, "cid", f"Client-{idx}")
        
                # Get raw metrics
                metrics = result.metrics
                fde = metrics.get("FDE", 0)
                ade = metrics.get("ADE", 0)
        
                # Compute combined metric
                combined_metric = self.compute_combined_metric(result)
        
                # Calculate weight (inverse of combined metric)
                epsilon = 1e-10
                if combined_metric > epsilon:
                    weight = 1.0 / combined_metric
                else:
                    # weight = 1.0 / epsilon
                    raise ValueError(f"{combined_metric} is to small")
            
                # Print detailed information for this client
                print(f"Client {client_id}:")
                print(f"  FDE: {fde}")
                print(f"  ADE: {ade}")
                print(f"  Combined Metric: {combined_metric:.4f}")
                print(f"  Weight (1/Combined): {weight:.4f}")
        
                client_weights.append(weight)
                total_weight += weight
                combined_metrics.append(combined_metric)
    
            # Print normalized weights
            normalized_weights = [w / total_weight for w in client_weights]
            print("\nNormalized Weights:")
            for idx, (weight, (client_proxy, _)) in enumerate(zip(normalized_weights, results)):
                client_id = getattr(client_proxy, "cid", f"Client-{idx}")
                print(f"  Client {client_id}: {weight:.4f}")
    
            print("=====================================\n")
    
            # Perform weighted aggregation of parameters
            aggregated_params = []
            for param_idx in range(len(parameters_list[0])):
                # Extract this parameter across all clients
                param_slice = [params[param_idx] for params in parameters_list]
                
                # Weighted average
                weighted_param = np.average(param_slice, axis=0, weights=normalized_weights)
                aggregated_params.append(weighted_param)
    
            # Convert back to Flower Parameters
            aggregated_parameters = ndarrays_to_parameters(aggregated_params)
            
            return aggregated_parameters, {"combined_metrics": combined_metrics}

















    def compute_combined_metric(self, fit_res: Union[FitRes, Tuple]) -> float:
        try:
            if isinstance(fit_res, tuple):
                metrics = fit_res[1].metrics if len(fit_res) > 1 else {}
            else:
                metrics = fit_res.metrics

            print(f"[DEBUG] Received metrics from client: {metrics}")

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
                print(f"[WARNING] Both FDE and ADE are zero or missing: fde={fde}, ade={ade}")
                return 1.0

            return 2 * (fde * ade) / (fde + ade)

        except Exception as e:
            print(f"[ERROR] Failed to compute combined metric: {e}")
            return 1.0


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




































''' #old 4 version
import flwr as fl
from typing import List, Tuple, Dict, Optional, Union, Callable
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, Scalar, FitIns, FitRes, EvaluateIns, EvaluateRes, MetricsAggregationFn, NDArrays, parameters_to_ndarrays

class FedCMB(fl.server.strategy.Strategy):
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
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        return [(client, fit_ins) for client in clients]

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
        total_weight = 0
        aggregated_weights = None
        combined_metric_sum = 0
        
        for client, fit_res in results:
            parameters = fit_res.parameters
            combined_metric = self.compute_combined_metric(fit_res)
            client_weight = combined_metric
            
            if aggregated_weights is None:
                aggregated_weights = parameters
            else:
                aggregated_weights = self.aggregate_weights(aggregated_weights, parameters, client_weight)
                
            total_weight += client_weight
            combined_metric_sum += combined_metric
        
        normalized_weights = self.normalize_weights(aggregated_weights, total_weight) if total_weight > 0 else None

        return normalized_weights, {"combined_metric_sum": combined_metric_sum}

    def compute_combined_metric(self, fit_res: FitRes) -> float:
        metrics = fit_res.metrics

        if not isinstance(metrics, dict):
            return 0

        fde = metrics.get("FDE")
        ade = metrics.get("ADE")

        if fde is None or ade is None:
            return 0

        if fde + ade == 0:
            return 0
        
        return 2 * (fde * ade) / (fde + ade)

    def aggregate_weights(self, aggregated_weights, parameters, client_weight) -> Parameters:
        for i in range(len(aggregated_weights)):
            aggregated_weights[i] = (
                aggregated_weights[i] * (1 - client_weight) + parameters[i] * client_weight
            )
        return aggregated_weights

    def normalize_weights(self, aggregated_weights, total_weight) -> Parameters:
        for i in range(len(aggregated_weights)):
            aggregated_weights[i] /= total_weight
        return aggregated_weights

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
    

'''    