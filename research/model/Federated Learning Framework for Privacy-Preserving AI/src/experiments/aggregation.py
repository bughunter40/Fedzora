import numpy as np
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

class FederatedAggregator:
    """Handles aggregation of model parameters in federated learning."""
    
    def __init__(self, strategy: str = 'fedavg', privacy_budget: float = None):
        self.strategy = strategy
        self.privacy_budget = privacy_budget
        self.aggregation_methods = {
            'fedavg': self._federated_averaging,
            'weighted': self._weighted_averaging,
            'secure': self._secure_aggregation,
            'dp': self._differentially_private_aggregation
        }
    
    def aggregate_parameters(self, client_parameters: List[Dict[str, np.ndarray]], 
                           weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Aggregate parameters from multiple clients using the specified strategy."""
        if not client_parameters:
            raise ValueError("No client parameters provided for aggregation")
            
        if self.strategy not in self.aggregation_methods:
            raise ValueError(f"Unsupported aggregation strategy: {self.strategy}")
            
        return self.aggregation_methods[self.strategy](client_parameters, weights)
    
    def _federated_averaging(self, client_parameters: List[Dict[str, np.ndarray]], 
                           weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Implement FedAvg algorithm for parameter aggregation."""
        if weights is None:
            weights = [1.0 / len(client_parameters)] * len(client_parameters)
            
        if len(weights) != len(client_parameters):
            raise ValueError("Number of weights must match number of client parameters")
            
        aggregated_params = {}
        for param_name in client_parameters[0].keys():
            weighted_sum = sum(w * params[param_name] 
                              for w, params in zip(weights, client_parameters))
            aggregated_params[param_name] = weighted_sum
            
        return aggregated_params
    
    def _weighted_averaging(self, client_parameters: List[Dict[str, np.ndarray]], 
                          weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Weighted averaging based on client contribution or data size."""
        if weights is None:
            # Calculate weights based on parameter sizes
            total_sizes = [sum(p.size for p in params.values()) for params in client_parameters]
            weights = [size / sum(total_sizes) for size in total_sizes]
            
        return self._federated_averaging(client_parameters, weights)
    
    def _secure_aggregation(self, client_parameters: List[Dict[str, np.ndarray]], 
                           weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Secure aggregation with noise masking for privacy preservation."""
        # Add secure noise masking
        noise_scale = 0.01  # Adjustable noise scale
        masked_parameters = []
        
        for params in client_parameters:
            masked = {}
            for name, param in params.items():
                noise = np.random.normal(0, noise_scale, param.shape)
                masked[name] = param + noise
            masked_parameters.append(masked)
            
        return self._federated_averaging(masked_parameters, weights)
    
    def _differentially_private_aggregation(self, client_parameters: List[Dict[str, np.ndarray]], 
                                          weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Differentially private aggregation with privacy budget management."""
        if self.privacy_budget is None:
            raise ValueError("Privacy budget must be set for DP aggregation")
            
        # Implement DP noise addition based on privacy budget
        sensitivity = 1.0  # Assuming normalized parameters
        noise_scale = sensitivity / self.privacy_budget
        
        aggregated = self._federated_averaging(client_parameters, weights)
        
        # Add calibrated noise to maintain differential privacy
        for param_name, param in aggregated.items():
            dp_noise = np.random.laplace(0, noise_scale, param.shape)
            aggregated[param_name] = param + dp_noise
            
        return aggregated
    
    def evaluate_aggregation(self, client_parameters: List[Dict[str, np.ndarray]], 
                            ground_truth: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Evaluate the quality of parameter aggregation."""
        aggregated = self.aggregate_parameters(client_parameters)
        metrics = {
            'parameter_variance': self._calculate_parameter_variance(client_parameters),
            'aggregation_bias': self._estimate_aggregation_bias(aggregated, client_parameters)
        }
        
        if ground_truth is not None:
            metrics['accuracy'] = self._calculate_accuracy(aggregated, ground_truth)
            
        return metrics
    
    def _calculate_parameter_variance(self, client_parameters: List[Dict[str, np.ndarray]]) -> float:
        """Calculate variance in parameters across clients."""
        variances = []
        for param_name in client_parameters[0].keys():
            param_values = [params[param_name] for params in client_parameters]
            variances.append(np.var(param_values))
        return float(np.mean(variances))
    
    def _estimate_aggregation_bias(self, aggregated: Dict[str, np.ndarray], 
                                  client_parameters: List[Dict[str, np.ndarray]]) -> float:
        """Estimate bias introduced by the aggregation process."""
        biases = []
        for param_name in aggregated.keys():
            client_means = np.mean([params[param_name] for params in client_parameters], axis=0)
            bias = np.mean(np.abs(aggregated[param_name] - client_means))
            biases.append(bias)
        return float(np.mean(biases))
    
    def _calculate_accuracy(self, aggregated: Dict[str, np.ndarray], 
                          ground_truth: Dict[str, np.ndarray]) -> float:
        """Calculate accuracy of aggregated parameters against ground truth."""
        errors = []
        for param_name in aggregated.keys():
            if param_name in ground_truth:
                error = np.mean(np.abs(aggregated[param_name] - ground_truth[param_name]))
                errors.append(error)
        return 1.0 - float(np.mean(errors)) if errors else 0.0