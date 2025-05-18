import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class FederatedDefender:
    """Implements defense mechanisms against attacks in federated learning systems."""
    
    def __init__(self, defense_type: str = 'robust_aggregation', defense_params: Dict[str, Any] = None):
        self.defense_type = defense_type
        self.defense_params = defense_params or {}
        self.defense_methods = {
            'robust_aggregation': self._robust_aggregation_defense,
            'gradient_clipping': self._gradient_clipping_defense,
            'anomaly_detection': self._anomaly_detection_defense,
            'byzantine_resilient': self._byzantine_resilient_defense
        }
    
    def apply_defense(self, client_parameters: List[Dict[str, np.ndarray]],
                     weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Apply defense mechanism to protect against potential attacks."""
        if self.defense_type not in self.defense_methods:
            raise ValueError(f"Unsupported defense type: {self.defense_type}")
            
        return self.defense_methods[self.defense_type](client_parameters, weights)
    
    def _robust_aggregation_defense(self, client_parameters: List[Dict[str, np.ndarray]],
                                   weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Implement robust aggregation to mitigate the impact of malicious updates."""
        if weights is None:
            weights = [1.0 / len(client_parameters)] * len(client_parameters)
            
        robust_params = {}
        trimming_threshold = self.defense_params.get('trimming_threshold', 2.0)
        
        for param_name in client_parameters[0].keys():
            # Calculate median and MAD for robust statistics
            param_values = [params[param_name] for params in client_parameters]
            median_value = np.median(param_values, axis=0)
            mad = np.median(np.abs(param_values - median_value), axis=0)
            
            # Filter out updates that deviate too much from median
            valid_updates = []
            valid_weights = []
            for i, param in enumerate(param_values):
                z_score = np.abs(param - median_value) / (mad + 1e-10)
                if np.mean(z_score) < trimming_threshold:
                    valid_updates.append(param)
                    valid_weights.append(weights[i])
            
            # Normalize weights and compute robust average
            if valid_weights:
                weight_sum = sum(valid_weights)
                normalized_weights = [w / weight_sum for w in valid_weights]
                robust_params[param_name] = sum(w * update for w, update in zip(normalized_weights, valid_updates))
            else:
                robust_params[param_name] = median_value
            
        return robust_params
    
    def _gradient_clipping_defense(self, client_parameters: List[Dict[str, np.ndarray]],
                                  weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Implement gradient clipping to prevent gradient explosion attacks."""
        clip_threshold = self.defense_params.get('clip_threshold', 1.0)
        clipped_parameters = []
        
        for params in client_parameters:
            clipped = {}
            for name, param in params.items():
                # Compute L2 norm and clip if necessary
                param_norm = np.linalg.norm(param)
                if param_norm > clip_threshold:
                    clipped[name] = param * (clip_threshold / param_norm)
                else:
                    clipped[name] = param
            clipped_parameters.append(clipped)
            
        return self._aggregate_parameters(clipped_parameters, weights)
    
    def _anomaly_detection_defense(self, client_parameters: List[Dict[str, np.ndarray]],
                                  weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Implement anomaly detection to identify and filter out malicious updates."""
        detection_threshold = self.defense_params.get('detection_threshold', 2.5)
        filtered_parameters = []
        filtered_weights = []
        
        # Calculate statistics for anomaly detection
        param_statistics = self._calculate_update_statistics(client_parameters)
        
        for i, params in enumerate(client_parameters):
            is_anomaly = False
            for name, param in params.items():
                mean, std = param_statistics[name]
                z_score = np.abs(np.mean(param - mean)) / (std + 1e-10)
                if z_score > detection_threshold:
                    is_anomaly = True
                    break
            
            if not is_anomaly:
                filtered_parameters.append(params)
                filtered_weights.append(weights[i] if weights else 1.0)
        
        if not filtered_parameters:
            return client_parameters[0]  # Fallback to first update if all detected as anomalies
        
        return self._aggregate_parameters(filtered_parameters, filtered_weights)
    
    def _byzantine_resilient_defense(self, client_parameters: List[Dict[str, np.ndarray]],
                                    weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Implement Byzantine-resilient aggregation to handle malicious clients."""
        krum_factor = self.defense_params.get('krum_factor', 0.7)  # Fraction of clients to keep
        num_keep = max(1, int(len(client_parameters) * krum_factor))
        
        distances = self._calculate_pairwise_distances(client_parameters)
        selected_indices = self._select_closest_updates(distances, num_keep)
        
        selected_parameters = [client_parameters[i] for i in selected_indices]
        selected_weights = None if weights is None else [weights[i] for i in selected_indices]
        
        return self._aggregate_parameters(selected_parameters, selected_weights)
    
    def _aggregate_parameters(self, parameters: List[Dict[str, np.ndarray]],
                             weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Helper method to aggregate parameters with weights."""
        if weights is None:
            weights = [1.0 / len(parameters)] * len(parameters)
            
        aggregated = {}
        for param_name in parameters[0].keys():
            weighted_sum = sum(w * params[param_name] for w, params in zip(weights, parameters))
            aggregated[param_name] = weighted_sum
            
        return aggregated
    
    def _calculate_update_statistics(self, parameters: List[Dict[str, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Calculate mean and standard deviation for each parameter across clients."""
        statistics = {}
        for param_name in parameters[0].keys():
            param_values = [params[param_name] for params in parameters]
            mean = np.mean(param_values, axis=0)
            std = np.std(param_values, axis=0)
            statistics[param_name] = (mean, std)
        return statistics
    
    def _calculate_pairwise_distances(self, parameters: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """Calculate pairwise distances between client updates."""
        n_clients = len(parameters)
        distances = np.zeros((n_clients, n_clients))
        
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = 0
                for param_name in parameters[i].keys():
                    dist += np.sum((parameters[i][param_name] - parameters[j][param_name]) ** 2)
                distances[i, j] = distances[j, i] = np.sqrt(dist)
                
        return distances
    
    def _select_closest_updates(self, distances: np.ndarray, num_keep: int) -> List[int]:
        """Select indices of updates with smallest pairwise distances."""
        n_clients = len(distances)
        scores = np.sum(distances, axis=1)
        return np.argsort(scores)[:num_keep].tolist()