import torch
import numpy as np
from typing import List, Union
from diffprivlib.mechanisms import Gaussian
from loguru import logger

class DifferentialPrivacy:
    """Implementation of differential privacy mechanisms for data protection."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = Gaussian(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity
        )
        
        logger.info(f"Initialized differential privacy with ε={epsilon}, δ={delta}")
    
    def add_noise(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Add Gaussian noise to ensure differential privacy."""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
            is_tensor = True
        else:
            is_tensor = False
            
        noisy_data = self.mechanism.randomise(data)
        
        return torch.from_numpy(noisy_data) if is_tensor else noisy_data
    
    def privatize_data(self, data: torch.Tensor) -> torch.Tensor:
        """Apply differential privacy to training data."""
        return self.add_noise(data)

class SecureAggregation:
    """Secure aggregation protocol for federated learning."""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        logger.info(f"Initialized secure aggregation with threshold {threshold}")
    
    def aggregate(self, updates: List[torch.Tensor]) -> torch.Tensor:
        """Securely aggregate model updates with outlier detection."""
        updates_stack = torch.stack(updates)
        
        # Remove outliers based on threshold
        mean = torch.mean(updates_stack, dim=0)
        std = torch.std(updates_stack, dim=0)
        z_scores = torch.abs((updates_stack - mean) / (std + 1e-8))
        mask = z_scores <= self.threshold
        
        # Compute secure aggregation
        filtered_updates = updates_stack[mask]
        if len(filtered_updates) == 0:
            logger.warning("All updates filtered as outliers, using original updates")
            filtered_updates = updates_stack
        
        aggregated = torch.mean(filtered_updates, dim=0)
        logger.debug(f"Aggregated {len(filtered_updates)}/{len(updates)} updates")
        
        return aggregated