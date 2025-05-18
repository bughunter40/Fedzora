"""Model Optimizers Module.

Implements advanced optimization techniques for federated learning models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class BaseFederatedOptimizer(ABC):
    """Base class for federated optimization techniques."""
    
    @abstractmethod
    def optimize(self, model_update: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model update.
        
        Args:
            model_update: Model parameters to optimize
            
        Returns:
            Optimized model parameters
        """
        pass

class FederatedOptimizer(BaseFederatedOptimizer):
    """Implements federated optimization with momentum and adaptive learning rates."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """Initialize federated optimizer.
        
        Args:
            learning_rate: Initial learning rate (default: 0.01)
            momentum: Momentum coefficient (default: 0.9)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
        
    def optimize(self, model_update: Dict[str, Any]) -> Dict[str, Any]:
        optimized = {}
        
        for param_name, gradient in model_update.items():
            if param_name not in self.velocity:
                self.velocity[param_name] = np.zeros_like(gradient)
                
            self.velocity[param_name] = (
                self.momentum * self.velocity[param_name] +
                self.learning_rate * gradient
            )
            optimized[param_name] = self.velocity[param_name]
            
        return optimized

class AdaptiveLearningRate(BaseFederatedOptimizer):
    """Implements adaptive learning rate optimization."""
    
    def __init__(self, initial_lr: float = 0.01, decay_rate: float = 0.99):
        """Initialize adaptive learning rate optimizer.
        
        Args:
            initial_lr: Initial learning rate (default: 0.01)
            decay_rate: Learning rate decay rate (default: 0.99)
        """
        self.learning_rate = initial_lr
        self.decay_rate = decay_rate
        self.iteration = 0
        
    def optimize(self, model_update: Dict[str, Any]) -> Dict[str, Any]:
        self.iteration += 1
        current_lr = self.learning_rate * (self.decay_rate ** self.iteration)
        
        return {
            param_name: current_lr * gradient
            for param_name, gradient in model_update.items()
        }

class GradientCompression(BaseFederatedOptimizer):
    """Implements gradient compression for communication efficiency."""
    
    def __init__(self, compression_ratio: float = 0.1):
        """Initialize gradient compression.
        
        Args:
            compression_ratio: Ratio of gradients to keep (default: 0.1)
        """
        if not 0 < compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1]")
        self.compression_ratio = compression_ratio
        self.residuals = {}
        
    def optimize(self, model_update: Dict[str, Any]) -> Dict[str, Any]:
        compressed = {}
        
        for param_name, gradient in model_update.items():
            if param_name not in self.residuals:
                self.residuals[param_name] = np.zeros_like(gradient)
                
            # Add residuals from previous compression
            gradient = gradient + self.residuals[param_name]
            
            # Select top-k components
            k = int(gradient.size * self.compression_ratio)
            threshold = np.sort(np.abs(gradient.flatten()))[-k]
            mask = np.abs(gradient) >= threshold
            
            # Update residuals
            self.residuals[param_name] = gradient * ~mask
            
            # Compress gradient
            compressed[param_name] = gradient * mask
            
        return compressed