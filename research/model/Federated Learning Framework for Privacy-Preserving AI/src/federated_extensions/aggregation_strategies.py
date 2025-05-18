"""Aggregation Strategies Module.

Provides various aggregation methods for federated learning model updates.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any

class BaseAggregator(ABC):
    """Base class for federated aggregation strategies."""
    
    @abstractmethod
    def aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate model updates from multiple clients.
        
        Args:
            updates: List of model updates from clients
            
        Returns:
            Aggregated model parameters
        """
        pass

class WeightedAverageAggregator(BaseAggregator):
    """Implements weighted averaging of model updates."""
    
    def aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not updates:
            raise ValueError("No updates provided for aggregation")
            
        weights = [update.get('weight', 1.0) for update in updates]
        total_weight = sum(weights)
        
        if total_weight == 0:
            raise ValueError("Total weight of updates is zero")
            
        aggregated = {}
        for param_name in updates[0]['parameters'].keys():
            weighted_sum = sum(w * u['parameters'][param_name] 
                             for w, u in zip(weights, updates))
            aggregated[param_name] = weighted_sum / total_weight
            
        return {'parameters': aggregated}

class MedianAggregator(BaseAggregator):
    """Implements median-based aggregation for robustness."""
    
    def aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not updates:
            raise ValueError("No updates provided for aggregation")
            
        aggregated = {}
        for param_name in updates[0]['parameters'].keys():
            param_updates = [u['parameters'][param_name] for u in updates]
            aggregated[param_name] = np.median(param_updates, axis=0)
            
        return {'parameters': aggregated}

class TrimmedMeanAggregator(BaseAggregator):
    """Implements trimmed mean aggregation for Byzantine-robust federated learning."""
    
    def __init__(self, trim_ratio: float = 0.1):
        """Initialize trimmed mean aggregator.
        
        Args:
            trim_ratio: Ratio of updates to trim from each end (default: 0.1)
        """
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = trim_ratio
    
    def aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not updates:
            raise ValueError("No updates provided for aggregation")
            
        n_updates = len(updates)
        n_trim = int(n_updates * self.trim_ratio)
        
        aggregated = {}
        for param_name in updates[0]['parameters'].keys():
            param_updates = [u['parameters'][param_name] for u in updates]
            sorted_updates = np.sort(param_updates, axis=0)
            trimmed = sorted_updates[n_trim:n_updates-n_trim]
            aggregated[param_name] = np.mean(trimmed, axis=0)
            
        return {'parameters': aggregated}