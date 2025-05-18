"""Federated Extensions Module.

This module provides extensible components for enhancing federated learning capabilities:
- Custom aggregation strategies
- Advanced privacy mechanisms
- Model optimization techniques
"""

from .aggregation_strategies import *
from .privacy_mechanisms import *
from .model_optimizers import *

__all__ = [
    # Aggregation Strategies
    'WeightedAverageAggregator',
    'MedianAggregator',
    'TrimmedMeanAggregator',
    
    # Privacy Mechanisms
    'DifferentialPrivacyEngine',
    'SecureAggregation',
    'HomomorphicEncryption',
    
    # Model Optimizers
    'FederatedOptimizer',
    'AdaptiveLearningRate',
    'GradientCompression'
]