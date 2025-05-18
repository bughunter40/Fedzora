import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class FederatedPersonalizer:
    """Implements personalization strategies for federated learning systems."""
    
    def __init__(self, strategy: str = 'fine_tuning', personalization_params: Dict[str, Any] = None):
        self.strategy = strategy
        self.personalization_params = personalization_params or {}
        self.personalization_methods = {
            'fine_tuning': self._fine_tuning_personalization,
            'meta_learning': self._meta_learning_personalization,
            'transfer_learning': self._transfer_learning_personalization,
            'adaptive_aggregation': self._adaptive_aggregation_personalization
        }
    
    def personalize_model(self, global_parameters: Dict[str, np.ndarray],
                         local_data: Optional[Dict[str, np.ndarray]] = None,
                         client_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Apply personalization strategy to adapt global model for specific client."""
        if self.strategy not in self.personalization_methods:
            raise ValueError(f"Unsupported personalization strategy: {self.strategy}")
            
        return self.personalization_methods[self.strategy](global_parameters, local_data, client_id)
    
    def _fine_tuning_personalization(self, global_parameters: Dict[str, np.ndarray],
                                    local_data: Optional[Dict[str, np.ndarray]] = None,
                                    client_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Implement fine-tuning based personalization using local data."""
        if local_data is None:
            return global_parameters
            
        learning_rate = self.personalization_params.get('learning_rate', 0.01)
        num_epochs = self.personalization_params.get('num_epochs', 5)
        
        personalized_params = {}
        for param_name, param in global_parameters.items():
            # Simulate local fine-tuning with gradient updates
            local_gradients = local_data.get(param_name, np.zeros_like(param))
            personalized_params[param_name] = param - learning_rate * local_gradients
            
        return personalized_params
    
    def _meta_learning_personalization(self, global_parameters: Dict[str, np.ndarray],
                                      local_data: Optional[Dict[str, np.ndarray]] = None,
                                      client_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Implement meta-learning based personalization for quick adaptation."""
        meta_lr = self.personalization_params.get('meta_learning_rate', 0.1)
        adaptation_steps = self.personalization_params.get('adaptation_steps', 3)
        
        adapted_params = {}
        for param_name, param in global_parameters.items():
            # Meta-initialization with learned adaptation rate
            if local_data and param_name in local_data:
                adaptation = meta_lr * local_data[param_name]
                adapted_params[param_name] = param + adaptation
            else:
                adapted_params[param_name] = param
            
        return adapted_params
    
    def _transfer_learning_personalization(self, global_parameters: Dict[str, np.ndarray],
                                          local_data: Optional[Dict[str, np.ndarray]] = None,
                                          client_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Implement transfer learning based personalization strategy."""
        transfer_layers = self.personalization_params.get('transfer_layers', [])
        adaptation_strength = self.personalization_params.get('adaptation_strength', 0.5)
        
        personalized_params = {}
        for param_name, param in global_parameters.items():
            if param_name in transfer_layers and local_data and param_name in local_data:
                # Adapt specific layers while preserving others
                local_features = local_data[param_name]
                personalized_params[param_name] = (
                    (1 - adaptation_strength) * param +
                    adaptation_strength * local_features
                )
            else:
                personalized_params[param_name] = param
            
        return personalized_params
    
    def _adaptive_aggregation_personalization(self, global_parameters: Dict[str, np.ndarray],
                                             local_data: Optional[Dict[str, np.ndarray]] = None,
                                             client_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Implement adaptive aggregation for personalized model updates."""
        similarity_threshold = self.personalization_params.get('similarity_threshold', 0.8)
        adaptation_rate = self.personalization_params.get('adaptation_rate', 0.3)
        
        adapted_params = {}
        for param_name, param in global_parameters.items():
            if local_data and param_name in local_data:
                local_param = local_data[param_name]
                # Calculate similarity between global and local parameters
                similarity = self._calculate_parameter_similarity(param, local_param)
                
                if similarity > similarity_threshold:
                    # Stronger adaptation for similar parameters
                    adaptation = adaptation_rate * (local_param - param)
                    adapted_params[param_name] = param + adaptation
                else:
                    # Conservative adaptation for dissimilar parameters
                    adapted_params[param_name] = param
            else:
                adapted_params[param_name] = param
            
        return adapted_params
    
    def evaluate_personalization(self, original_parameters: Dict[str, np.ndarray],
                                personalized_parameters: Dict[str, np.ndarray],
                                validation_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Evaluate the effectiveness of personalization."""
        metrics = {
            'parameter_adaptation': self._calculate_adaptation_degree(original_parameters, personalized_parameters),
            'model_similarity': self._calculate_model_similarity(original_parameters, personalized_parameters)
        }
        
        if validation_data is not None:
            metrics['validation_performance'] = self._evaluate_validation_performance(
                personalized_parameters, validation_data)
            
        return metrics
    
    def _calculate_parameter_similarity(self, param1: np.ndarray, param2: np.ndarray) -> float:
        """Calculate cosine similarity between parameter vectors."""
        norm1 = np.linalg.norm(param1)
        norm2 = np.linalg.norm(param2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.sum(param1 * param2) / (norm1 * norm2))
    
    def _calculate_adaptation_degree(self, original: Dict[str, np.ndarray],
                                    personalized: Dict[str, np.ndarray]) -> float:
        """Calculate the degree of adaptation in parameters."""
        adaptations = []
        for param_name in original.keys():
            if param_name in personalized:
                adaptation = np.mean(np.abs(personalized[param_name] - original[param_name]))
                adaptations.append(adaptation)
        return float(np.mean(adaptations)) if adaptations else 0.0
    
    def _calculate_model_similarity(self, original: Dict[str, np.ndarray],
                                   personalized: Dict[str, np.ndarray]) -> float:
        """Calculate overall similarity between original and personalized models."""
        similarities = []
        for param_name in original.keys():
            if param_name in personalized:
                similarity = self._calculate_parameter_similarity(
                    original[param_name], personalized[param_name])
                similarities.append(similarity)
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _evaluate_validation_performance(self, parameters: Dict[str, np.ndarray],
                                        validation_data: Dict[str, np.ndarray]) -> float:
        """Evaluate model performance on validation data."""
        # Simplified validation metric (e.g., parameter-wise correlation)
        correlations = []
        for param_name in parameters.keys():
            if param_name in validation_data:
                correlation = np.corrcoef(parameters[param_name].flatten(),
                                         validation_data[param_name].flatten())[0, 1]
                correlations.append(correlation)
        return float(np.mean(correlations)) if correlations else 0.0