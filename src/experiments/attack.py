import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class FederatedAttacker:
    """Implements various attack strategies for federated learning systems."""
    
    def __init__(self, attack_type: str = 'model_poisoning', attack_params: Dict[str, Any] = None):
        self.attack_type = attack_type
        self.attack_params = attack_params or {}
        self.attack_methods = {
            'model_poisoning': self._model_poisoning_attack,
            'membership_inference': self._membership_inference_attack,
            'gradient_manipulation': self._gradient_manipulation_attack,
            'backdoor': self._backdoor_attack
        }
    
    def generate_attack(self, client_parameters: Dict[str, np.ndarray],
                       training_data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Generate attacked parameters based on the specified attack strategy."""
        if self.attack_type not in self.attack_methods:
            raise ValueError(f"Unsupported attack type: {self.attack_type}")
            
        return self.attack_methods[self.attack_type](client_parameters, training_data)
    
    def _model_poisoning_attack(self, client_parameters: Dict[str, np.ndarray],
                               training_data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Implement model poisoning attack by manipulating model parameters."""
        poisoned_parameters = {}
        scale = self.attack_params.get('scale', 0.1)
        
        for param_name, param in client_parameters.items():
            # Add targeted perturbation to parameters
            noise = np.random.normal(0, scale, param.shape)
            poisoned_parameters[param_name] = param + noise
            
        return poisoned_parameters
    
    def _membership_inference_attack(self, client_parameters: Dict[str, np.ndarray],
                                    training_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Implement membership inference attack to detect training data membership."""
        if training_data is None:
            raise ValueError("Training data required for membership inference attack")
            
        confidence_scores = {}
        threshold = self.attack_params.get('threshold', 0.8)
        
        # Simplified membership inference based on parameter distribution
        for param_name, param in client_parameters.items():
            param_mean = np.mean(param)
            param_std = np.std(param)
            confidence_scores[param_name] = float(param_std > threshold * param_mean)
            
        return confidence_scores
    
    def _gradient_manipulation_attack(self, client_parameters: Dict[str, np.ndarray],
                                     training_data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Implement gradient manipulation attack to mislead model updates."""
        manipulated_parameters = {}
        manipulation_scale = self.attack_params.get('manipulation_scale', 0.2)
        
        for param_name, param in client_parameters.items():
            # Scale gradients to manipulate model updates
            gradient_scale = np.random.uniform(1 - manipulation_scale, 1 + manipulation_scale)
            manipulated_parameters[param_name] = param * gradient_scale
            
        return manipulated_parameters
    
    def _backdoor_attack(self, client_parameters: Dict[str, np.ndarray],
                         training_data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Implement backdoor attack by injecting malicious patterns."""
        backdoored_parameters = {}
        trigger_scale = self.attack_params.get('trigger_scale', 0.15)
        
        for param_name, param in client_parameters.items():
            # Inject backdoor trigger pattern
            trigger = np.random.binomial(1, trigger_scale, param.shape)
            backdoored_parameters[param_name] = param * (1 + trigger)
            
        return backdoored_parameters
    
    def evaluate_attack_success(self, original_parameters: Dict[str, np.ndarray],
                               attacked_parameters: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate the effectiveness of the attack."""
        metrics = {
            'parameter_distortion': self._calculate_parameter_distortion(original_parameters, attacked_parameters),
            'attack_magnitude': self._calculate_attack_magnitude(original_parameters, attacked_parameters),
            'stealthiness': self._estimate_attack_stealthiness(original_parameters, attacked_parameters)
        }
        return metrics
    
    def _calculate_parameter_distortion(self, original: Dict[str, np.ndarray],
                                      attacked: Dict[str, np.ndarray]) -> float:
        """Calculate the average distortion in parameters caused by the attack."""
        distortions = []
        for param_name in original.keys():
            if param_name in attacked:
                distortion = np.mean(np.abs(original[param_name] - attacked[param_name]))
                distortions.append(distortion)
        return float(np.mean(distortions)) if distortions else 0.0
    
    def _calculate_attack_magnitude(self, original: Dict[str, np.ndarray],
                                   attacked: Dict[str, np.ndarray]) -> float:
        """Calculate the magnitude of changes introduced by the attack."""
        magnitudes = []
        for param_name in original.keys():
            if param_name in attacked:
                magnitude = np.linalg.norm(attacked[param_name] - original[param_name])
                magnitudes.append(magnitude)
        return float(np.mean(magnitudes)) if magnitudes else 0.0
    
    def _estimate_attack_stealthiness(self, original: Dict[str, np.ndarray],
                                     attacked: Dict[str, np.ndarray]) -> float:
        """Estimate how detectable the attack is based on parameter distribution changes."""
        distribution_changes = []
        for param_name in original.keys():
            if param_name in attacked:
                orig_std = np.std(original[param_name])
                attack_std = np.std(attacked[param_name])
                distribution_change = np.abs(orig_std - attack_std) / (orig_std + 1e-10)
                distribution_changes.append(distribution_change)
        return 1.0 - float(np.mean(distribution_changes)) if distribution_changes else 0.0