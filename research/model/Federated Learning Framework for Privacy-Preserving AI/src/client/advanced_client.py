import torch
from typing import Dict, List, Optional, Tuple
from loguru import logger
from ..privacy import DifferentialPrivacy
from ..crypto import AdvancedHomomorphicEncryption, SecureModelCompression, SecureKeyExchange
from ..models import ModelRegistry

class AdvancedFederatedClient:
    """Advanced client node with enhanced privacy-preserving mechanisms and secure protocols."""
    
    def __init__(
        self,
        client_id: int,
        model_name: str,
        local_data: torch.Tensor,
        privacy_budget: float = 1.0,
        key_size: int = 2048,
        compression_ratio: float = 0.1,
        sparsity: float = 0.001
    ):
        self.client_id = client_id
        self.model = ModelRegistry.get_model(model_name)
        self.local_data = local_data
        
        # Enhanced privacy and security mechanisms
        self.diff_privacy = DifferentialPrivacy(epsilon=privacy_budget)
        self.encryption = AdvancedHomomorphicEncryption(key_size=key_size)
        self.compression = SecureModelCompression(
            compression_ratio=compression_ratio,
            sparsity=sparsity
        )
        self.key_exchange = SecureKeyExchange(key_size=key_size)
        
        # Privacy accounting
        self.remaining_privacy_budget = privacy_budget
        self.noise_scale = 1.0
        
        logger.info(f"Initialized advanced federated client {client_id}")
    
    def establish_secure_connection(self, server_public_key: torch.Tensor) -> torch.Tensor:
        """Establish secure connection with server using key exchange protocol."""
        shared_secret = self.key_exchange.generate_shared_secret(server_public_key)
        session_key = self.key_exchange.derive_session_key(shared_secret)
        return self.key_exchange.public_key
    
    def update_local_model(self, encrypted_parameters: Dict[str, torch.Tensor]) -> None:
        """Update local model with encrypted global parameters."""
        decrypted_params = {}
        for name, param in encrypted_parameters.items():
            decrypted_params[name] = self.encryption.decrypt_matrix(param)
        
        self.model.load_state_dict(decrypted_params)
        logger.debug(f"Client {self.client_id} updated local model with decrypted parameters")
    
    def train_local_model(self, epochs: int = 1, batch_size: int = 32) -> None:
        """Train local model with advanced privacy preservation."""
        self.model.train()
        
        for epoch in range(epochs):
            # Adaptive noise calibration
            noise_multiplier = self._calculate_noise_multiplier()
            self.diff_privacy.update_noise_scale(noise_multiplier)
            
            # Apply enhanced differential privacy
            private_data = self.diff_privacy.privatize_data(
                self.local_data,
                noise_scale=noise_multiplier
            )
            
            loss = self.model.train_epoch(private_data, batch_size)
            self._update_privacy_accounting(epochs)
            
            logger.debug(
                f"Client {self.client_id} - Epoch {epoch + 1}/{epochs}, "
                f"Loss: {loss:.4f}, Privacy Budget: {self.remaining_privacy_budget:.4f}"
            )
    
    def get_model_update(self) -> Tuple[torch.Tensor, Dict]:
        """Get secure and compressed model update."""
        model_update = self.model.state_dict()
        
        # Apply differential privacy with adaptive noise
        private_update = self.diff_privacy.add_noise(
            model_update,
            noise_scale=self.noise_scale
        )
        
        # Compress the private update
        compressed_update, metadata = self.compression.compress(private_update)
        
        # Encrypt the compressed update
        encrypted_update = self.encryption.encrypt_matrix(compressed_update)
        
        return encrypted_update, metadata
    
    def evaluate_local_model(self, test_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Evaluate local model with detailed metrics."""
        if test_data is None:
            test_data = self.local_data
            
        self.model.eval()
        with torch.no_grad():
            metrics = self.model.evaluate_detailed(test_data)
        
        return {
            'accuracy': metrics['accuracy'],
            'privacy_budget_remaining': self.remaining_privacy_budget,
            'noise_scale': self.noise_scale
        }
    
    def _calculate_noise_multiplier(self) -> float:
        """Calculate adaptive noise multiplier based on remaining privacy budget."""
        return max(1.0, 2.0 / self.remaining_privacy_budget)
    
    def _update_privacy_accounting(self, epochs: int) -> None:
        """Update privacy budget accounting."""
        budget_per_epoch = 1.0 / (epochs * 10)  # Conservative estimate
        self.remaining_privacy_budget -= budget_per_epoch
        self.noise_scale = self._calculate_noise_multiplier()