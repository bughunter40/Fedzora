import torch
from typing import Dict, Optional
from loguru import logger
from ..privacy import DifferentialPrivacy
from ..crypto import HomomorphicEncryption
from ..models import ModelRegistry

class FederatedClient:
    """Client node for federated learning with privacy-preserving mechanisms."""
    
    def __init__(
        self,
        client_id: int,
        model_name: str,
        local_data: torch.Tensor,
        privacy_budget: float = 1.0,
        encryption_enabled: bool = True
    ):
        self.client_id = client_id
        self.model = ModelRegistry.get_model(model_name)
        self.local_data = local_data
        
        # Privacy mechanisms
        self.diff_privacy = DifferentialPrivacy(epsilon=privacy_budget)
        self.encryption = HomomorphicEncryption() if encryption_enabled else None
        
        logger.info(f"Initialized federated client {client_id}")
    
    def update_local_model(self, global_parameters: Dict[str, torch.Tensor]) -> None:
        """Update local model with global parameters."""
        self.model.load_state_dict(global_parameters)
        logger.debug(f"Client {self.client_id} updated local model with global parameters")
    
    def train_local_model(self, epochs: int = 1, batch_size: int = 32) -> None:
        """Train local model on private data with differential privacy."""
        self.model.train()
        
        for epoch in range(epochs):
            # Apply differential privacy during training
            private_data = self.diff_privacy.privatize_data(self.local_data)
            loss = self.model.train_epoch(private_data, batch_size)
            logger.debug(f"Client {self.client_id} - Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    def get_model_update(self) -> torch.Tensor:
        """Get privacy-preserved model update for server aggregation."""
        model_update = self.model.state_dict()
        
        # Apply differential privacy to model update
        private_update = self.diff_privacy.add_noise(model_update)
        
        # Encrypt update if enabled
        if self.encryption:
            private_update = self.encryption.encrypt(private_update)
        
        return private_update
    
    def evaluate_local_model(self, test_data: Optional[torch.Tensor] = None) -> float:
        """Evaluate local model performance."""
        if test_data is None:
            test_data = self.local_data
            
        self.model.eval()
        with torch.no_grad():
            accuracy = self.model.evaluate(test_data)
        return accuracy