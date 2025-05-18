import torch
from typing import Dict, List, Optional
from loguru import logger
from ..privacy import DifferentialPrivacy, SecureAggregation
from ..crypto import HomomorphicEncryption
from ..models import ModelRegistry

class FederatedServer:
    """Central server for coordinating federated learning process with privacy preservation."""
    
    def __init__(
        self,
        model_name: str,
        num_clients: int,
        privacy_budget: float = 1.0,
        secure_aggregation: bool = True,
        encryption_enabled: bool = True
    ):
        self.model = ModelRegistry.get_model(model_name)
        self.num_clients = num_clients
        self.client_updates: Dict[int, torch.Tensor] = {}
        
        # Privacy mechanisms
        self.diff_privacy = DifferentialPrivacy(epsilon=privacy_budget)
        self.secure_aggregation = SecureAggregation() if secure_aggregation else None
        self.encryption = HomomorphicEncryption() if encryption_enabled else None
        
        logger.info(f"Initialized federated server with {num_clients} clients")
    
    def aggregate_updates(self, client_id: int, model_update: torch.Tensor) -> None:
        """Aggregate model updates from clients with privacy preservation."""
        if self.encryption:
            model_update = self.encryption.decrypt(model_update)
            
        if self.diff_privacy:
            model_update = self.diff_privacy.add_noise(model_update)
            
        self.client_updates[client_id] = model_update
        
        if len(self.client_updates) == self.num_clients:
            self._perform_secure_aggregation()
    
    def _perform_secure_aggregation(self) -> None:
        """Securely aggregate model updates from all clients."""
        if self.secure_aggregation:
            aggregated_update = self.secure_aggregation.aggregate(list(self.client_updates.values()))
        else:
            aggregated_update = torch.mean(torch.stack(list(self.client_updates.values())), dim=0)
        
        self.model.load_state_dict(aggregated_update)
        self.client_updates.clear()
        logger.info("Completed secure model aggregation")
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Return current model parameters."""
        return self.model.state_dict()
    
    def evaluate_global_model(self, test_data: Optional[torch.Tensor] = None) -> float:
        """Evaluate the performance of the global model."""
        if test_data is None:
            logger.warning("No test data provided for evaluation")
            return 0.0
            
        self.model.eval()
        with torch.no_grad():
            accuracy = self.model.evaluate(test_data)
        return accuracy