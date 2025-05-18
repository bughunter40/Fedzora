import torch
from loguru import logger
from typing import List, Tuple
from client.client import FederatedClient
from server.server import FederatedServer
from models.model_registry import ModelRegistry

def simulate_federated_learning(
    num_clients: int = 3,
    num_rounds: int = 10,
    local_epochs: int = 5,
    batch_size: int = 32,
    model_name: str = "simple_cnn",
    privacy_budget: float = 1.0,
    secure_aggregation: bool = True,
    encryption_enabled: bool = True
) -> None:
    """Simulate federated learning with privacy-preserving mechanisms."""
    
    # Initialize server
    server = FederatedServer(
        model_name=model_name,
        num_clients=num_clients,
        privacy_budget=privacy_budget,
        secure_aggregation=secure_aggregation,
        encryption_enabled=encryption_enabled
    )
    
    # Initialize clients with simulated data
    clients: List[FederatedClient] = []
    for i in range(num_clients):
        # Simulate non-IID data distribution
        local_data = generate_non_iid_data(client_id=i, num_samples=1000)
        client = FederatedClient(
            client_id=i,
            model_name=model_name,
            local_data=local_data,
            privacy_budget=privacy_budget,
            encryption_enabled=encryption_enabled
        )
        clients.append(client)
    
    # Training loop
    for round_idx in range(num_rounds):
        logger.info(f"\nFederated Learning Round {round_idx + 1}/{num_rounds}")
        
        # Distribute global model to clients
        global_params = server.get_model_parameters()
        for client in clients:
            client.update_local_model(global_params)
        
        # Local training on each client
        for client in clients:
            client.train_local_model(epochs=local_epochs, batch_size=batch_size)
            model_update = client.get_model_update()
            server.aggregate_updates(client.client_id, model_update)
        
        # Evaluate global model
        test_data = generate_test_data(1000)
        accuracy = server.evaluate_global_model(test_data)
        logger.info(f"Global Model Accuracy: {accuracy:.4f}")

def generate_non_iid_data(client_id: int, num_samples: int) -> torch.Tensor:
    """Generate non-IID data for federated learning simulation.
    
    This is a simplified implementation. In practice, you would load real data
    and distribute it non-uniformly across clients.
    """
    # Simulate different data distributions for each client
    mean = float(client_id)
    std = 1.0
    
    # Generate features
    features = torch.randn(num_samples, 1, 28, 28) * std + mean
    
    # Generate labels (simplified: 2 classes per client)
    labels = torch.randint(client_id * 2, (client_id + 1) * 2, (num_samples, 1))
    
    # Combine features and labels
    data = torch.cat([features.view(num_samples, -1), labels.float()], dim=1)
    return data

def generate_test_data(num_samples: int) -> torch.Tensor:
    """Generate test data for model evaluation."""
    features = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (num_samples, 1))
    return torch.cat([features.view(num_samples, -1), labels.float()], dim=1)

if __name__ == "__main__":
    logger.info("Starting Federated Learning Simulation with Privacy Preservation")
    
    # Configure simulation parameters
    params = {
        "num_clients": 3,
        "num_rounds": 10,
        "local_epochs": 5,
        "batch_size": 32,
        "model_name": "simple_cnn",
        "privacy_budget": 1.0,
        "secure_aggregation": True,
        "encryption_enabled": True
    }
    
    try:
        simulate_federated_learning(**params)
        logger.info("Federated Learning Simulation Completed Successfully")
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")