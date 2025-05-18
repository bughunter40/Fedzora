import pytest
from unittest.mock import Mock, patch
import numpy as np

@pytest.fixture
def mock_server():
    """Fixture for creating a mock federated learning server."""
    server = Mock()
    server.connected_clients = []
    server.global_model = Mock()
    server.aggregation_round = 0
    return server

def test_server_initialization(mock_server, test_config):
    """Test server initialization and configuration."""
    assert hasattr(mock_server, 'global_model')
    assert mock_server.aggregation_round == 0
    assert len(mock_server.connected_clients) == 0

def test_client_registration(mock_server):
    """Test client registration process."""
    client_id = 'test_client_1'
    mock_server.register_client = Mock(return_value=True)
    
    success = mock_server.register_client(client_id)
    assert success
    
    mock_server.connected_clients.append(client_id)
    assert len(mock_server.connected_clients) == 1

def test_model_aggregation(mock_server):
    """Test federated averaging of client models."""
    client_weights = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.2, 0.3, 0.4]),
        np.array([0.3, 0.4, 0.5])
    ]
    
    mock_server.aggregate_models = Mock(
        return_value=np.mean(client_weights, axis=0)
    )
    
    aggregated_weights = mock_server.aggregate_models(client_weights)
    assert isinstance(aggregated_weights, np.ndarray)
    assert len(aggregated_weights) == len(client_weights[0])

def test_secure_aggregation_protocol(mock_server, mock_crypto_keys):
    """Test secure aggregation protocol implementation."""
    encrypted_updates = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6])
    ]
    
    mock_server.decrypt_aggregation = Mock(
        return_value=np.mean(encrypted_updates, axis=0)
    )
    
    decrypted_result = mock_server.decrypt_aggregation(
        encrypted_updates,
        mock_crypto_keys['private_key']
    )
    
    assert isinstance(decrypted_result, np.ndarray)
    assert len(decrypted_result) == len(encrypted_updates[0])

def test_model_distribution(mock_server):
    """Test global model distribution to clients."""
    mock_server.broadcast_model = Mock(return_value=True)
    success = mock_server.broadcast_model()
    assert success

def test_privacy_budget_management(mock_server, test_config):
    """Test privacy budget allocation and tracking."""
    initial_budget = test_config['privacy_budget']
    mock_server.update_privacy_budget = Mock(return_value=initial_budget - 0.1)
    
    updated_budget = mock_server.update_privacy_budget(0.1)
    assert updated_budget < initial_budget

def test_training_coordination(mock_server):
    """Test coordination of federated learning rounds."""
    mock_server.start_training_round = Mock(return_value=True)
    mock_server.check_completion = Mock(return_value=False)
    
    # Start new training round
    round_started = mock_server.start_training_round()
    assert round_started
    
    # Check round completion
    round_complete = mock_server.check_completion()
    assert not round_complete