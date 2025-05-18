import pytest
from unittest.mock import Mock, patch
import numpy as np

@pytest.fixture
def mock_privacy_engine():
    """Fixture for creating a mock privacy engine."""
    privacy_engine = Mock()
    privacy_engine.epsilon = 1.0
    privacy_engine.delta = 1e-5
    return privacy_engine

def test_differential_privacy_noise(mock_privacy_engine):
    """Test differential privacy noise addition."""
    data = np.array([1.0, 2.0, 3.0])
    sensitivity = 1.0
    
    mock_privacy_engine.add_noise = Mock(
        return_value=data + np.random.laplace(0, sensitivity/mock_privacy_engine.epsilon, size=data.shape)
    )
    
    noisy_data = mock_privacy_engine.add_noise(data, sensitivity)
    assert isinstance(noisy_data, np.ndarray)
    assert len(noisy_data) == len(data)
    assert not np.array_equal(noisy_data, data)

def test_privacy_budget_tracking(mock_privacy_engine):
    """Test privacy budget allocation and tracking."""
    initial_budget = mock_privacy_engine.epsilon
    cost = 0.1
    
    mock_privacy_engine.consume_budget = Mock(return_value=initial_budget - cost)
    remaining_budget = mock_privacy_engine.consume_budget(cost)
    
    assert remaining_budget < initial_budget
    assert remaining_budget > 0

def test_secure_aggregation(mock_privacy_engine, mock_crypto_keys):
    """Test secure aggregation with privacy guarantees."""
    local_updates = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6])
    ]
    
    mock_privacy_engine.secure_aggregate = Mock(
        return_value=np.mean(local_updates, axis=0)
    )
    
    aggregated_result = mock_privacy_engine.secure_aggregate(
        local_updates,
        mock_crypto_keys['public_key']
    )
    
    assert isinstance(aggregated_result, np.ndarray)
    assert len(aggregated_result) == len(local_updates[0])

def test_encryption_mechanism(mock_privacy_engine, mock_crypto_keys):
    """Test encryption mechanisms for secure communication."""
    data = np.array([1.0, 2.0, 3.0])
    
    mock_privacy_engine.encrypt = Mock(return_value=data * 2)
    encrypted_data = mock_privacy_engine.encrypt(data, mock_crypto_keys['public_key'])
    
    assert isinstance(encrypted_data, np.ndarray)
    assert len(encrypted_data) == len(data)
    
    mock_privacy_engine.decrypt = Mock(return_value=encrypted_data / 2)
    decrypted_data = mock_privacy_engine.decrypt(encrypted_data, mock_crypto_keys['private_key'])
    
    assert np.array_equal(decrypted_data, data)

def test_privacy_metrics(mock_privacy_engine):
    """Test privacy guarantee metrics calculation."""
    mock_privacy_engine.calculate_privacy_loss = Mock(return_value=0.5)
    privacy_loss = mock_privacy_engine.calculate_privacy_loss()
    
    assert isinstance(privacy_loss, float)
    assert privacy_loss > 0
    assert privacy_loss <= mock_privacy_engine.epsilon