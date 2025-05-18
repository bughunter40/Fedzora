import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry

@pytest.fixture
def mock_model_registry():
    """Fixture for creating a mock model registry."""
    registry = Mock(spec=ModelRegistry)
    return registry

@pytest.fixture
def mock_model():
    """Fixture for creating a mock model."""
    model = Mock()
    model.forward = Mock(return_value=np.array([[0.6, 0.4], [0.3, 0.7]]))
    model.backward = Mock(return_value=[np.array([[0.1, -0.1], [-0.2, 0.2]]) for _ in range(5)])
    model.parameters = Mock(return_value=[np.array([[1.0, -1.0], [-1.0, 1.0]]) for _ in range(5)])
    return model

@pytest.fixture
def large_mock_training_data():
    """Fixture for creating large mock training data for stress testing."""
    features = np.random.randn(1000, 4)  # 1000 samples, 4 features
    labels = np.eye(2)[np.random.choice(2, 1000)]  # Binary classification
    return features, labels

def test_model_convergence(mock_model_registry, mock_model, large_mock_training_data):
    """Test model convergence over extended training period."""
    trainer = ModelTrainer(mock_model_registry, epochs=50, batch_size=32)
    initial_budget = 2.0
    
    results = trainer.train_local_model(mock_model, large_mock_training_data, initial_budget)
    
    # Verify convergence
    loss_history = trainer.training_history['loss']
    assert loss_history[-1] < loss_history[0]  # Loss should decrease
    assert results['accuracy'] > 0.6  # Minimum accuracy threshold

def test_privacy_attack_resistance(mock_model_registry, mock_model):
    """Test resistance against privacy attacks."""
    trainer = ModelTrainer(mock_model_registry)
    loss = 1.0
    privacy_budget = 0.1  # Very strict privacy budget
    
    # Generate multiple sets of private gradients
    gradient_sets = []
    for _ in range(10):
        gradients = trainer._compute_private_gradients(mock_model, loss, privacy_budget)
        gradient_sets.append(gradients[0])
    
    # Check gradient diversity (privacy protection)
    gradient_mean = np.mean(gradient_sets, axis=0)
    gradient_std = np.std(gradient_sets, axis=0)
    
    assert np.all(gradient_std > 0)  # Ensure noise addition
    assert not np.any(np.isnan(gradient_mean))  # No NaN values

def test_performance_under_load(mock_model_registry, mock_model, large_mock_training_data):
    """Test system performance under heavy load."""
    trainer = ModelTrainer(mock_model_registry, batch_size=128)
    start_time = time.time()
    
    # Train with large dataset
    trainer.train_local_model(mock_model, large_mock_training_data, privacy_budget=1.0)
    
    training_time = time.time() - start_time
    assert training_time < 60  # Training should complete within reasonable time

def test_fault_tolerance(mock_model_registry, mock_model, mock_training_data):
    """Test system behavior under simulated failures."""
    trainer = ModelTrainer(mock_model_registry)
    
    # Simulate model failure during training
    mock_model.forward.side_effect = [Exception("Simulated failure"), np.array([[0.6, 0.4], [0.3, 0.7]])]
    
    try:
        trainer.train_local_model(mock_model, mock_training_data, privacy_budget=1.0)
    except Exception as e:
        assert "Simulated failure" in str(e)
        assert trainer.training_history['loss']  # History should be preserved

def test_memory_efficiency(mock_model_registry, mock_model, large_mock_training_data):
    """Test memory usage during training."""
    trainer = ModelTrainer(mock_model_registry, batch_size=32)
    initial_params = mock_model.parameters()
    
    # Train with increasing dataset sizes
    for i in range(1, 4):
        subset_size = i * 250
        features, labels = large_mock_training_data
        subset_data = (features[:subset_size], labels[:subset_size])
        
        trainer.train_local_model(mock_model, subset_data, privacy_budget=1.0)
        
        # Verify memory cleanup
        assert len(trainer.training_history['loss']) == trainer.epochs
        assert not hasattr(trainer, '_temp_gradients')

def test_numerical_stability(mock_model_registry, mock_model):
    """Test numerical stability with extreme values."""
    trainer = ModelTrainer(mock_model_registry)
    
    # Test with very small gradients
    small_gradients = [np.array([[1e-10, -1e-10], [-1e-10, 1e-10]])]
    trainer._update_model(mock_model, small_gradients)
    
    # Test with very large gradients
    large_gradients = [np.array([[1e10, -1e10], [-1e10, 1e10]])]
    trainer._update_model(mock_model, large_gradients)
    
    final_params = mock_model.parameters()[0]
    assert not np.any(np.isnan(final_params))  # No NaN values
    assert not np.any(np.isinf(final_params))  # No infinite values

def test_privacy_budget_exhaustion(mock_model_registry, mock_model, mock_training_data):
    """Test system behavior when privacy budget is exhausted."""
    trainer = ModelTrainer(mock_model_registry, epochs=5)
    privacy_budget = 0.1  # Very small privacy budget
    
    results = trainer.train_local_model(mock_model, mock_training_data, privacy_budget)
    
    assert results['privacy_cost'] <= privacy_budget  # Should not exceed budget
    assert len(trainer.training_history['privacy_budget']) <= 5  # Should stop when budget exhausted