import pytest
import numpy as np
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
    model.backward = Mock(return_value=[np.array([[0.1, -0.1], [-0.2, 0.2]])])
    model.parameters = Mock(return_value=[np.array([[1.0, -1.0], [-1.0, 1.0]])])
    return model

@pytest.fixture
def mock_training_data():
    """Fixture for creating mock training data."""
    features = np.random.randn(10, 4)  # 10 samples, 4 features
    labels = np.eye(2)[np.random.choice(2, 10)]  # Binary classification
    return features, labels

def test_trainer_initialization(mock_model_registry):
    """Test ModelTrainer initialization."""
    trainer = ModelTrainer(mock_model_registry)
    
    assert trainer.learning_rate == 0.01
    assert trainer.batch_size == 32
    assert trainer.epochs == 10
    assert isinstance(trainer.training_history, dict)
    assert all(key in trainer.training_history for key in ['loss', 'accuracy', 'privacy_budget'])

def test_local_training(mock_model_registry, mock_model, mock_training_data):
    """Test local model training process."""
    trainer = ModelTrainer(mock_model_registry, epochs=2, batch_size=5)
    privacy_budget = 1.0
    
    results = trainer.train_local_model(mock_model, mock_training_data, privacy_budget)
    
    assert isinstance(results, dict)
    assert all(key in results for key in ['loss', 'accuracy', 'model_updates', 'privacy_cost'])
    assert results['privacy_cost'] == privacy_budget
    assert len(trainer.training_history['loss']) == 2  # Two epochs

def test_private_gradient_computation(mock_model_registry, mock_model):
    """Test privacy-preserving gradient computation."""
    trainer = ModelTrainer(mock_model_registry)
    loss = 0.5
    privacy_budget = 1.0
    
    gradients = trainer._compute_private_gradients(mock_model, loss, privacy_budget)
    
    assert isinstance(gradients, list)
    assert len(gradients) > 0
    assert isinstance(gradients[0], np.ndarray)
    # Verify noise addition
    raw_gradients = mock_model.backward(loss)
    assert not np.array_equal(gradients[0], raw_gradients[0])

def test_model_update_mechanism(mock_model_registry, mock_model):
    """Test model parameter update process."""
    trainer = ModelTrainer(mock_model_registry, learning_rate=0.1)
    gradients = [np.array([[0.1, -0.1], [-0.1, 0.1]])]  # Mock gradients
    
    initial_params = mock_model.parameters()[0].copy()
    trainer._update_model(mock_model, gradients)
    
    # Verify parameters were updated
    assert not np.array_equal(mock_model.parameters()[0], initial_params)

def test_training_metrics(mock_model_registry, mock_model, mock_training_data):
    """Test training metrics tracking."""
    trainer = ModelTrainer(mock_model_registry, epochs=3)
    trainer.train_local_model(mock_model, mock_training_data, privacy_budget=1.0)
    
    metrics = trainer.get_training_metrics()
    
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ['loss', 'accuracy', 'privacy_budget'])
    assert len(metrics['loss']) == 3  # Three epochs
    assert len(metrics['accuracy']) == 3
    assert len(metrics['privacy_budget']) == 3
    assert all(isinstance(val, float) for val in metrics['loss'])

def test_privacy_budget_consumption(mock_model_registry, mock_model, mock_training_data):
    """Test privacy budget management during training."""
    trainer = ModelTrainer(mock_model_registry, epochs=2)
    initial_budget = 1.0
    
    results = trainer.train_local_model(mock_model, mock_training_data, initial_budget)
    
    # Verify privacy budget consumption
    assert results['privacy_cost'] == initial_budget
    assert len(trainer.training_history['privacy_budget']) == 2
    assert all(budget == initial_budget/2 for budget in trainer.training_history['privacy_budget'])