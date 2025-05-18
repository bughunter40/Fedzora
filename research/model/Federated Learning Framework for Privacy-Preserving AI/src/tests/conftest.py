import pytest
import os
import sys

# Add src directory to Python path for test imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope='session')
def test_config():
    """Global test configuration fixture."""
    return {
        'test_data_dir': 'tests/data',
        'mock_server_port': 8000,
        'mock_client_count': 3,
        'test_model_path': 'tests/models/test_model.h5',
        'privacy_budget': 1.0,
        'batch_size': 32
    }

@pytest.fixture
def mock_database():
    """Fixture for creating a test database connection."""
    # Initialize test database
    test_db = {
        'users': [],
        'models': [],
        'training_logs': []
    }
    yield test_db
    # Cleanup after tests
    test_db.clear()

@pytest.fixture
def mock_crypto_keys():
    """Fixture for generating test cryptographic keys."""
    return {
        'public_key': 'test_public_key',
        'private_key': 'test_private_key',
        'shared_key': 'test_shared_key'
    }

@pytest.fixture
def mock_client_data():
    """Fixture for generating mock client training data."""
    return {
        'features': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        'labels': [0, 1]
    }