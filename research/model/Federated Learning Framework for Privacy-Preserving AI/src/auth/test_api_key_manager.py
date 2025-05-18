"""Test module for API key management functionality.

Provides test cases to validate API key generation, validation,
rotation, and revocation capabilities.
"""

import unittest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from .api_key_manager import APIKeyManager

class TestAPIKeyManager(unittest.TestCase):
    """Test cases for APIKeyManager class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.test_dir = tempfile.mkdtemp()
        self.key_manager = APIKeyManager(storage_dir=self.test_dir)

    def test_api_key_generation(self):
        """Test API key generation and validation."""
        # Generate a new API key
        user_id = 'test_user'
        key_id, key_secret = self.key_manager.generate_api_key(user_id)

        # Verify key format
        self.assertIsInstance(key_id, str)
        self.assertIsInstance(key_secret, str)
        self.assertTrue(len(key_id) > 0)
        self.assertTrue(len(key_secret) > 0)

        # Validate the key
        self.assertTrue(self.key_manager.validate_api_key(key_id, key_secret))

    def test_invalid_api_key(self):
        """Test validation of invalid API keys."""
        # Generate a valid key
        key_id, key_secret = self.key_manager.generate_api_key('test_user')

        # Test with invalid secret
        self.assertFalse(self.key_manager.validate_api_key(key_id, 'invalid_secret'))

        # Test with invalid key ID
        self.assertFalse(self.key_manager.validate_api_key('invalid_id', key_secret))

    def test_key_expiration(self):
        """Test API key expiration."""
        # Generate a key that expires in 1 day
        key_id, key_secret = self.key_manager.generate_api_key('test_user', expiry_days=1)

        # Key should be valid initially
        self.assertTrue(self.key_manager.validate_api_key(key_id, key_secret))

        # Manually expire the key
        self.key_manager.keys[key_id]['expiry'] = (
            datetime.utcnow() - timedelta(days=1)
        ).isoformat()

        # Key should be invalid after expiration
        self.assertFalse(self.key_manager.validate_api_key(key_id, key_secret))

    def test_key_revocation(self):
        """Test API key revocation."""
        # Generate a new key
        key_id, key_secret = self.key_manager.generate_api_key('test_user')

        # Key should be valid initially
        self.assertTrue(self.key_manager.validate_api_key(key_id, key_secret))

        # Revoke the key
        self.assertTrue(self.key_manager.revoke_api_key(key_id))

        # Key should be invalid after revocation
        self.assertFalse(self.key_manager.validate_api_key(key_id, key_secret))

    def test_key_rotation(self):
        """Test API key rotation."""
        # Generate initial key
        key_id, initial_secret = self.key_manager.generate_api_key('test_user')

        # Rotate the key
        rotation_result = self.key_manager.rotate_api_key(key_id)
        self.assertIsNotNone(rotation_result)
        rotated_key_id, new_secret = rotation_result

        # Verify rotated key
        self.assertEqual(key_id, rotated_key_id)
        self.assertNotEqual(initial_secret, new_secret)

        # Old secret should be invalid
        self.assertFalse(self.key_manager.validate_api_key(key_id, initial_secret))

        # New secret should be valid
        self.assertTrue(self.key_manager.validate_api_key(key_id, new_secret))

    def test_key_info_retrieval(self):
        """Test API key information retrieval."""
        # Generate a key
        user_id = 'test_user'
        key_id, _ = self.key_manager.generate_api_key(user_id)

        # Get key info
        key_info = self.key_manager.get_key_info(key_id)

        # Verify key info
        self.assertIsNotNone(key_info)
        self.assertEqual(key_info['user_id'], user_id)
        self.assertTrue(key_info['is_active'])
        self.assertIsNotNone(key_info['created_at'])
        self.assertIsNotNone(key_info['expiry'])

    def test_persistence(self):
        """Test API key persistence across manager instances."""
        # Generate a key with the first manager instance
        key_id, key_secret = self.key_manager.generate_api_key('test_user')

        # Create a new manager instance
        new_manager = APIKeyManager(storage_dir=self.test_dir)

        # Verify key is still valid with new instance
        self.assertTrue(new_manager.validate_api_key(key_id, key_secret))

if __name__ == '__main__':
    unittest.main()