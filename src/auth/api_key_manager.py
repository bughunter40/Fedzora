"""API Key Management Module for Federated Learning Framework.

Provides secure API key management capabilities including key generation,
validation, rotation, and access control for authentication and authorization.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages API keys for secure authentication and authorization."""

    def __init__(self, storage_dir: str = 'auth_data'):
        """Initialize API key manager.

        Args:
            storage_dir: Directory to store encrypted API key data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.key_file = self.storage_dir / 'api_keys.enc'
        self.master_key = self._generate_master_key()
        self.fernet = Fernet(self.master_key)
        self.keys: Dict[str, dict] = self._load_keys()

    def _generate_master_key(self) -> bytes:
        """Generate or load master encryption key.

        Returns:
            Bytes containing the master encryption key
        """
        key_path = self.storage_dir / 'master.key'
        if key_path.exists():
            return key_path.read_bytes()
        
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(os.urandom(32)))
        key_path.write_bytes(key)
        return key

    def _load_keys(self) -> Dict[str, dict]:
        """Load encrypted API keys from storage.

        Returns:
            Dictionary containing API key data
        """
        if not self.key_file.exists():
            return {}
        
        try:
            encrypted_data = self.key_file.read_bytes()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data)
        except Exception as e:
            logger.error(f'Failed to load API keys: {e}')
            return {}

    def _save_keys(self) -> None:
        """Save API keys to encrypted storage."""
        try:
            data = json.dumps(self.keys).encode()
            encrypted_data = self.fernet.encrypt(data)
            self.key_file.write_bytes(encrypted_data)
        except Exception as e:
            logger.error(f'Failed to save API keys: {e}')

    def generate_api_key(self, user_id: str, expiry_days: int = 30) -> Tuple[str, str]:
        """Generate a new API key pair.

        Args:
            user_id: Unique identifier for the key owner
            expiry_days: Number of days until key expires

        Returns:
            Tuple containing (api_key_id, api_key_secret)
        """
        key_id = base64.urlsafe_b64encode(os.urandom(18)).decode('ascii')
        key_secret = base64.urlsafe_b64encode(os.urandom(32)).decode('ascii')
        
        # Hash the secret for storage
        secret_hash = hashlib.sha256(key_secret.encode()).hexdigest()
        
        expiry = datetime.utcnow() + timedelta(days=expiry_days)
        self.keys[key_id] = {
            'user_id': user_id,
            'secret_hash': secret_hash,
            'expiry': expiry.isoformat(),
            'created_at': datetime.utcnow().isoformat(),
            'last_used': None,
            'is_active': True
        }
        
        self._save_keys()
        return key_id, key_secret

    def validate_api_key(self, key_id: str, key_secret: str) -> bool:
        """Validate an API key pair.

        Args:
            key_id: API key identifier
            key_secret: API key secret

        Returns:
            Boolean indicating if the key is valid
        """
        if key_id not in self.keys:
            return False

        key_data = self.keys[key_id]
        if not key_data['is_active']:
            return False

        # Check expiry
        expiry = datetime.fromisoformat(key_data['expiry'])
        if datetime.utcnow() > expiry:
            key_data['is_active'] = False
            self._save_keys()
            return False

        # Validate secret
        secret_hash = hashlib.sha256(key_secret.encode()).hexdigest()
        if not hmac.compare_digest(secret_hash, key_data['secret_hash']):
            return False

        # Update last used timestamp
        key_data['last_used'] = datetime.utcnow().isoformat()
        self._save_keys()
        return True

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key.

        Args:
            key_id: API key identifier

        Returns:
            Boolean indicating if the key was successfully revoked
        """
        if key_id not in self.keys:
            return False

        self.keys[key_id]['is_active'] = False
        self._save_keys()
        return True

    def get_key_info(self, key_id: str) -> Optional[dict]:
        """Get information about an API key.

        Args:
            key_id: API key identifier

        Returns:
            Dictionary containing key information or None if not found
        """
        key_data = self.keys.get(key_id)
        if not key_data:
            return None

        return {
            'user_id': key_data['user_id'],
            'created_at': key_data['created_at'],
            'expiry': key_data['expiry'],
            'last_used': key_data['last_used'],
            'is_active': key_data['is_active']
        }

    def rotate_api_key(self, key_id: str) -> Optional[Tuple[str, str]]:
        """Rotate an API key while maintaining the same key ID.

        Args:
            key_id: API key identifier

        Returns:
            Tuple containing (api_key_id, new_api_key_secret) or None if key not found
        """
        if key_id not in self.keys:
            return None

        key_data = self.keys[key_id]
        if not key_data['is_active']:
            return None

        # Generate new secret
        new_secret = base64.urlsafe_b64encode(os.urandom(32)).decode('ascii')
        key_data['secret_hash'] = hashlib.sha256(new_secret.encode()).hexdigest()
        key_data['last_used'] = datetime.utcnow().isoformat()
        
        self._save_keys()
        return key_id, new_secret