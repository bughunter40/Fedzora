from phe import paillier
import torch
import numpy as np
from typing import Union, Dict
from loguru import logger

class HomomorphicEncryption:
    """Homomorphic encryption implementation for secure parameter transmission."""
    
    def __init__(self, key_length: int = 2048):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)
        logger.info(f"Initialized homomorphic encryption with {key_length}-bit key")
    
    def encrypt(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Encrypt data using homomorphic encryption."""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
            is_tensor = True
        else:
            is_tensor = False
            
        # Flatten data for encryption
        original_shape = data.shape
        flattened = data.flatten()
        
        # Encrypt each value
        encrypted = np.array([self.public_key.encrypt(float(x)) for x in flattened])
        encrypted = encrypted.reshape(original_shape)
        
        return torch.from_numpy(encrypted) if is_tensor else encrypted
    
    def decrypt(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Decrypt homomorphically encrypted data."""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
            is_tensor = True
        else:
            is_tensor = False
            
        # Flatten data for decryption
        original_shape = data.shape
        flattened = data.flatten()
        
        # Decrypt each value
        decrypted = np.array([self.private_key.decrypt(x) for x in flattened])
        decrypted = decrypted.reshape(original_shape)
        
        return torch.from_numpy(decrypted) if is_tensor else decrypted
    
    def get_public_key(self) -> paillier.PaillierPublicKey:
        """Get public key for encryption."""
        return self.public_key
    
    def encrypt_model_params(self, model_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encrypt model parameters."""
        encrypted_params = {}
        for name, param in model_params.items():
            encrypted_params[name] = self.encrypt(param)
        return encrypted_params
    
    def decrypt_model_params(self, encrypted_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decrypt model parameters."""
        decrypted_params = {}
        for name, param in encrypted_params.items():
            decrypted_params[name] = self.decrypt(param)
        return decrypted_params