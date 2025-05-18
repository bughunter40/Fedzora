"""Authentication Module for Federated Learning Framework.

Provides secure authentication and authorization capabilities including
API key management, access control, and secure token handling.
"""

from .api_key_manager import APIKeyManager

__all__ = ['APIKeyManager']