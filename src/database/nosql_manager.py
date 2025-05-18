from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
import json
import os

class NoSQLDatabaseManager:
    """NoSQL database manager for handling unstructured data and client configurations."""
    
    def __init__(self, data_dir: str = 'nosql_data'):
        self.data_dir = data_dir
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize directory structure for NoSQL storage."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'client_configs'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'training_logs'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'model_metadata'), exist_ok=True)
        logger.info("Initialized NoSQL storage directories")
    
    def _get_file_path(self, collection: str, document_id: str) -> str:
        """Get the file path for a document in a collection."""
        return os.path.join(self.data_dir, collection, f"{document_id}.json")
    
    def save_client_config(self, client_id: int, config: Dict[str, Any]) -> None:
        """Save client-specific configuration data."""
        file_path = self._get_file_path('client_configs', str(client_id))
        config['last_updated'] = datetime.now().isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.debug(f"Saved configuration for client {client_id}")
    
    def get_client_config(self, client_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve client-specific configuration data."""
        file_path = self._get_file_path('client_configs', str(client_id))
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def save_training_log(self, client_id: int, round_number: int,
                         log_data: Dict[str, Any]) -> None:
        """Save detailed training log data."""
        log_id = f"{client_id}_{round_number}"
        file_path = self._get_file_path('training_logs', log_id)
        
        log_data['timestamp'] = datetime.now().isoformat()
        log_data['client_id'] = client_id
        log_data['round_number'] = round_number
        
        with open(file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        logger.debug(f"Saved training log for client {client_id}, round {round_number}")
    
    def get_training_logs(self, client_id: int) -> List[Dict[str, Any]]:
        """Retrieve all training logs for a specific client."""
        logs_dir = os.path.join(self.data_dir, 'training_logs')
        logs = []
        
        for filename in os.listdir(logs_dir):
            if filename.startswith(f"{client_id}_") and filename.endswith('.json'):
                with open(os.path.join(logs_dir, filename), 'r') as f:
                    logs.append(json.load(f))
        
        return sorted(logs, key=lambda x: x['round_number'])
    
    def save_model_metadata(self, model_name: str, version: int,
                           metadata: Dict[str, Any]) -> None:
        """Save additional model metadata and configuration."""
        metadata_id = f"{model_name}_v{version}"
        file_path = self._get_file_path('model_metadata', metadata_id)
        
        metadata['model_name'] = model_name
        metadata['version'] = version
        metadata['last_updated'] = datetime.now().isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved metadata for {model_name} version {version}")
    
    def get_model_metadata(self, model_name: str, version: int) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata and configuration."""
        metadata_id = f"{model_name}_v{version}"
        file_path = self._get_file_path('model_metadata', metadata_id)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all available versions of a specific model."""
        metadata_dir = os.path.join(self.data_dir, 'model_metadata')
        versions = []
        
        for filename in os.listdir(metadata_dir):
            if filename.startswith(f"{model_name}_v") and filename.endswith('.json'):
                with open(os.path.join(metadata_dir, filename), 'r') as f:
                    versions.append(json.load(f))
        
        return sorted(versions, key=lambda x: x['version'])