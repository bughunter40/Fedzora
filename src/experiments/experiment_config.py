from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

@dataclass
class ExperimentConfig:
    """Configuration for federated learning experiments."""
    
    # Basic experiment settings
    name: str
    description: str = ""
    base_dir: Path = Path("experiments")
    
    # Model configuration
    model_params: Dict[str, Any] = field(default_factory=dict)
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    num_rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Privacy settings
    privacy_budget: float = 1.0
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    
    # Client configuration
    num_clients: int = 10
    clients_per_round: int = 5
    client_sampling_strategy: str = "random"
    
    # Evaluation settings
    eval_frequency: int = 1
    metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy"])
    
    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to JSON file."""
        if config_path is None:
            config_path = self.base_dir / self.name / "config.json"
            
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, config_path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        # Convert string path back to Path object
        if 'base_dir' in config_dict:
            config_dict['base_dir'] = Path(config_dict['base_dir'])
            
        return cls(**config_dict)
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")