import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class ExperimentManager:
    """Manages experiment configuration, tracking, and result analysis."""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.metrics: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': [],
            'privacy_budget': [],
            'training_time': [],
            'memory_usage': []
        }
        self.config: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set experiment configuration parameters."""
        self.config = config
        self._save_config()
        
    def start_experiment(self) -> None:
        """Initialize experiment tracking."""
        self.start_time = time.time()
        self.metrics = {key: [] for key in self.metrics.keys()}
        
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics for current experiment iteration."""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        self._save_metrics()
        
    def end_experiment(self) -> Dict[str, Any]:
        """Finalize experiment and return summary statistics."""
        duration = time.time() - (self.start_time or 0)
        
        summary = {
            'duration': duration,
            'total_iterations': len(self.metrics['loss']),
            'final_metrics': {}
        }
        
        for metric_name, values in self.metrics.items():
            if values:
                summary['final_metrics'][metric_name] = {
                    'final': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        self._save_summary(summary)
        return summary
    
    def _save_config(self) -> None:
        """Save experiment configuration to file."""
        config_path = self.experiment_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def _save_metrics(self) -> None:
        """Save current metrics to file."""
        metrics_path = self.experiment_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
    def _save_summary(self, summary: Dict[str, Any]) -> None:
        """Save experiment summary to file."""
        summary_path = self.experiment_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
    
    def load_experiment(self) -> Dict[str, Any]:
        """Load existing experiment data."""
        result = {}
        
        config_path = self.experiment_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                result['config'] = json.load(f)
                
        metrics_path = self.experiment_dir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                result['metrics'] = json.load(f)
                
        summary_path = self.experiment_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                result['summary'] = json.load(f)
                
        return result