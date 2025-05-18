import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from .experiment_manager import ExperimentManager

class ResultAnalyzer:
    """Analyzes and compares results from multiple experiments."""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        
    def load_experiments(self, experiment_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Load data from multiple experiments for comparison."""
        results = {}
        
        if experiment_names is None:
            # Load all experiments in the base directory
            experiment_names = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        
        for name in experiment_names:
            exp_manager = ExperimentManager(name, str(self.base_dir))
            results[name] = exp_manager.load_experiment()
            
        return results
    
    def compare_metrics(self, experiment_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare final metrics across experiments."""
        results = self.load_experiments(experiment_names)
        comparison = []
        
        for exp_name, exp_data in results.items():
            if 'summary' in exp_data and 'final_metrics' in exp_data['summary']:
                metrics = exp_data['summary']['final_metrics']
                row = {'experiment': exp_name}
                
                for metric_name, values in metrics.items():
                    for stat_name, value in values.items():
                        row[f'{metric_name}_{stat_name}'] = value
                        
                comparison.append(row)
        
        return pd.DataFrame(comparison)
    
    def analyze_convergence(self, experiment_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze training convergence across experiments."""
        results = self.load_experiments(experiment_names)
        convergence_analysis = {}
        
        for exp_name, exp_data in results.items():
            if 'metrics' in exp_data and 'loss' in exp_data['metrics']:
                loss_values = exp_data['metrics']['loss']
                
                # Calculate convergence metrics
                convergence_analysis[exp_name] = {
                    'iterations_to_converge': self._find_convergence_point(loss_values),
                    'final_loss': loss_values[-1] if loss_values else None,
                    'loss_improvement': (loss_values[0] - loss_values[-1]) / loss_values[0] 
                        if loss_values else None,
                    'stability_score': self._calculate_stability(loss_values)
                }
        
        return convergence_analysis
    
    def analyze_privacy_impact(self, experiment_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze the impact of privacy budget on model performance."""
        results = self.load_experiments(experiment_names)
        privacy_analysis = {}
        
        for exp_name, exp_data in results.items():
            if 'metrics' in exp_data:
                metrics = exp_data['metrics']
                if all(key in metrics for key in ['privacy_budget', 'accuracy']):
                    privacy_analysis[exp_name] = {
                        'privacy_efficiency': self._calculate_privacy_efficiency(
                            metrics['privacy_budget'],
                            metrics['accuracy']
                        ),
                        'budget_utilization': self._analyze_budget_utilization(
                            metrics['privacy_budget']
                        )
                    }
        
        return privacy_analysis
    
    def _find_convergence_point(self, loss_values: List[float], threshold: float = 0.01) -> Optional[int]:
        """Find the iteration where loss stabilizes."""
        if not loss_values:
            return None
            
        window_size = 5
        for i in range(len(loss_values) - window_size):
            window = loss_values[i:i + window_size]
            if np.std(window) < threshold:
                return i
        
        return len(loss_values)
    
    def _calculate_stability(self, values: List[float], window_size: int = 5) -> float:
        """Calculate training stability score."""
        if len(values) < window_size:
            return 0.0
            
        windows = [values[i:i + window_size] for i in range(len(values) - window_size + 1)]
        return np.mean([np.std(window) for window in windows])
    
    def _calculate_privacy_efficiency(self, 
                                     privacy_budgets: List[float], 
                                     accuracies: List[float]) -> float:
        """Calculate privacy efficiency (accuracy gained per privacy budget unit)."""
        if not privacy_budgets or not accuracies:
            return 0.0
            
        total_budget = sum(privacy_budgets)
        final_accuracy = accuracies[-1]
        return final_accuracy / total_budget if total_budget > 0 else 0.0
    
    def _analyze_budget_utilization(self, privacy_budgets: List[float]) -> Dict[str, float]:
        """Analyze how privacy budget is utilized over time."""
        if not privacy_budgets:
            return {'rate': 0.0, 'efficiency': 0.0}
            
        budget_usage = np.cumsum(privacy_budgets)
        rate = np.polyfit(range(len(budget_usage)), budget_usage, 1)[0]
        efficiency = budget_usage[-1] / len(budget_usage)
        
        return {
            'rate': rate,
            'efficiency': efficiency
        }