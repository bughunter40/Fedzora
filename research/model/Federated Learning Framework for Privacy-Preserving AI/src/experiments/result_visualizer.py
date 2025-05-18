import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from .result_analyzer import ResultAnalyzer

class ResultVisualizer:
    """Visualizes experiment results and comparisons."""
    
    def __init__(self, analyzer: ResultAnalyzer):
        self.analyzer = analyzer
        self.style_config = {
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        }
        plt.rcParams.update(self.style_config)
    
    def plot_metric_comparison(self, experiment_names: Optional[List[str]] = None,
                             metric: str = 'accuracy', save_path: Optional[str] = None) -> None:
        """Plot comparison of a specific metric across experiments."""
        df = self.analyzer.compare_metrics(experiment_names)
        
        plt.figure()
        metrics_cols = [col for col in df.columns if metric in col]
        
        if not metrics_cols:
            raise ValueError(f"Metric '{metric}' not found in experiment results")
            
        plot_data = df.melt(id_vars=['experiment'],
                           value_vars=metrics_cols,
                           var_name='Statistic',
                           value_name='Value')
        
        sns.barplot(data=plot_data, x='experiment', y='Value', hue='Statistic')
        plt.title(f'{metric.capitalize()} Comparison Across Experiments')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
    def plot_convergence_analysis(self, experiment_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> None:
        """Plot convergence analysis results."""
        convergence_data = self.analyzer.analyze_convergence(experiment_names)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot iterations to converge
        iterations = [data['iterations_to_converge'] for data in convergence_data.values()]
        ax1.bar(convergence_data.keys(), iterations)
        ax1.set_title('Iterations to Convergence')
        ax1.set_xticklabels(convergence_data.keys(), rotation=45)
        
        # Plot stability scores
        stability = [data['stability_score'] for data in convergence_data.values()]
        ax2.bar(convergence_data.keys(), stability)
        ax2.set_title('Training Stability Scores')
        ax2.set_xticklabels(convergence_data.keys(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
    def plot_privacy_analysis(self, experiment_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None) -> None:
        """Plot privacy impact analysis results."""
        privacy_data = self.analyzer.analyze_privacy_impact(experiment_names)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot privacy efficiency
        efficiency = [data['privacy_efficiency'] for data in privacy_data.values()]
        ax1.bar(privacy_data.keys(), efficiency)
        ax1.set_title('Privacy Efficiency (Accuracy/Budget)')
        ax1.set_xticklabels(privacy_data.keys(), rotation=45)
        
        # Plot budget utilization rate
        utilization = [data['budget_utilization']['rate'] for data in privacy_data.values()]
        ax2.bar(privacy_data.keys(), utilization)
        ax2.set_title('Privacy Budget Utilization Rate')
        ax2.set_xticklabels(privacy_data.keys(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
    def plot_training_progress(self, experiment_names: Optional[List[str]] = None,
                              metric: str = 'loss', save_path: Optional[str] = None) -> None:
        """Plot training progress over time for multiple experiments."""
        results = self.analyzer.load_experiments(experiment_names)
        
        plt.figure()
        for exp_name, exp_data in results.items():
            if 'metrics' in exp_data and metric in exp_data['metrics']:
                values = exp_data['metrics'][metric]
                plt.plot(values, label=exp_name)
                
        plt.title(f'{metric.capitalize()} Over Training')
        plt.xlabel('Iteration')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
    def generate_summary_report(self, experiment_names: Optional[List[str]] = None,
                              output_dir: Optional[str] = None) -> None:
        """Generate a comprehensive visual report of experiment results."""
        if output_dir is None:
            output_dir = str(self.analyzer.base_dir / 'reports')
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate individual plots
        self.plot_metric_comparison(experiment_names, 'accuracy',
                                  str(output_path / 'accuracy_comparison.png'))
        self.plot_metric_comparison(experiment_names, 'loss',
                                  str(output_path / 'loss_comparison.png'))
        self.plot_convergence_analysis(experiment_names,
                                     str(output_path / 'convergence_analysis.png'))
        self.plot_privacy_analysis(experiment_names,
                                 str(output_path / 'privacy_analysis.png'))
        self.plot_training_progress(experiment_names, 'accuracy',
                                  str(output_path / 'training_progress.png'))