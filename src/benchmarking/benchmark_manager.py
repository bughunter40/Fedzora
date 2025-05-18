"""Benchmarking Manager for Federated Learning Framework.

Provides comprehensive benchmarking capabilities for evaluating system performance,
privacy guarantees, and resource utilization metrics.
"""

import logging
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    model_accuracy: float
    training_time: float
    communication_cost: float
    privacy_score: float
    memory_usage: float
    cpu_utilization: float

class BenchmarkManager:
    """Manages comprehensive benchmarking of the federated learning system."""

    def __init__(self, output_dir: str = 'benchmarks'):
        """Initialize benchmark manager.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_metrics: Dict[int, BenchmarkMetrics] = {}
        self.start_time = time.time()

    def start_round_benchmark(self, round_idx: int) -> None:
        """Start benchmarking for a training round.

        Args:
            round_idx: Current training round index
        """
        self.start_time = time.time()
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    def end_round_benchmark(
        self,
        round_idx: int,
        model_accuracy: float,
        communication_cost: float,
        privacy_score: float
    ) -> BenchmarkMetrics:
        """End benchmarking for a training round and collect metrics.

        Args:
            round_idx: Current training round index
            model_accuracy: Achieved model accuracy
            communication_cost: Total communication overhead
            privacy_score: Privacy preservation metric

        Returns:
            BenchmarkMetrics containing collected metrics
        """
        training_time = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_usage = current_memory - self.initial_memory
        cpu_utilization = psutil.cpu_percent()

        metrics = BenchmarkMetrics(
            model_accuracy=model_accuracy,
            training_time=training_time,
            communication_cost=communication_cost,
            privacy_score=privacy_score,
            memory_usage=memory_usage,
            cpu_utilization=cpu_utilization
        )

        self.round_metrics[round_idx] = metrics
        return metrics

    def calculate_system_efficiency(self) -> Dict[str, float]:
        """Calculate overall system efficiency metrics.

        Returns:
            Dictionary containing aggregated efficiency metrics
        """
        if not self.round_metrics:
            return {}

        total_rounds = len(self.round_metrics)
        avg_metrics = {
            'avg_accuracy': np.mean([m.model_accuracy for m in self.round_metrics.values()]),
            'avg_training_time': np.mean([m.training_time for m in self.round_metrics.values()]),
            'avg_communication_cost': np.mean([m.communication_cost for m in self.round_metrics.values()]),
            'avg_privacy_score': np.mean([m.privacy_score for m in self.round_metrics.values()]),
            'avg_memory_usage': np.mean([m.memory_usage for m in self.round_metrics.values()]),
            'avg_cpu_utilization': np.mean([m.cpu_utilization for m in self.round_metrics.values()]),
            'total_training_time': sum(m.training_time for m in self.round_metrics.values()),
            'total_communication_cost': sum(m.communication_cost for m in self.round_metrics.values())
        }

        return avg_metrics

    def export_benchmark_results(self, filename: str = 'benchmark_results.txt') -> None:
        """Export benchmark results to a file.

        Args:
            filename: Name of the output file
        """
        efficiency_metrics = self.calculate_system_efficiency()
        
        with open(self.output_dir / filename, 'w') as f:
            f.write('=== Federated Learning System Benchmark Results ===\n\n')
            
            f.write('System Efficiency Metrics:\n')
            for metric, value in efficiency_metrics.items():
                f.write(f'{metric}: {value:.4f}\n')
            
            f.write('\nDetailed Round Metrics:\n')
            for round_idx, metrics in self.round_metrics.items():
                f.write(f'\nRound {round_idx + 1}:\n')
                f.write(f'Model Accuracy: {metrics.model_accuracy:.4f}\n')
                f.write(f'Training Time: {metrics.training_time:.2f}s\n')
                f.write(f'Communication Cost: {metrics.communication_cost:.2f} bytes\n')
                f.write(f'Privacy Score: {metrics.privacy_score:.4f}\n')
                f.write(f'Memory Usage: {metrics.memory_usage:.2f} MB\n')
                f.write(f'CPU Utilization: {metrics.cpu_utilization:.2f}%\n')

        logger.info(f'Benchmark results exported to {self.output_dir / filename}')