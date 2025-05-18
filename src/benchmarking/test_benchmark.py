"""Test module for benchmarking functionality.

Provides test cases to validate the benchmark manager's capabilities
for measuring system performance and resource utilization.
"""

import unittest
import tempfile
from pathlib import Path
from .benchmark_manager import BenchmarkManager, BenchmarkMetrics

class TestBenchmarkManager(unittest.TestCase):
    """Test cases for BenchmarkManager class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.test_dir = tempfile.mkdtemp()
        self.benchmark_manager = BenchmarkManager(output_dir=self.test_dir)

    def test_round_benchmarking(self):
        """Test benchmarking for a single training round."""
        # Start benchmarking
        round_idx = 0
        self.benchmark_manager.start_round_benchmark(round_idx)

        # Simulate some work
        test_accuracy = 0.95
        test_communication = 1000.0
        test_privacy = 0.85

        # End benchmarking and collect metrics
        metrics = self.benchmark_manager.end_round_benchmark(
            round_idx=round_idx,
            model_accuracy=test_accuracy,
            communication_cost=test_communication,
            privacy_score=test_privacy
        )

        # Verify metrics
        self.assertIsInstance(metrics, BenchmarkMetrics)
        self.assertEqual(metrics.model_accuracy, test_accuracy)
        self.assertEqual(metrics.communication_cost, test_communication)
        self.assertEqual(metrics.privacy_score, test_privacy)
        self.assertGreater(metrics.training_time, 0)
        self.assertGreaterEqual(metrics.memory_usage, 0)
        self.assertGreaterEqual(metrics.cpu_utilization, 0)

    def test_system_efficiency_calculation(self):
        """Test calculation of system efficiency metrics."""
        # Add multiple rounds of metrics
        for i in range(3):
            self.benchmark_manager.start_round_benchmark(i)
            self.benchmark_manager.end_round_benchmark(
                round_idx=i,
                model_accuracy=0.9 + i * 0.02,
                communication_cost=1000.0 * (i + 1),
                privacy_score=0.85 - i * 0.05
            )

        # Calculate efficiency metrics
        efficiency_metrics = self.benchmark_manager.calculate_system_efficiency()

        # Verify metrics
        self.assertIn('avg_accuracy', efficiency_metrics)
        self.assertIn('avg_training_time', efficiency_metrics)
        self.assertIn('avg_communication_cost', efficiency_metrics)
        self.assertIn('avg_privacy_score', efficiency_metrics)
        self.assertIn('total_training_time', efficiency_metrics)
        self.assertIn('total_communication_cost', efficiency_metrics)

    def test_export_results(self):
        """Test exporting benchmark results to file."""
        # Add a test round
        round_idx = 0
        self.benchmark_manager.start_round_benchmark(round_idx)
        self.benchmark_manager.end_round_benchmark(
            round_idx=round_idx,
            model_accuracy=0.95,
            communication_cost=1000.0,
            privacy_score=0.85
        )

        # Export results
        test_filename = 'test_benchmark_results.txt'
        self.benchmark_manager.export_benchmark_results(test_filename)

        # Verify file exists and contains content
        result_file = Path(self.test_dir) / test_filename
        self.assertTrue(result_file.exists())
        self.assertGreater(result_file.stat().st_size, 0)

if __name__ == '__main__':
    unittest.main()