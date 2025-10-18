"""Tests for the MetricRegistry component."""

import pytest
from unittest.mock import Mock
from evaluate import MetricRegistry


class TestMetricRegistry:
    """Test cases for MetricRegistry."""

    def test_get_metric(self):
        """Test loading a metric."""
        registry = MetricRegistry()

        # Mock the load function call in the get_metric method
        mock_metric = Mock()

        # We need to access the hf_evaluate object from the metrics module
        import evaluate.metrics as metrics_module
        original_hf_evaluate = metrics_module.hf_evaluate
        mock_hf_evaluate = Mock()
        mock_hf_evaluate.load.return_value = mock_metric
        metrics_module.hf_evaluate = mock_hf_evaluate

        try:
            metric = registry.get_metric("accuracy")
            mock_hf_evaluate.load.assert_called_once_with("accuracy")
            assert metric == mock_metric
        finally:
            # Restore original
            metrics_module.hf_evaluate = original_hf_evaluate

    def test_add_custom_metric(self):
        """Test adding a custom metric."""
        registry = MetricRegistry()
        custom_metric = Mock()

        registry.add_custom_metric("custom_metric", custom_metric)

        assert "custom_metric" in registry.list_available_metrics()