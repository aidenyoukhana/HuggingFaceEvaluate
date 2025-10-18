"""Tests for the ModelEvaluator component."""

import pytest
from unittest.mock import Mock, patch
from evaluate import ModelEvaluator, MetricRegistry, DatasetLoader


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()

        assert evaluator.device == "auto"
        assert isinstance(evaluator.metric_registry, MetricRegistry)
        assert isinstance(evaluator.dataset_loader, DatasetLoader)

    @patch('evaluate.evaluator.DatasetLoader.load_dataset')
    @patch('evaluate.evaluator.MetricRegistry.get_metric')
    def test_evaluate_without_model(self, mock_get_metric, mock_load_dataset):
        """Test evaluation without a model (dataset-only metrics)."""
        evaluator = ModelEvaluator()

        # Mock dataset and metric
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset

        mock_metric = Mock()
        mock_metric.name = "accuracy"
        mock_metric.compute.return_value = {"accuracy": 0.85}
        mock_get_metric.return_value = mock_metric

        report = evaluator.evaluate(
            dataset="test_dataset",
            metrics=["accuracy"],
            task_type="text-classification"
        )

        assert report.task_type == "text-classification"
        assert "accuracy" in report.results
        assert report.results["accuracy"]["accuracy"] == 0.85