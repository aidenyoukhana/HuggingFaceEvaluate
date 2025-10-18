"""Tests for the EvaluationReport component."""

import pytest
from evaluate import EvaluationReport


class TestEvaluationReport:
    """Test cases for EvaluationReport."""

    def test_report_creation(self):
        """Test creating an evaluation report."""
        report = EvaluationReport(
            task_type="text-classification",
            metrics=["accuracy", "f1"],
            results={"accuracy": {"value": 0.85}, "f1": {"value": 0.82}},
            dataset_info={"size": 1000}
        )

        assert report.task_type == "text-classification"
        assert len(report.metrics) == 2
        assert report.results["accuracy"]["value"] == 0.85

    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = EvaluationReport(
            task_type="test",
            metrics=["acc"],
            results={"acc": 0.9},
            dataset_info={"size": 100}
        )

        data = report.to_dict()
        assert data["task_type"] == "test"
        assert data["results"]["acc"] == 0.9

    def test_report_from_dict(self):
        """Test creating report from dictionary."""
        data = {
            "task_type": "test",
            "metrics": ["acc"],
            "results": {"acc": 0.9},
            "dataset_info": {"size": 100}
        }

        report = EvaluationReport.from_dict(data)
        assert report.task_type == "test"
        assert report.results["acc"] == 0.9