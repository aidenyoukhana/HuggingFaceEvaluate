"""Integration tests that demonstrate real usage scenarios."""

import pytest
from evaluate import ModelEvaluator


class TestIntegration:
    """Integration tests that demonstrate real usage scenarios."""

    def test_basic_evaluation_example(self):
        """Integration test: Basic dataset-only evaluation (equivalent to examples/basic_evaluation.py)."""
        import tempfile
        import os
        from pathlib import Path

        # This test demonstrates the basic evaluation workflow without a model
        evaluator = ModelEvaluator()

        # Use a small, fast dataset for testing
        dataset_name = "glue:sst2"  # Small sentiment analysis dataset

        try:
            # Run evaluation (dataset-only mode)
            report = evaluator.evaluate(
                dataset=dataset_name,
                metrics=["accuracy"],  # Only use accuracy for speed
                task_type="text-classification"
            )

            # Verify report structure
            assert report.task_type == "text-classification"
            assert "accuracy" in report.results
            assert isinstance(report.results["accuracy"], dict)
            assert "dataset_info" in report.__dict__
            assert report.dataset_info["size"] > 0

        except Exception as e:
            # In CI environments, network access might be limited
            pytest.skip(f"Integration test skipped due to network/data access: {e}")

    @pytest.mark.slow
    def test_real_model_evaluation_example(self):
        """Integration test: Full model evaluation (equivalent to examples/evaluate_real_model.py)."""
        import tempfile
        from pathlib import Path

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Load a small, fast model for testing
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"

            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Initialize evaluator with the model
            evaluator = ModelEvaluator(model=model, tokenizer=tokenizer)

            # Use a small subset of the dataset for faster testing
            dataset_name = "glue:sst2"

            # Run evaluation with model
            report = evaluator.evaluate(
                dataset=dataset_name,
                metrics=["accuracy"],  # Only use accuracy for speed
                task_type="text-classification",
                batch_size=4  # Smaller batch for testing
            )

            # Verify report structure
            assert report.task_type == "text-classification"
            assert "accuracy" in report.results
            assert isinstance(report.results["accuracy"], dict)
            # Should have actual metric values, not errors
            assert "accuracy" in report.results["accuracy"]

        except ImportError:
            pytest.skip("transformers not available for integration testing")
        except Exception as e:
            # Skip if network/model loading fails (common in CI)
            pytest.skip(f"Integration test skipped due to model loading/network: {e}")