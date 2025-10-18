"""Model Evaluator

Core evaluation functionality for machine learning models.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json

import evaluate
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from .config import config

from .metrics import MetricRegistry
from .datasets import DatasetLoader
from .reports import EvaluationReport

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Main evaluator class for running model evaluations."""

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: str = None,
    ):
        """Initialize the evaluator.

        Args:
            model: Pre-trained model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run evaluation on ('cpu', 'cuda', 'auto')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or config.DEVICE
        self.metric_registry = MetricRegistry()
        self.dataset_loader = DatasetLoader()

    def evaluate(
        self,
        dataset: Union[str, Dataset] = None,
        metrics: Optional[List[str]] = None,
        task_type: str = "text-classification",
        batch_size: Optional[int] = None,
        **kwargs
    ) -> EvaluationReport:
        """Run evaluation on a dataset with specified metrics.

        Args:
            dataset: Dataset name or Dataset object (uses DEFAULT_DATASET if None)
            metrics: List of metric names to compute (uses DEFAULT_METRICS if None)
            task_type: Type of task (e.g., 'text-classification', 'question-answering')
            batch_size: Batch size for evaluation (uses config default if None)
            **kwargs: Additional arguments for evaluation

        Returns:
            EvaluationReport containing results
        """
        # Use config defaults if not provided
        dataset = dataset or config.DEFAULT_DATASET
        metrics = metrics or config.DEFAULT_METRICS
        batch_size = batch_size or config.BATCH_SIZE

        logger.info(f"Starting evaluation with metrics: {metrics}")

        # Load dataset if string provided
        if isinstance(dataset, str):
            dataset = self.dataset_loader.load_dataset(dataset)

        # Load metrics
        loaded_metrics = []
        for metric_name in metrics:
            metric = self.metric_registry.get_metric(metric_name)
            loaded_metrics.append(metric)

        # Run evaluation
        results = {}
        for metric in loaded_metrics:
            logger.info(f"Computing metric: {metric.name}")
            try:
                if self.model and self.tokenizer:
                    # Model-based evaluation
                    result = self._evaluate_with_model(
                        dataset, metric, task_type, batch_size, **kwargs
                    )
                else:
                    # Dataset-only evaluation (for reference metrics)
                    result = metric.compute(dataset=dataset, **kwargs)

                results[metric.name] = result
            except Exception as e:
                logger.error(f"Failed to compute metric {metric.name}: {e}")
                results[metric.name] = {"error": str(e)}

        return EvaluationReport(
            task_type=task_type,
            metrics=metrics,
            results=results,
            dataset_info={"size": len(dataset)},
        )

    def _evaluate_with_model(
        self,
        dataset: Dataset,
        metric: Any,  # evaluate metric function
        task_type: str,
        batch_size: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate using model predictions."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be provided for model-based evaluation")

        logger.info(f"Running inference with {self.model.__class__.__name__}")

        # Get predictions from model
        predictions = []
        references = []

        for batch in dataset.iter(batch_size=batch_size):
            # Prepare inputs based on task type
            if task_type == "text-classification":
                inputs = self.tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                if hasattr(self.model, 'to'):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                    predictions.extend(preds)
                    references.extend(batch["label"])

            elif task_type == "question-answering":
                # TODO: Implement QA evaluation
                pass
            else:
                raise NotImplementedError(f"Task type {task_type} not implemented yet")

        # Compute metric on predictions
        result = metric.compute(predictions=predictions, references=references)
        return result

    def save_results(self, report: EvaluationReport, output_path: Union[str, Path]):
        """Save evaluation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def load_results(self, input_path: Union[str, Path]) -> EvaluationReport:
        """Load evaluation results from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)

        return EvaluationReport.from_dict(data)