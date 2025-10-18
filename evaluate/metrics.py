"""Metrics Registry

Registry for managing evaluation metrics.
"""

import logging
from typing import Dict, Any, Optional
import importlib
# Import Hugging Face evaluate library explicitly
hf_evaluate = importlib.import_module('evaluate')

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Registry for evaluation metrics."""

    def __init__(self):
        """Initialize the metric registry."""
        self._metrics: Dict[str, Any] = {}

    def get_metric(self, name: str, **kwargs) -> Any:
        """Get or load a metric by name.

        Args:
            name: Name of the metric
            **kwargs: Additional arguments for metric loading

        Returns:
            Loaded metric module
        """
        if name not in self._metrics:
            try:
                logger.info(f"Loading metric: {name}")
                self._metrics[name] = hf_evaluate.load(name, **kwargs)
            except Exception as e:
                logger.error(f"Failed to load metric {name}: {e}")
                raise

        return self._metrics[name]

    def list_available_metrics(self) -> list:
        """List all available metrics in the registry."""
        return list(self._metrics.keys())

    def add_custom_metric(self, name: str, metric: Any):
        """Add a custom metric to the registry.

        Args:
            name: Name of the metric
            metric: Metric module instance
        """
        self._metrics[name] = metric
        logger.info(f"Added custom metric: {name}")