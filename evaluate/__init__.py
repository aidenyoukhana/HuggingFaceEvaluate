"""Hugging Face Evaluate Framework

A Python framework for evaluating machine learning models using Hugging Face Evaluate.
"""

from .evaluator import ModelEvaluator
from .metrics import MetricRegistry
from .datasets import DatasetLoader
from .reports import EvaluationReport
from .config import Config, config

__version__ = "0.1.0"
__all__ = ["ModelEvaluator", "MetricRegistry", "DatasetLoader", "EvaluationReport", "Config", "config"]