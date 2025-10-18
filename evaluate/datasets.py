"""Dataset Loader

Utilities for loading and preparing datasets for evaluation.
"""

import logging
from typing import Union, Dict, Any, Optional
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loader for datasets used in evaluation."""

    def __init__(self):
        """Initialize the dataset loader."""
        pass

    def load_dataset(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        split: str = "test",
        **kwargs
    ) -> Union[Dataset, DatasetDict]:
        """Load a dataset from Hugging Face Hub.

        Args:
            dataset_name: Name of the dataset
            config_name: Configuration name (for datasets with multiple configs)
            split: Dataset split to load
            **kwargs: Additional arguments for dataset loading

        Returns:
            Loaded dataset
        """
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split, **kwargs)
            else:
                dataset = load_dataset(dataset_name, split=split, **kwargs)

            logger.info(f"Loaded dataset with {len(dataset)} samples")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

    def prepare_dataset_for_task(
        self,
        dataset: Dataset,
        task_type: str,
        **kwargs
    ) -> Dataset:
        """Prepare dataset for a specific task type.

        Args:
            dataset: Input dataset
            task_type: Type of task (e.g., 'text-classification')
            **kwargs: Additional preparation arguments

        Returns:
            Prepared dataset
        """
        logger.info(f"Preparing dataset for task: {task_type}")

        if task_type == "text-classification":
            # Ensure required columns exist
            required_columns = ["text", "label"]
            if not all(col in dataset.column_names for col in required_columns):
                raise ValueError(f"Dataset must contain columns: {required_columns}")

        elif task_type == "question-answering":
            required_columns = ["question", "context", "answers"]
            if not all(col in dataset.column_names for col in required_columns):
                raise ValueError(f"Dataset must contain columns: {required_columns}")

        # Add any preprocessing here
        return dataset

    def get_dataset_info(self, dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
        """Get information about a dataset.

        Args:
            dataset: Dataset to analyze

        Returns:
            Dictionary with dataset information
        """
        if isinstance(dataset, DatasetDict):
            info = {
                "type": "DatasetDict",
                "splits": list(dataset.keys()),
                "total_samples": sum(len(split) for split in dataset.values()),
            }
        else:
            info = {
                "type": "Dataset",
                "num_samples": len(dataset),
                "features": list(dataset.features.keys()) if hasattr(dataset, 'features') else None,
            }

        return info