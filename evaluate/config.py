"""Configuration management for the evaluation framework."""

import os
from pathlib import Path
from typing import List, Optional
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass


class Config:
    """Configuration class for the evaluation framework."""

    # Hugging Face settings
    HUGGINGFACE_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN")

    # Default model settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    DEFAULT_DATASET: str = os.getenv("DEFAULT_DATASET", "glue:sst2")

    # Default metrics (comma-separated string to list)
    DEFAULT_METRICS_STR: str = os.getenv("DEFAULT_METRICS", "accuracy,f1")
    DEFAULT_METRICS: List[str] = [m.strip() for m in DEFAULT_METRICS_STR.split(",")]

    # Evaluation settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "8"))
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "512"))
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Logging
    LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_LEVEL: int = getattr(logging, LOG_LEVEL_STR.upper(), logging.INFO)

    # Output settings
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./results")

    @classmethod
    def setup_logging(cls):
        """Setup logging configuration."""
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        """Get full path for output file."""
        output_dir = Path(cls.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename

    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)."""
        print("=== Evaluation Framework Configuration ===")
        print(f"Default Model: {cls.DEFAULT_MODEL}")
        print(f"Default Dataset: {cls.DEFAULT_DATASET}")
        print(f"Default Metrics: {', '.join(cls.DEFAULT_METRICS)}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Max Length: {cls.MAX_LENGTH}")
        print(f"Device: {cls.DEVICE}")
        print(f"Log Level: {cls.LOG_LEVEL_STR}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print(f"HuggingFace Token: {'Set' if cls.HUGGINGFACE_API_TOKEN else 'Not set'}")
        print("=" * 40)


# Initialize logging when config is imported
Config.setup_logging()

# Global config instance
config = Config()