# Evaluate Framework

A Python framework for evaluating machine learning models using Hugging Face Evaluate, inspired by Azure AI Foundry evaluation patterns.

## Features

- **Unified Evaluation Interface**: Simple API for evaluating models across different tasks
- **Metric Registry**: Centralized management of evaluation metrics
- **Dataset Utilities**: Easy loading and preparation of datasets
- **Comprehensive Reporting**: Structured evaluation reports with summaries
- **Extensible Design**: Add custom metrics and evaluation pipelines

## Installation

### From Source

```bash
git clone <your-repo-url>
cd evaluate-framework
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Configuration

The framework uses environment variables for configuration. Copy `.env.example` to `.env` and customize the values:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### Environment Variables

- `HUGGINGFACE_API_TOKEN`: API token for private models
- `DEFAULT_MODEL`: Default model for evaluation
- `DEFAULT_DATASET`: Default dataset for evaluation
- `DEFAULT_METRICS`: Default metrics (comma-separated)
- `BATCH_SIZE`: Batch size for evaluation
- `DEVICE`: Device to run on (auto, cpu, cuda)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `OUTPUT_DIR`: Directory for saving results

## How It Works

The framework provides two evaluation modes:

### 1. **Dataset-Only Evaluation** (No Model Required)
```python
evaluator = ModelEvaluator()
report = evaluator.evaluate(
    dataset="my_dataset",
    metrics=["accuracy"],  # Reference metrics only
    task_type="text-classification"
)
```
Useful for computing dataset statistics or reference metrics that don't require model predictions.

### 2. **Model-Based Evaluation** (With Your LLM)
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your model
model = AutoModelForSequenceClassification.from_pretrained("your-model-name")
tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# Evaluate with the framework
evaluator = ModelEvaluator(model=model, tokenizer=tokenizer)
report = evaluator.evaluate(
    dataset="glue:sst2",  # Dataset with ground truth labels
    metrics=["accuracy", "f1", "precision", "recall"],
    task_type="text-classification"
)
```

**What happens:**
1. **Loads your model** and runs inference on the dataset
2. **Generates predictions** for each example
3. **Compares predictions** against ground truth labels
4. **Computes metrics** (accuracy, F1, etc.) on the results
5. **Returns structured report** with all results

## Quick Start

First, set up your configuration:

```bash
cp .env.example .env
# Edit .env with your model and dataset preferences
```

Then use the framework:

```python
from evaluate import ModelEvaluator, config

# Print current configuration
config.print_config()

# Use config defaults
evaluator = ModelEvaluator()
report = evaluator.evaluate()  # Uses all config defaults

# Or specify custom values
report = evaluator.evaluate(
    dataset="your-custom-dataset",
    metrics=["accuracy", "f1", "precision"]
)
```

For model-based evaluation:

```python
from evaluate import ModelEvaluator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your model
model = AutoModelForSequenceClassification.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

evaluator = ModelEvaluator(model=model, tokenizer=tokenizer)
report = evaluator.evaluate()  # Uses config defaults
```

## Project Structure

```
evaluate/
├── __init__.py       # Package initialization
├── evaluator.py      # Main evaluation logic
├── metrics.py        # Metric management
├── datasets.py       # Dataset utilities
└── reports.py        # Result reporting
```

## Usage Examples

See the integration tests in `tests/test_integration.py` for detailed usage examples:

- `TestIntegration.test_basic_evaluation_example()`: Basic evaluation workflow (dataset-only)
- `TestIntegration.test_real_model_evaluation_example()`: Evaluating an actual LLM with real predictions

## Development

### Running Tests

```bash
# Run unit tests (fast, no external dependencies)
pytest tests/test_metrics.py tests/test_datasets.py tests/test_reports.py tests/test_evaluator.py

# Run integration tests (require network access)
pytest tests/test_integration.py -v

# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_metrics.py::TestMetricRegistry::test_get_metric -v
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 evaluate/
mypy evaluate/
```

### Building Documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```

## API Reference

### ModelEvaluator

Main class for running evaluations.

```python
class ModelEvaluator:
    def evaluate(self, dataset, metrics, task_type="text-classification", **kwargs):
        """Run evaluation and return results."""
```

### MetricRegistry

Manages evaluation metrics.

```python
class MetricRegistry:
    def get_metric(self, name):
        """Load and return a metric by name."""
```

### DatasetLoader

Handles dataset loading and preparation.

```python
class DatasetLoader:
    def load_dataset(self, name, **kwargs):
        """Load a dataset from Hugging Face Hub."""
```

### EvaluationReport

Container for evaluation results.

```python
class EvaluationReport:
    def print_summary(self):
        """Print human-readable results."""
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.