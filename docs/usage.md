# Usage

The framework provides two evaluation modes:

## Dataset-Only Evaluation

```python
evaluator = ModelEvaluator()
report = evaluator.evaluate(
    dataset="my_dataset",
    metrics=["accuracy"],  # Reference metrics only
    task_type="text-classification"
)
```

## Model-Based Evaluation

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

## Quick Start

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