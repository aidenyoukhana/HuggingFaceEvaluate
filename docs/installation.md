# Installation

## From Source

```bash
git clone <your-repo-url>
cd evaluate-framework
pip install -e .
```

## Development Installation

```bash
pip install -e ".[dev]"
```

## Configuration

The framework uses environment variables for configuration. Copy `.env.example` to `.env` and customize the values:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Environment Variables

- `HUGGINGFACE_API_TOKEN`: API token for private models
- `DEFAULT_MODEL`: Default model for evaluation
- `DEFAULT_DATASET`: Default dataset for evaluation
- `DEFAULT_METRICS`: Default metrics (comma-separated)
- `BATCH_SIZE`: Batch size for evaluation
- `DEVICE`: Device to run on (auto, cpu, cuda)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `OUTPUT_DIR`: Directory for saving results