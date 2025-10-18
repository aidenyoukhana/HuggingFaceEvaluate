# Contributing

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

## Guidelines

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request