"""Evaluation Reports

Classes for storing and managing evaluation results.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class EvaluationReport:
    """Container for evaluation results."""

    task_type: str
    metrics: List[str]
    results: Dict[str, Any]
    dataset_info: Dict[str, Any]
    timestamp: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationReport":
        """Create report from dictionary."""
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "EvaluationReport":
        """Load report from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        summary = {
            "task_type": self.task_type,
            "timestamp": self.timestamp,
            "num_metrics": len(self.metrics),
            "metrics_computed": self.metrics,
        }

        # Add key results
        key_results = {}
        for metric_name, result in self.results.items():
            if isinstance(result, dict) and "error" not in result:
                # Extract main metric values
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        key_results[f"{metric_name}_{key}"] = value

        summary["key_results"] = key_results
        return summary

    def print_summary(self):
        """Print a human-readable summary."""
        print(f"Evaluation Report - {self.task_type}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Metrics: {', '.join(self.metrics)}")
        print("\nResults:")
        for metric_name, result in self.results.items():
            print(f"  {metric_name}:")
            if isinstance(result, dict):
                if "error" in result:
                    print(f"    Error: {result['error']}")
                else:
                    for key, value in result.items():
                        print(f"    {key}: {value}")
            else:
                print(f"    {result}")