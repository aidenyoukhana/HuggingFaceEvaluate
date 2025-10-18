"""Tests for the DatasetLoader component."""

import pytest
from unittest.mock import Mock, patch
from evaluate import DatasetLoader


class TestDatasetLoader:
    """Test cases for DatasetLoader."""

    @patch('evaluate.datasets.load_dataset')
    def test_load_dataset(self, mock_load):
        """Test loading a dataset."""
        loader = DatasetLoader()
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load.return_value = mock_dataset

        dataset = loader.load_dataset("test_dataset")

        mock_load.assert_called_once_with("test_dataset", split="test")
        assert dataset == mock_dataset