"""
Dataset Adapter Registry and Factory

This module provides the registry pattern for dataset adapters, eliminating
the need for if/elif branching when selecting adapters.

Instead of:
    if dataset == "Telecom":
        adapter = TelecomAdapter()
    elif dataset == "Market":
        adapter = MarketAdapter()
    ...

Use:
    adapter = DatasetRegistry.get_adapter("Telecom")

This makes the system more extensible and cleaner.
"""

from typing import Dict, Type, Optional
import os

from rca.adapters.base import DatasetAdapter
from rca.adapters.config import DatasetConfig


class DatasetRegistry:
    """
    Registry for dataset adapters using factory pattern.

    This class maintains mappings from dataset names to adapter classes
    and their configurations. It provides methods to register new datasets
    and retrieve adapter instances without conditional branching.

    Usage:
        # Register a new adapter
        DatasetRegistry.register(
            "Market",
            MarketAdapter,
            DatasetConfig.from_yaml("market_config.yaml")
        )

        # Get adapter instance
        adapter = DatasetRegistry.get_adapter("Market")

        # Load adapter config
        config = DatasetRegistry.load_config("Market", "market_config.yaml")
    """

    # Class-level dictionaries storing adapter classes and configurations
    _adapters: Dict[str, Type[DatasetAdapter]] = {}
    _configs: Dict[str, DatasetConfig] = {}

    @classmethod
    def register(
        cls,
        dataset_name: str,
        adapter_class: Type[DatasetAdapter],
        config: DatasetConfig,
    ) -> None:
        """
        Register a new dataset adapter.

        Args:
            dataset_name: Unique identifier for the dataset (e.g., "Telecom")
            adapter_class: Adapter class (must be subclass of DatasetAdapter)
            config: DatasetConfig instance for this adapter

        Raises:
            TypeError: If adapter_class is not a DatasetAdapter subclass
            ValueError: If dataset_name is already registered
        """
        if not issubclass(adapter_class, DatasetAdapter):
            raise TypeError(
                f"{adapter_class.__name__} must be a DatasetAdapter subclass"
            )

        if dataset_name in cls._adapters:
            raise ValueError(f"Dataset '{dataset_name}' is already registered")

        cls._adapters[dataset_name] = adapter_class
        cls._configs[dataset_name] = config

    @classmethod
    def get_adapter(cls, dataset_name: str) -> DatasetAdapter:
        """
        Get an adapter instance for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DatasetAdapter instance

        Raises:
            ValueError: If dataset is not registered
        """
        if dataset_name not in cls._adapters:
            raise ValueError(
                f"No adapter registered for dataset '{dataset_name}'. "
                f"Available: {', '.join(cls._adapters.keys())}"
            )

        adapter_class = cls._adapters[dataset_name]
        config = cls._configs[dataset_name]
        return adapter_class(config)

    @classmethod
    def load_config(
        cls,
        dataset_name: str,
        config_path: str,
    ) -> DatasetConfig:
        """
        Load configuration for a dataset from YAML file.

        Args:
            dataset_name: Name of the dataset
            config_path: Path to YAML config file

        Returns:
            DatasetConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = DatasetConfig.from_yaml(config_path)

        # Verify dataset name matches
        if config.dataset_name != dataset_name:
            raise ValueError(
                f"Config dataset_name '{config.dataset_name}' "
                f"does not match expected '{dataset_name}'"
            )

        return config

    @classmethod
    def get_config(cls, dataset_name: str) -> DatasetConfig:
        """
        Get configuration for a registered dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DatasetConfig instance

        Raises:
            ValueError: If dataset is not registered
        """
        if dataset_name not in cls._configs:
            raise ValueError(
                f"No config registered for dataset '{dataset_name}'. "
                f"Available: {', '.join(cls._configs.keys())}"
            )

        return cls._configs[dataset_name]

    @classmethod
    def is_registered(cls, dataset_name: str) -> bool:
        """
        Check if a dataset is registered.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if dataset is registered
        """
        return dataset_name in cls._adapters

    @classmethod
    def list_datasets(cls) -> Dict[str, str]:
        """
        List all registered datasets with their adapter classes.

        Returns:
            Dictionary mapping dataset names to adapter class names
        """
        return {
            name: cls._adapters[name].__name__
            for name in cls._adapters.keys()
        }

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registrations (useful for testing).

        WARNING: This removes all registered adapters and configs.
        """
        cls._adapters.clear()
        cls._configs.clear()

    @classmethod
    def _init_default_adapters(cls) -> None:
        """
        Initialize default adapters at module load time.

        This is called automatically when the registry is first used.
        It registers adapters for Telecom, Market, and Bank datasets.
        """
        # This will be populated as concrete adapters are created
        # See adapters/telecom/adapter.py, adapters/market/adapter.py, etc.
        pass
