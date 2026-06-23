"""
Dataset Adapter Framework

This module provides a unified interface for loading and processing data
from multiple different RCA systems (Telecom, Market, Bank, etc.)

Core classes:
- DatasetAdapter: Base class for dataset-specific adapters
- DatasetRegistry: Factory and registry for adapter instances
- DatasetConfig: Configuration loading from YAML
"""

from rca.adapters.config import DatasetConfig, ComponentSchema, MetricGroup
from rca.adapters.base import DatasetAdapter
from rca.adapters.registry import DatasetRegistry

__all__ = [
    "DatasetAdapter",
    "DatasetRegistry",
    "DatasetConfig",
    "ComponentSchema",
    "MetricGroup",
]
