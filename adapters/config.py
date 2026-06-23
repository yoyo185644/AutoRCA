"""
Dataset Configuration Management

This module provides dataclasses for loading and managing dataset-specific
configuration from YAML files. Each dataset can define:
- Component schema (naming patterns and types)
- Metric groups (organization and sampling)
- Data paths and formats
- Root cause classifications
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
import os


@dataclass
class ComponentSchema:
    """
    Describes a component type within a dataset.

    Attributes:
        id_pattern: Pattern for matching component IDs (e.g., "db_*", "cache_*")
        type: Normalized component type (e.g., "database", "cache", "service")
        examples: List of example component IDs
    """
    id_pattern: str
    type: str
    examples: List[str]


@dataclass
class MetricGroup:
    """
    Groups related metrics within a dataset.

    Attributes:
        name: Group name (e.g., "cpu", "network", "cache")
        patterns: List of metric name patterns (supports * wildcards)
        sampling_interval_s: Sampling interval in seconds
        threshold_percentile: Percentile threshold for anomaly detection (0.0-1.0)
    """
    name: str
    patterns: List[str]
    sampling_interval_s: int
    threshold_percentile: float = 0.95


@dataclass
class DatasetConfig:
    """
    Complete configuration for a dataset.

    This class encapsulates all dataset-specific settings needed by adapters,
    including schema definitions, metric organization, data paths, and
    topology/root cause information.

    Attributes:
        dataset_name: Unique identifier for the dataset (e.g., "Telecom", "Market", "Bank")
        components: List of component schema definitions
        metric_groups: List of metric group definitions
        data_path: Root directory where data files are stored
        data_format: Data file format ("csv", "parquet", etc.)
        default_sampling_interval_s: Default sampling interval in seconds
        has_topology: Whether topology data is available
        topology_source: Source of topology data ("trace", "config", "discovery")
        root_cause_types: List of possible root cause classifications
    """
    dataset_name: str
    components: List[ComponentSchema]
    metric_groups: List[MetricGroup]
    data_path: str
    data_format: str = "csv"
    default_sampling_interval_s: int = 60
    has_topology: bool = False
    topology_source: Optional[str] = None
    root_cause_types: Optional[List[str]] = None

    @staticmethod
    def from_yaml(config_path: str) -> "DatasetConfig":
        """
        Load dataset configuration from YAML file.

        Expected YAML format:
            dataset_name: "DatasetName"
            data_path: "/path/to/data"
            data_format: "csv"
            has_topology: true/false
            topology_source: "trace"

            components:
              - id_pattern: "db_*"
                type: "database"
                examples: ["db_mysql_1", "db_redis_1"]
              - id_pattern: "cache_*"
                type: "cache"
                examples: ["cache_redis_1"]

            metric_groups:
              - name: "cpu"
                patterns: ["cpu_*", "system_cpu"]
                sampling_interval_s: 60
                threshold_percentile: 0.95
              - name: "memory"
                patterns: ["memory_*"]
                sampling_interval_s: 60

        Args:
            config_path: Path to YAML configuration file

        Returns:
            DatasetConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            KeyError: If required fields are missing
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        if cfg is None:
            raise ValueError(f"Config file is empty: {config_path}")

        # Build ComponentSchema list
        components = [
            ComponentSchema(**c) for c in cfg.get('components', [])
        ]

        # Build MetricGroup list
        metric_groups = [
            MetricGroup(**m) for m in cfg.get('metric_groups', [])
        ]

        # Extract and normalize root cause types
        root_cause_types = cfg.get('root_cause_types')
        if root_cause_types is not None and not isinstance(root_cause_types, list):
            root_cause_types = [root_cause_types]

        return DatasetConfig(
            dataset_name=cfg['dataset_name'],
            components=components,
            metric_groups=metric_groups,
            data_path=cfg['data_path'],
            data_format=cfg.get('data_format', 'csv'),
            default_sampling_interval_s=cfg.get('sampling_interval', 60),
            has_topology=cfg.get('has_topology', False),
            topology_source=cfg.get('topology_source'),
            root_cause_types=root_cause_types,
        )

    def get_component_schema(self, component_type: str) -> Optional[ComponentSchema]:
        """
        Get component schema by type.

        Args:
            component_type: Component type to look up

        Returns:
            ComponentSchema if found, None otherwise
        """
        for schema in self.components:
            if schema.type == component_type:
                return schema
        return None

    def get_metric_group(self, metric_name: str) -> Optional[MetricGroup]:
        """
        Find metric group that matches the given metric name.

        Uses simple wildcard matching with * patterns.

        Args:
            metric_name: Name of metric to look up

        Returns:
            MetricGroup if a match is found, None otherwise
        """
        import fnmatch

        for group in self.metric_groups:
            for pattern in group.patterns:
                if fnmatch.fnmatch(metric_name, pattern):
                    return group
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.

        Returns:
            Dictionary representation of config
        """
        return {
            'dataset_name': self.dataset_name,
            'data_path': self.data_path,
            'data_format': self.data_format,
            'default_sampling_interval_s': self.default_sampling_interval_s,
            'has_topology': self.has_topology,
            'topology_source': self.topology_source,
            'root_cause_types': self.root_cause_types,
            'components': [
                {
                    'id_pattern': c.id_pattern,
                    'type': c.type,
                    'examples': c.examples
                }
                for c in self.components
            ],
            'metric_groups': [
                {
                    'name': m.name,
                    'patterns': m.patterns,
                    'sampling_interval_s': m.sampling_interval_s,
                    'threshold_percentile': m.threshold_percentile
                }
                for m in self.metric_groups
            ]
        }
