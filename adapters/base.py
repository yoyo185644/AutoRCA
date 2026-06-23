"""
Dataset Adapter Base Class

This module provides the abstract base class for all dataset-specific adapters.
Each adapter implements dataset-specific logic for:
1. Loading metric data
2. Parsing component identifiers
3. Normalizing timestamps
4. Extracting anomaly segments
5. Extracting topology relationships
6. Parsing root cause labels

Key principle: All dataset-specific logic is encapsulated in adapters,
keeping core RCA logic clean and maintainable.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import fnmatch

from rca.adapters.config import DatasetConfig, MetricGroup


class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.

    Each dataset (Telecom, Market, Bank, etc.) must implement a concrete
    subclass of this adapter. The adapter is responsible for translating
    dataset-specific formats, schemas, and naming conventions into a
    unified format expected by the RCA engine.

    Design principles:
    - All dataset-specific logic goes in the adapter
    - Core RCA logic remains unchanged
    - Adapters are composable and testable independently
    - Configuration is externalized to YAML
    """

    def __init__(self, config: DatasetConfig):
        """
        Initialize adapter with configuration.

        Args:
            config: DatasetConfig instance for this adapter
        """
        self.config = config
        self.dataset_name = config.dataset_name

    # ============ Abstract Methods - Must Implement ============

    @abstractmethod
    def load_metric_data(
        self,
        metric_path: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Load metric data for a given time range.

        This method should load data from the specified metric file and
        filter it to the given time range. The returned DataFrame must
        have the following columns:
        - component: Component identifier (string)
        - metric_name: Name of the metric (string)
        - timestamp: Unix epoch seconds (int)
        - value: Metric value (float)

        Args:
            metric_path: Path to metric data file
            start: Start of time window (pandas Timestamp)
            end: End of time window (pandas Timestamp)

        Returns:
            DataFrame with columns: component, metric_name, timestamp, value
        """
        pass

    @abstractmethod
    def parse_component_id(self, raw_id: str) -> Tuple[str, str]:
        """
        Parse component identifier and extract its type.

        This method should identify the component type based on the ID format.
        Different datasets use different naming conventions:
        - Telecom: "db_mysql_1" -> ("db_mysql_1", "database")
        - Market: "cache_redis_1" -> ("cache_redis_1", "cache")
        - Bank: "queue_txn_1" -> ("queue_txn_1", "queue")

        Args:
            raw_id: Raw component identifier from data

        Returns:
            Tuple of (normalized_id, component_type)
        """
        pass

    # ============ Optional Methods - Have Defaults ============

    def normalize_timestamp(self, ts: Any) -> int:
        """
        Convert timestamp to Unix epoch seconds.

        Default implementation handles:
        - pandas.Timestamp objects
        - Integer timestamps (seconds or milliseconds)

        Override this method if your dataset uses a different timestamp format.

        Args:
            ts: Timestamp in any format

        Returns:
            Unix epoch seconds (integer)

        Raises:
            ValueError: If timestamp format is unsupported
        """
        if isinstance(ts, pd.Timestamp):
            return int(ts.timestamp())

        if isinstance(ts, int):
            # Heuristic: if > 10^11, assume milliseconds
            if ts > 10**11:
                return ts // 1000
            return ts

        if isinstance(ts, float):
            return int(ts)

        raise ValueError(f"Unsupported timestamp format: {type(ts).__name__}")

    def extract_anomaly_segments(
        self,
        metric_df: pd.DataFrame,
        metric_name: str,
        threshold: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """
        Extract anomaly segments from metric time series.

        Default implementation uses percentile-based thresholding:
        1. Compute baseline as threshold percentile (default 95th)
        2. Identify all breaches above baseline
        3. Group consecutive breaches into segments
        4. Calculate severity for each segment

        Override this method for more sophisticated anomaly detection.

        Args:
            metric_df: DataFrame with columns: component, metric_name, timestamp, value
            metric_name: Name of the metric being analyzed
            threshold: Percentile threshold (0.0-1.0) for anomaly detection

        Returns:
            List of anomaly segments, each containing:
            - component: Component ID
            - metric_name: Metric name
            - start_time: Unix seconds
            - end_time: Unix seconds
            - severity: Normalized severity (0.0-1.0)
            - num_breaches: Number of breached data points
        """
        segments = []

        if metric_df.empty:
            return segments

        # Calculate baseline
        baseline_value = metric_df['value'].quantile(threshold)

        # Find all breach points (above baseline)
        breaches = metric_df[metric_df['value'] > baseline_value].copy()
        if breaches.empty:
            return segments

        # Group consecutive breaches by time (gap > 300 seconds = new group)
        breaches = breaches.sort_values('timestamp')
        breaches['ts_group'] = (
            breaches['timestamp'].diff() > 300
        ).cumsum()

        # Process each group
        for group_id, group_data in breaches.groupby('ts_group'):
            max_value = group_data['value'].max()
            severity = (max_value - baseline_value) / max(baseline_value, 1e-6)

            segments.append({
                'component': group_data['component'].iloc[0],
                'metric_name': metric_name,
                'start_time': int(group_data['timestamp'].min()),
                'end_time': int(group_data['timestamp'].max()),
                'severity': min(1.0, severity),
                'num_breaches': len(group_data),
            })

        return segments

    def extract_topology_edges(self) -> List[Tuple[str, str, str]]:
        """
        Extract topology relationships between components.

        Default implementation returns empty list (no topology data).
        Override this method to load topology from:
        - Configuration files
        - Trace data
        - Service discovery APIs
        - System configuration

        Returns:
            List of edges as (source_component, target_component, relation_type) tuples
        """
        if not self.config.has_topology:
            return []

        # Subclasses can override to load from various sources
        return []

    def parse_root_cause_label(self, label: str) -> Dict[str, Any]:
        """
        Parse root cause label from annotation data.

        Default implementation assumes format: "component:reason_type:detail"

        Override this method if your dataset uses different label formats.

        Args:
            label: Root cause label string

        Returns:
            Dictionary with keys:
            - component: Component identifier
            - reason_type: Classification of root cause
            - reason_detail: Additional details (optional)
        """
        parts = label.split(':')
        return {
            'component': parts[0],
            'reason_type': parts[1] if len(parts) > 1 else 'unknown',
            'reason_detail': ':'.join(parts[2:]) if len(parts) > 2 else '',
        }

    # ============ Helper Methods ============

    def get_component_type(self, component_id: str) -> str:
        """
        Get the type of a component.

        Args:
            component_id: Component identifier

        Returns:
            Component type string
        """
        _, comp_type = self.parse_component_id(component_id)
        return comp_type

    def get_metric_group(self, metric_name: str) -> Optional[MetricGroup]:
        """
        Find the metric group for a given metric.

        Args:
            metric_name: Name of the metric

        Returns:
            MetricGroup if found, None otherwise
        """
        if not self.config.metric_groups:
            return None

        for group in self.config.metric_groups:
            for pattern in group.patterns:
                if self._pattern_match(metric_name, pattern):
                    return group
        return None

    @staticmethod
    def _pattern_match(text: str, pattern: str) -> bool:
        """
        Simple pattern matching with * wildcards.

        Args:
            text: Text to match
            pattern: Pattern with * wildcards

        Returns:
            True if text matches pattern
        """
        return fnmatch.fnmatch(text, pattern)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert adapter to dictionary representation.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            'dataset_name': self.dataset_name,
            'adapter_class': self.__class__.__name__,
            'config': self.config.to_dict()
        }
