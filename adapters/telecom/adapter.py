"""
Telecom Dataset Adapter

This adapter wraps the existing Telecom RCA logic into the DatasetAdapter interface.
It does NOT modify any existing Telecom code - it only calls existing functions
and translates the results into the unified adapter format.

Key principle: TelecomAdapter is a WRAPPER, not a REWRITE.
Existing functions from field_solvers, component_candidates, etc. remain unchanged.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import os

from rca.adapters.base import DatasetAdapter
from rca.adapters.config import DatasetConfig


class TelecomAdapter(DatasetAdapter):
    """
    Adapter for Telecom dataset.

    This adapter wraps existing Telecom RCA logic without modification.
    It translates between Telecom-specific data formats and the unified
    DatasetAdapter interface.

    Wraps functions from:
    - rca.execution.field_solvers._load_metric_csv
    - rca.execution.field_solvers._to_tz_timestamp_ms
    - rca.component_candidates._comp_type
    """

    def load_metric_data(
        self,
        metric_path: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Load Telecom metric data using existing Telecom CSV loader.

        This calls the existing _load_metric_csv function from field_solvers,
        which handles Telecom-specific CSV format and timezone conversion.

        Args:
            metric_path: Path to metric CSV file
            start: Start timestamp
            end: End timestamp

        Returns:
            DataFrame with columns: component, metric_name, timestamp, value
        """
        try:
            # Import existing Telecom CSV loader
            from rca.execution.field_solvers import _load_metric_csv, _to_tz_timestamp_ms
        except ImportError:
            # Fallback if imports not available
            return pd.DataFrame(
                columns=['component', 'metric_name', 'timestamp', 'value']
            )

        # Check if file exists
        if not os.path.exists(metric_path):
            return pd.DataFrame(
                columns=['component', 'metric_name', 'timestamp', 'value']
            )

        try:
            # Use existing Telecom loader
            df = _load_metric_csv(metric_path)

            if df.empty:
                return df

            # Normalize timestamp column if present
            if "timestamp" in df.columns:
                df = df.copy()
                try:
                    df["timestamp"] = df["timestamp"].apply(_to_tz_timestamp_ms)
                except Exception:
                    # If conversion fails, try direct normalization
                    df["timestamp"] = df["timestamp"].apply(self.normalize_timestamp)

            # Filter by time range
            start_s = int(start.timestamp())
            end_s = int(end.timestamp())
            df = df[(df["timestamp"] >= start_s) & (df["timestamp"] <= end_s)].copy()

            # Ensure correct column order
            if "component" not in df.columns:
                # Try to infer from 'service' or other column names
                if "service" in df.columns:
                    df = df.rename(columns={"service": "component"})

            if "metric_name" not in df.columns:
                if "metric" in df.columns:
                    df = df.rename(columns={"metric": "metric_name"})

            return df[[col for col in ['component', 'metric_name', 'timestamp', 'value']
                      if col in df.columns]]

        except Exception as e:
            # Log but don't crash - return empty DataFrame
            print(f"Error loading Telecom metric data: {e}")
            return pd.DataFrame(
                columns=['component', 'metric_name', 'timestamp', 'value']
            )

    def parse_component_id(self, raw_id: str) -> Tuple[str, str]:
        """
        Parse Telecom component ID using existing Telecom type classification.

        This implements Telecom's component type mapping:
        - db_* -> database
        - os_*, node-* -> node
        - docker_*, container* -> container
        - Other with "-" -> service
        - Default -> other

        Args:
            raw_id: Raw component identifier

        Returns:
            Tuple of (normalized_id, component_type)
        """
        c = str(raw_id or "").strip()

        if not c:
            return c, "other"

        # Telecom-specific component type mapping
        if c.startswith("db_"):
            return c, "database"
        if c.startswith(("os_", "node-")):
            return c, "node"
        if c.startswith(("docker_", "container")):
            return c, "container"
        if "-" in c:
            return c, "service"

        return c, "other"

    def normalize_timestamp(self, ts: Any) -> int:
        """
        Normalize timestamp using Telecom timezone handling.

        Telecom uses UTC+8 timezone. This method converts to Unix epoch seconds.

        Args:
            ts: Timestamp in any format

        Returns:
            Unix epoch seconds
        """
        if isinstance(ts, pd.Timestamp):
            # If timezone-aware, convert to UTC first
            if ts.tz is not None:
                return int(ts.tz_convert("UTC").timestamp())
            else:
                return int(ts.timestamp())

        # Fall back to parent class implementation
        return super().normalize_timestamp(ts)

    def extract_anomaly_segments(
        self,
        metric_df: pd.DataFrame,
        metric_name: str,
        threshold: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """
        Extract anomaly segments using Telecom's breach detection logic.

        Telecom has existing breach detection in _breach_stats_with_baseline.
        For Phase 1, we use the default percentile-based implementation.
        This can be replaced with existing Telecom logic when integrated.

        Args:
            metric_df: Metric data
            metric_name: Name of metric
            threshold: Percentile threshold

        Returns:
            List of anomaly segments
        """
        # For Phase 1, use default implementation
        # Phase 2 can integrate with existing Telecom breach detection
        return super().extract_anomaly_segments(metric_df, metric_name, threshold)

    def extract_topology_edges(self) -> List[Tuple[str, str, str]]:
        """
        Extract Telecom topology edges.

        Telecom has network topology that could be loaded from:
        - System configuration
        - Trace data
        - Service discovery

        For Phase 1, returns empty list. Can be extended in Phase 2.

        Returns:
            List of topology edges
        """
        # Telecom topology loading would go here
        # For now, use default (no topology)
        return super().extract_topology_edges()

    def parse_root_cause_label(self, label: str) -> Dict[str, Any]:
        """
        Parse Telecom root cause label.

        Telecom root causes are classified as:
        - network loss
        - network delay
        - db connection limit
        - db close
        - CPU fault

        Args:
            label: Root cause label string

        Returns:
            Parsed label dictionary
        """
        return super().parse_root_cause_label(label)
