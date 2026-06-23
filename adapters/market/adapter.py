"""
Market Dataset Adapter

Example adapter for e-commerce market system.
Demonstrates component types: cache, search, database, service, gateway.
Demonstrates metric groups: cache, search, database, throughput.

This adapter shows how to implement a new dataset with different
naming conventions and root cause types from Telecom.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import os
import pytz

from rca.adapters.base import DatasetAdapter
from rca.adapters.config import DatasetConfig


class MarketAdapter(DatasetAdapter):
    """
    Adapter for Market (e-commerce platform) dataset.

    The Market system has different characteristics from Telecom:
    - Different component types (cache, search, database, service, gateway)
    - Different metric organization (cache hits, search latency, QPS)
    - Different root causes (cache misses, slow search, database slowdown)
    - Possibly different data format

    This adapter demonstrates how to implement a new dataset
    while following the DatasetAdapter pattern.
    """

    def load_metric_data(
        self,
        metric_path: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Load Market metric data from CSV files.

        Expected CSV format:
        - timestamp: unix seconds or ISO 8601
        - service: component identifier (cache_redis_1, search_es_1, etc.)
        - metric: metric name (hit_rate, latency, qps, etc.)
        - value: metric value

        Args:
            metric_path: Path to metric CSV file
            start: Start timestamp
            end: End timestamp

        Returns:
            DataFrame with standardized columns
        """
        if not os.path.exists(metric_path):
            return pd.DataFrame(
                columns=['component', 'metric_name', 'timestamp', 'value']
            )

        try:
            # Load CSV - Market might use 'service' instead of 'component'
            df = pd.read_csv(metric_path, low_memory=False)

            if df.empty:
                return df

            # Rename Market-specific columns to standard names
            df = df.rename(columns={
                'service': 'component',
                'metric': 'metric_name'
            })

            # Ensure required columns exist
            if 'component' not in df.columns:
                return pd.DataFrame(
                    columns=['component', 'metric_name', 'timestamp', 'value']
                )

            # Normalize timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].apply(self.normalize_timestamp)
            else:
                return pd.DataFrame(
                    columns=['component', 'metric_name', 'timestamp', 'value']
                )

            # Filter by time range
            start_s = int(start.timestamp())
            end_s = int(end.timestamp())
            df = df[(df['timestamp'] >= start_s) & (df['timestamp'] <= end_s)].copy()

            # Return standardized columns
            return df[['component', 'metric_name', 'timestamp', 'value']]

        except Exception as e:
            print(f"Error loading Market metric data: {e}")
            return pd.DataFrame(
                columns=['component', 'metric_name', 'timestamp', 'value']
            )

    def parse_component_id(self, raw_id: str) -> Tuple[str, str]:
        """
        Parse Market component ID.

        Market uses different naming conventions from Telecom:
        - cache_*, redis_*, memcache_* -> cache
        - search_*, es_*, elasticsearch_* -> search
        - db_* -> database
        - svc_* -> service
        - gateway_* -> gateway

        Args:
            raw_id: Raw component identifier

        Returns:
            Tuple of (normalized_id, component_type)
        """
        c = str(raw_id or "").strip()

        if not c:
            return c, "other"

        # Market-specific component type mapping
        if c.startswith(("cache_", "redis_", "memcache_")):
            return c, "cache"
        if c.startswith(("search_", "es_", "elasticsearch_")):
            return c, "search"
        if c.startswith("db_"):
            return c, "database"
        if c.startswith("svc_"):
            return c, "service"
        if c.startswith("gateway_"):
            return c, "gateway"

        # Default for Market is service type
        return c, "service"

    def normalize_timestamp(self, ts: Any) -> int:
        """
        Normalize timestamp for Market dataset.

        Market might use Asia/Shanghai timezone (UTC+8).
        This method handles various input formats and converts to Unix seconds.

        Args:
            ts: Timestamp in any format

        Returns:
            Unix epoch seconds
        """
        if isinstance(ts, pd.Timestamp):
            # If timezone-aware, convert to UTC
            if ts.tz is not None:
                return int(ts.tz_convert("UTC").timestamp())
            else:
                # Assume UTC+8 (Asia/Shanghai) if no timezone info
                try:
                    ts = ts.tz_localize("Asia/Shanghai")
                    return int(ts.tz_convert("UTC").timestamp())
                except Exception:
                    return int(ts.timestamp())

        if isinstance(ts, str):
            # Parse ISO 8601 format
            try:
                ts = pd.to_datetime(ts)
                if ts.tz is not None:
                    return int(ts.tz_convert("UTC").timestamp())
                else:
                    # Assume UTC+8
                    ts = ts.tz_localize("Asia/Shanghai")
                    return int(ts.tz_convert("UTC").timestamp())
            except Exception:
                pass

        # Fall back to parent class implementation
        return super().normalize_timestamp(ts)

    def extract_anomaly_segments(
        self,
        metric_df: pd.DataFrame,
        metric_name: str,
        threshold: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """
        Extract anomaly segments from Market metrics.

        Market-specific anomaly detection:
        - Cache metrics: lower is better (hit rate, etc.)
        - Search metrics: higher latency is anomalous
        - QPS metrics: both too high and too low can be anomalous

        For Phase 1, we use the default percentile implementation.
        Phase 2 can add Market-specific logic.

        Args:
            metric_df: Metric data
            metric_name: Name of metric
            threshold: Percentile threshold

        Returns:
            List of anomaly segments
        """
        # Market-specific detection could go here
        # For now, use default
        return super().extract_anomaly_segments(metric_df, metric_name, threshold)

    def extract_topology_edges(self) -> List[Tuple[str, str, str]]:
        """
        Extract Market system topology.

        Market topology might be:
        gateway -> service -> cache -> database
        gateway -> service -> search -> database

        For Phase 1, returns empty. Phase 2 can implement.

        Returns:
            List of topology edges
        """
        return super().extract_topology_edges()

    def parse_root_cause_label(self, label: str) -> Dict[str, Any]:
        """
        Parse Market root cause label.

        Market root causes might be:
        - cache hit rate drop
        - search latency spike
        - database slow query
        - service timeout
        - QPS drop

        Args:
            label: Root cause label string

        Returns:
            Parsed label dictionary
        """
        return super().parse_root_cause_label(label)
