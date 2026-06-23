"""
Bank Dataset Adapter

Example adapter for financial transaction processing system.
Demonstrates a completely different system architecture from Telecom and Market:
- Component types: processor, queue, ledger, settlement
- Different failure modes: transaction deadlock, queue overflow, ledger inconsistency
- Different metrics: transaction rate, settlement lag, balance discrepancy

This adapter shows how the DatasetAdapter pattern handles diverse systems.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import os
import pytz

from rca.adapters.base import DatasetAdapter
from rca.adapters.config import DatasetConfig


class BankAdapter(DatasetAdapter):
    """
    Adapter for Bank (financial transaction processing) dataset.

    The Bank system has fundamentally different characteristics:
    - Component types: processor, queue, ledger, settlement engine
    - Metrics: transaction throughput, settlement lag, balance discrepancy
    - Root causes: deadlock, queue overflow, ledger corruption, cascade
    - Critical: Strong ordering and consistency requirements

    This adapter demonstrates implementing a system with different
    failure modes and metrics from Telecom and Market.
    """

    def load_metric_data(
        self,
        metric_path: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Load Bank metric data from CSV files.

        Expected CSV format:
        - timestamp: unix seconds
        - processor_id: processor identifier (proc_txn_1, proc_settle_1)
        - metric_name: metric name (txn_rate, settlement_lag, etc.)
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
            df = pd.read_csv(metric_path, low_memory=False)

            if df.empty:
                return df

            # Rename Bank-specific columns to standard names
            df = df.rename(columns={
                'processor_id': 'component',
            })

            # Ensure required columns exist
            if 'component' not in df.columns:
                if 'processor' in df.columns:
                    df = df.rename(columns={'processor': 'component'})
                else:
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
            print(f"Error loading Bank metric data: {e}")
            return pd.DataFrame(
                columns=['component', 'metric_name', 'timestamp', 'value']
            )

    def parse_component_id(self, raw_id: str) -> Tuple[str, str]:
        """
        Parse Bank component ID.

        Bank uses its own naming conventions:
        - proc_* -> processor
        - queue_* -> queue
        - ledger_* -> ledger
        - settle_* -> settlement
        - monitor_* -> monitor

        Args:
            raw_id: Raw component identifier

        Returns:
            Tuple of (normalized_id, component_type)
        """
        c = str(raw_id or "").strip()

        if not c:
            return c, "other"

        # Bank-specific component type mapping
        if c.startswith("proc_"):
            return c, "processor"
        if c.startswith("queue_"):
            return c, "queue"
        if c.startswith("ledger_"):
            return c, "ledger"
        if c.startswith("settle_"):
            return c, "settlement"
        if c.startswith("monitor_"):
            return c, "monitor"

        return c, "other"

    def normalize_timestamp(self, ts: Any) -> int:
        """
        Normalize timestamp for Bank dataset.

        Bank uses UTC timezone. This method converts various formats
        to Unix epoch seconds.

        Args:
            ts: Timestamp in any format

        Returns:
            Unix epoch seconds
        """
        if isinstance(ts, pd.Timestamp):
            # Convert to UTC if needed
            if ts.tz is not None:
                return int(ts.tz_convert("UTC").timestamp())
            else:
                return int(ts.timestamp())

        if isinstance(ts, str):
            # Parse ISO 8601 format
            try:
                ts = pd.to_datetime(ts)
                return int(ts.tz_convert("UTC").timestamp() if ts.tz else ts.timestamp())
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
        Extract anomaly segments from Bank metrics.

        Bank-specific anomalies:
        - Transaction rate drops (could indicate deadlock)
        - Settlement lag spikes (could indicate ledger issues)
        - Balance discrepancies (could indicate corruption)

        For Phase 1, uses default percentile implementation.
        Phase 2 can add Bank-specific logic.

        Args:
            metric_df: Metric data
            metric_name: Name of metric
            threshold: Percentile threshold

        Returns:
            List of anomaly segments
        """
        # Bank-specific detection could go here
        # For now, use default
        return super().extract_anomaly_segments(metric_df, metric_name, threshold)

    def extract_topology_edges(self) -> List[Tuple[str, str, str]]:
        """
        Extract Bank system topology.

        Bank topology is highly structured:
        processor -> queue -> processor -> ledger -> settlement

        For Phase 1, returns empty. Phase 2 can implement.

        Returns:
            List of topology edges
        """
        return super().extract_topology_edges()

    def parse_root_cause_label(self, label: str) -> Dict[str, Any]:
        """
        Parse Bank root cause label.

        Bank root causes might be:
        - transaction deadlock
        - queue overflow
        - ledger inconsistency
        - settlement delay
        - cascading failure

        Args:
            label: Root cause label string

        Returns:
            Parsed label dictionary
        """
        return super().parse_root_cause_label(label)
