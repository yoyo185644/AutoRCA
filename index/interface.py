"""
Public Index Interface

This module defines the abstract interface for querying RCA results
across datasets, and provides a simple implementation.

All index implementations must return results as UnifiedCandidate lists
to ensure compatibility with downstream query/ranking layers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import glob

from rca.adapters.registry import DatasetRegistry
from rca.index.candidate import UnifiedCandidate


class PublicIndexInterface(ABC):
    """
    Abstract interface for RCA index implementations.

    An RCA index is responsible for:
    1. Building index from raw metric data
    2. Loading pre-built indexes
    3. Querying the index to return candidate root causes

    All implementations must return results as UnifiedCandidate lists.
    """

    @abstractmethod
    def build_index(
        self,
        dataset: str,
        start_time: int,
        end_time: int,
        metric_dir: str,
    ) -> None:
        """
        Build index from raw metric data for a time window.

        Args:
            dataset: Dataset name (e.g., "Telecom")
            start_time: Start time (unix seconds)
            end_time: End time (unix seconds)
            metric_dir: Directory containing metric files
        """
        pass

    @abstractmethod
    def load_index(
        self,
        dataset: str,
        index_dir: str,
    ) -> None:
        """
        Load pre-built index from disk.

        Args:
            dataset: Dataset name
            index_dir: Directory containing pre-built index files
        """
        pass

    @abstractmethod
    def query(
        self,
        dataset: str,
        instruction: str,
        top_k: int = 10,
    ) -> List[UnifiedCandidate]:
        """
        Query the index for root cause candidates.

        Args:
            dataset: Dataset name
            instruction: RCA instruction/query
            top_k: Return top K results

        Returns:
            List of UnifiedCandidate results sorted by score (highest first)
        """
        pass


class SimpleIndexImpl(PublicIndexInterface):
    """
    Simple index implementation that loads CSV files using adapters.

    This implementation:
    1. Uses DatasetRegistry to get appropriate adapter
    2. Loads metric CSV files using the adapter
    3. Computes anomaly segments
    4. Returns UnifiedCandidate results

    This is a basic implementation suitable for Phase 1.
    Phase 2+ can implement more sophisticated indexing.
    """

    def __init__(self):
        """Initialize simple index."""
        self.data: Dict[str, Dict[str, Any]] = {}

    def build_index(
        self,
        dataset: str,
        start_time: int,
        end_time: int,
        metric_dir: str,
    ) -> None:
        """
        Build index by loading metric CSVs and computing anomalies.

        Args:
            dataset: Dataset name
            start_time: Start time (unix seconds)
            end_time: End time (unix seconds)
            metric_dir: Directory containing metric files
        """
        # Get adapter for this dataset
        try:
            adapter = DatasetRegistry.get_adapter(dataset)
        except ValueError:
            print(f"Error: Dataset '{dataset}' not registered")
            return

        # Convert timestamps to pandas
        start_ts = pd.Timestamp(start_time, unit='s')
        end_ts = pd.Timestamp(end_time, unit='s')

        # Find all CSV files in metric directory
        if not os.path.exists(metric_dir):
            print(f"Error: Metric directory not found: {metric_dir}")
            return

        csv_files = glob.glob(os.path.join(metric_dir, "*.csv"))

        all_anomalies = []

        # Process each metric file
        for csv_file in csv_files:
            try:
                # Load metric data using adapter
                df = adapter.load_metric_data(csv_file, start_ts, end_ts)

                if df.empty:
                    continue

                # Extract anomaly segments
                metric_name = os.path.basename(csv_file).replace('.csv', '')

                for component in df['component'].unique():
                    component_df = df[df['component'] == component]

                    # Extract anomalies for this component
                    segments = adapter.extract_anomaly_segments(
                        component_df,
                        metric_name,
                        threshold=0.95
                    )

                    all_anomalies.extend(segments)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

        # Store processed data
        self.data[dataset] = {
            'anomalies': all_anomalies,
            'start_time': start_time,
            'end_time': end_time,
        }

        print(f"Built index for {dataset}: {len(all_anomalies)} anomalies detected")

    def load_index(
        self,
        dataset: str,
        index_dir: str,
    ) -> None:
        """
        Load pre-built index from disk.

        For Phase 1, this is a placeholder.
        Phase 2 can implement persistent storage.

        Args:
            dataset: Dataset name
            index_dir: Directory containing pre-built index
        """
        print(f"Loading index for {dataset} from {index_dir}")
        # Phase 2: Implement index file loading
        pass

    def query(
        self,
        dataset: str,
        instruction: str,
        top_k: int = 10,
    ) -> List[UnifiedCandidate]:
        """
        Query index for root cause candidates.

        Simple implementation: returns top-scoring anomaly components
        as candidates.

        Args:
            dataset: Dataset name
            instruction: Query instruction (not used in simple implementation)
            top_k: Return top K results

        Returns:
            List of UnifiedCandidate sorted by score
        """
        if dataset not in self.data:
            print(f"Error: No index loaded for dataset '{dataset}'")
            return []

        try:
            adapter = DatasetRegistry.get_adapter(dataset)
        except ValueError:
            print(f"Error: Dataset '{dataset}' not registered")
            return []

        anomalies = self.data[dataset]['anomalies']

        # Convert anomalies to UnifiedCandidate format
        candidates = []

        for anomaly in anomalies:
            # Calculate score based on severity and number of breaches
            severity = anomaly.get('severity', 0.5)
            num_breaches = anomaly.get('num_breaches', 1)

            # Heuristic: score = severity * log(num_breaches)
            score = min(1.0, severity * (1.0 + 0.1 * (num_breaches - 1)))

            component = anomaly['component']
            component_type = adapter.get_component_type(component)

            candidate = UnifiedCandidate(
                dataset=dataset,
                component=component,
                component_type=component_type,
                score=score,
                anomaly_start_time=anomaly['start_time'],
                anomaly_end_time=anomaly['end_time'],
                peak_time=anomaly['end_time'],  # Approximate peak
                evidence={
                    'metric_name': anomaly['metric_name'],
                    'severity': anomaly['severity'],
                    'num_breaches': anomaly['num_breaches'],
                },
            )

            candidates.append(candidate)

        # Sort by score (highest first) and return top K
        candidates.sort(key=lambda c: c.score, reverse=True)

        return candidates[:top_k]

    def get_all_candidates(
        self,
        dataset: str,
    ) -> List[UnifiedCandidate]:
        """
        Get all candidates without limit.

        Args:
            dataset: Dataset name

        Returns:
            All UnifiedCandidate results
        """
        return self.query(dataset, "", top_k=1000000)
