"""
Unified Candidate Schema

This module defines the UnifiedCandidate class that standardizes
the output format across all datasets.

All datasets must return candidates in this format to ensure
compatibility with the query/ranking/agent layers.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json


@dataclass
class UnifiedCandidate:
    """
    Unified root cause candidate representation.

    This class standardizes the output format for all datasets,
    ensuring that different RCA systems (Telecom, Market, Bank, etc.)
    produce compatible results that can be processed by the same
    query/ranking/agent layers.

    Required fields (always present):
    - dataset: Which dataset this candidate comes from
    - component: Component identified as root cause
    - component_type: Type of component (database, service, cache, etc.)
    - score: Confidence score (0.0-1.0)
    - anomaly_start_time: When anomaly started (unix seconds)
    - anomaly_end_time: When anomaly ended (unix seconds)

    Optional fields:
    - peak_time: Time of maximum severity
    - evidence: Supporting evidence
    - dataset_specific: Dataset-specific data that doesn't fit standard fields
    """

    # Required fields
    dataset: str
    component: str
    component_type: str
    score: float
    anomaly_start_time: int
    anomaly_end_time: int

    # Optional fields
    peak_time: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    dataset_specific: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate fields after initialization."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")

        if self.anomaly_start_time > self.anomaly_end_time:
            raise ValueError(
                f"Start time {self.anomaly_start_time} cannot be after "
                f"end time {self.anomaly_end_time}"
            )

    @staticmethod
    def from_telecom_result(
        component: str,
        component_type: str,
        score: float,
        anomaly_start: int,
        anomaly_end: int,
        evidence: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "UnifiedCandidate":
        """
        Create UnifiedCandidate from Telecom RCA result.

        Factory method for converting Telecom-specific result format
        to unified format.

        Args:
            component: Component identifier
            component_type: Component type
            score: Confidence score
            anomaly_start: Anomaly start time (unix seconds)
            anomaly_end: Anomaly end time (unix seconds)
            evidence: Optional evidence dictionary
            **kwargs: Additional fields stored in dataset_specific

        Returns:
            UnifiedCandidate instance
        """
        return UnifiedCandidate(
            dataset="Telecom",
            component=component,
            component_type=component_type,
            score=score,
            anomaly_start_time=anomaly_start,
            anomaly_end_time=anomaly_end,
            peak_time=kwargs.pop('peak_time', None),
            evidence=evidence or {},
            dataset_specific=kwargs,
        )

    @staticmethod
    def from_market_result(
        component: str,
        component_type: str,
        score: float,
        anomaly_start: int,
        anomaly_end: int,
        evidence: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "UnifiedCandidate":
        """
        Create UnifiedCandidate from Market RCA result.

        Factory method for converting Market-specific result format
        to unified format.

        Args:
            component: Component identifier
            component_type: Component type
            score: Confidence score
            anomaly_start: Anomaly start time (unix seconds)
            anomaly_end: Anomaly end time (unix seconds)
            evidence: Optional evidence dictionary
            **kwargs: Additional fields stored in dataset_specific

        Returns:
            UnifiedCandidate instance
        """
        return UnifiedCandidate(
            dataset="Market",
            component=component,
            component_type=component_type,
            score=score,
            anomaly_start_time=anomaly_start,
            anomaly_end_time=anomaly_end,
            peak_time=kwargs.pop('peak_time', None),
            evidence=evidence or {},
            dataset_specific=kwargs,
        )

    @staticmethod
    def from_bank_result(
        component: str,
        component_type: str,
        score: float,
        anomaly_start: int,
        anomaly_end: int,
        evidence: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "UnifiedCandidate":
        """
        Create UnifiedCandidate from Bank RCA result.

        Factory method for converting Bank-specific result format
        to unified format.

        Args:
            component: Component identifier
            component_type: Component type
            score: Confidence score
            anomaly_start: Anomaly start time (unix seconds)
            anomaly_end: Anomaly end time (unix seconds)
            evidence: Optional evidence dictionary
            **kwargs: Additional fields stored in dataset_specific

        Returns:
            UnifiedCandidate instance
        """
        return UnifiedCandidate(
            dataset="Bank",
            component=component,
            component_type=component_type,
            score=score,
            anomaly_start_time=anomaly_start,
            anomaly_end_time=anomaly_end,
            peak_time=kwargs.pop('peak_time', None),
            evidence=evidence or {},
            dataset_specific=kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary representation of candidate
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        Convert to JSON string.

        Returns:
            JSON representation of candidate
        """
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UnifiedCandidate("
            f"dataset={self.dataset}, "
            f"component={self.component}, "
            f"type={self.component_type}, "
            f"score={self.score:.2f})"
        )

    def __lt__(self, other: "UnifiedCandidate") -> bool:
        """Compare by score (for sorting)."""
        return self.score > other.score  # Higher score sorts first
