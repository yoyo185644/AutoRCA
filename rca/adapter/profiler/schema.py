from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TelemetryAvailability:
    metric_files: int = 0
    trace_files: int = 0
    log_files: int = 0

    @property
    def has_metrics(self) -> bool:
        return self.metric_files > 0

    @property
    def has_traces(self) -> bool:
        return self.trace_files > 0

    @property
    def has_logs(self) -> bool:
        return self.log_files > 0


@dataclass(frozen=True)
class MetricFamilyProfile:
    family: str
    metric_count: int = 0
    component_count: int = 0
    sample_count: int = 0
    median_sampling_interval_s: Optional[float] = None
    sparsity: float = 0.0
    noise_mad_ratio: float = 0.0
    burstiness_p99_p50: float = 1.0
    value_p50: float = 0.0
    value_p95: float = 0.0
    value_p99: float = 0.0


@dataclass(frozen=True)
class CandidateBudgetProfile:
    component_count: int = 0
    metric_count: int = 0
    estimated_top_k: int = 8
    estimated_reason_top_r: int = 3
    estimated_time_top_k: int = 6
    score_entropy_hint: float = 0.0
    score_elbow_hint: int = 8


@dataclass(frozen=True)
class AdaptiveKnobs:
    confidence_level: float = 0.95
    evidence_budget: str = "medium"
    history_weight: float = 0.0
    llm_risk_tolerance: str = "conservative"


@dataclass(frozen=True)
class DatasetProfile:
    dataset: str
    telemetry_root: str
    timezone: str = "Asia/Shanghai"
    availability: TelemetryAvailability = field(default_factory=TelemetryAvailability)
    day_count: int = 0
    component_count: int = 0
    metric_count: int = 0
    component_types: Dict[str, int] = field(default_factory=dict)
    metric_families: Dict[str, MetricFamilyProfile] = field(default_factory=dict)
    time_span_seconds: int = 0
    median_sampling_interval_s: Optional[float] = None
    typical_anomaly_duration_s: int = 300
    typical_propagation_lag_s: int = 300
    budget: CandidateBudgetProfile = field(default_factory=CandidateBudgetProfile)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
