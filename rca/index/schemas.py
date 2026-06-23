from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def component_type(component: str) -> str:
    c = str(component or "")
    if c.startswith(("os_", "node-")):
        return "node"
    if c.startswith(("docker_", "container")):
        return "container"
    if c.startswith("db_"):
        return "db"
    if "-" in c:
        return "service"
    return "other"


@dataclass(frozen=True)
class AnomalySegment:
    """A compact anomaly object used by BASIL instead of raw telemetry rows."""

    segment_id: str
    dataset: str
    date_dir: str
    component: str
    component_type: str
    signal_type: str
    signal_name: str
    start_s: int
    peak_s: int
    end_s: int
    severity: float
    reason_hints: List[str] = field(default_factory=list)
    top_metrics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def overlaps(self, start_s: int, end_s: int) -> bool:
        return int(self.start_s) <= int(end_s) and int(self.end_s) >= int(start_s)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "AnomalySegment":
        return AnomalySegment(
            segment_id=str(obj.get("segment_id", "")),
            dataset=str(obj.get("dataset", "")),
            date_dir=str(obj.get("date_dir", "")),
            component=str(obj.get("component", "")),
            component_type=str(obj.get("component_type", component_type(obj.get("component", "")))),
            signal_type=str(obj.get("signal_type", "")),
            signal_name=str(obj.get("signal_name", "")),
            start_s=int(obj.get("start_s", 0) or 0),
            peak_s=int(obj.get("peak_s", 0) or 0),
            end_s=int(obj.get("end_s", 0) or 0),
            severity=float(obj.get("severity", 0.0) or 0.0),
            reason_hints=[str(x) for x in obj.get("reason_hints", []) or []],
            top_metrics=[str(x) for x in obj.get("top_metrics", []) or []],
            metadata=dict(obj.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class TraceTopologyEdge:
    src: str
    dst: str
    edge_type: str = "trace_latency"
    support_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalEdgeSketch:
    """Lag-bucketed propagation summary for a typed topology edge."""

    src: str
    dst: str
    edge_type: str
    lag_bin: str
    support_count: int
    transfer_max: float
    lift_max: float
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlockMaxBlock:
    block_id: int
    members: List[str]
    ub: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RCAHypothesis:
    component: str
    reason: str = ""
    onset_s: Optional[int] = None
    local_score: float = 0.0
    causal_score: float = 0.0
    reason_score: float = 0.0
    score: float = 0.0
    upper_bound: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BasilQueryStats:
    total_components: int
    total_blocks: int
    visited_blocks: int
    refined_components: int
    pruned_blocks: int
    kth_score: float
    max_remaining_ub: float
    epsilon: float
    exact_mode: bool
    ub_slack: Dict[str, float] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BasilComponentResult:
    dataset: str
    date_dir: str
    start_local: str
    end_local: str
    local_start_s: int
    local_end_s: int
    estimated_failure_time: str
    candidates: List[str]
    metric_scores: Dict[str, float]
    trace_scores: Dict[str, float]
    causal_scores: Dict[str, float]
    reason_scores: Dict[str, float]
    combined_scores: Dict[str, float]
    peak_times: Dict[str, str]
    first_times: Dict[str, str]
    top_metrics: Dict[str, List[str]]
    hypotheses: List[RCAHypothesis]
    causal_edges: List[CausalEdgeSketch]
    stats: BasilQueryStats
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hypotheses"] = [h.to_dict() for h in self.hypotheses]
        d["causal_edges"] = [e.to_dict() for e in self.causal_edges]
        d["stats"] = self.stats.to_dict()
        return d


DEFAULT_LAG_BINS: Tuple[Tuple[str, int, int], ...] = (
    ("0-30s", 0, 30),
    ("30-60s", 30, 60),
    ("1-3m", 60, 180),
    ("3-5m", 180, 300),
    ("5-15m", 300, 900),
    ("15m+", 900, 10**12),
)


def lag_bin_name(lag_s: int) -> str:
    lag = max(0, int(lag_s))
    for name, lo, hi in DEFAULT_LAG_BINS:
        if lo <= lag < hi:
            return name
    return DEFAULT_LAG_BINS[-1][0]
