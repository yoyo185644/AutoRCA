from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CandidateHypothesis:
    component: str
    reason: str
    datetime: str = ""
    evidence: List[str] = field(default_factory=list)
    score: float = 0.0
    source: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeterministicEvidence:
    metric_anomaly_score: float = 0.0
    peak_timestamp_alignment: float = 0.0
    upstream_dependency_score: float = 0.0
    downstream_dependency_score: float = 0.0
    trace_duration_anomaly: float = 0.0
    historical_frequency: float = 0.0
    causal_graph_consistency: float = 0.0

    matched_metrics: List[str] = field(default_factory=list)
    matched_traces: List[str] = field(default_factory=list)
    dependency_paths: List[str] = field(default_factory=list)
    historical_cases: List[str] = field(default_factory=list)
    graph_explanations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualCheck:
    candidate: CandidateHypothesis
    counterfactual_question: str
    expected_effect: str
    judgement: str  # support / weaken / uncertain
    confidence: float
    rationale: str
    supporting_evidence: List[str] = field(default_factory=list)
    deterministic_evidence: Optional[DeterministicEvidence] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["candidate"] = self.candidate.to_dict()
        if self.deterministic_evidence is not None:
            d["deterministic_evidence"] = self.deterministic_evidence.to_dict()
        return d

