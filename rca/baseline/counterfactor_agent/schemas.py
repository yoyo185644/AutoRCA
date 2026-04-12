from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CandidateHypothesis:
    component: str
    reason: str
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


@dataclass
class ControllerState:
    instruction: str
    dataset_name: str = ""
    task_focus: str = ""
    max_step: int = 25
    max_turn: int = 5

    step: int = 0
    turn: int = 0
    done: bool = False

    candidates: List[CandidateHypothesis] = field(default_factory=list)
    checks: List[CounterfactualCheck] = field(default_factory=list)
    final_candidate: Optional[CandidateHypothesis] = None
    final_confidence: float = 0.0
    final_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "dataset_name": self.dataset_name,
            "task_focus": self.task_focus,
            "max_step": self.max_step,
            "max_turn": self.max_turn,
            "step": self.step,
            "turn": self.turn,
            "done": self.done,
            "candidates": [c.to_dict() for c in self.candidates],
            "checks": [c.to_dict() for c in self.checks],
            "final_candidate": self.final_candidate.to_dict() if self.final_candidate else None,
            "final_confidence": self.final_confidence,
            "final_reasoning": self.final_reasoning,
        }