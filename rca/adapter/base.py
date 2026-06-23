from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TimeSolveResult:
    root_dt: str
    center_time_s: Optional[int]
    evidence: Dict[str, Any]


@dataclass(frozen=True)
class ReasonSolveResult:
    det_reason: str
    det_reason_other: str
    ready: bool
    evidence: Dict[str, Any]


@dataclass(frozen=True)
class ComponentSolveResult:
    component: str
    dt_override: Optional[str]
    evidence: Any
    reason: str


@dataclass(frozen=True)
class AnomalySegment:
    component: str
    start_s: int
    end_s: int
    signal_name: str
    severity: float
    value: float


@dataclass(frozen=True)
class TopologyEdge:
    src: str
    dst: str
    weight: float


@dataclass(frozen=True)
class CandidateSchema:
    component: str
    score: float
    reason: str
    occurrence_datetime: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter:
    """Base class for dataset-specific RCA strategies.

    Subclasses override specific methods to provide custom behavior.
    Returning None from any method signals "use default implementation".
    """

    def __init__(self, dataset: str, config: Dict[str, Any]) -> None:
        self.dataset = str(dataset)
        self.config = dict(config or {})

    def solve_time(
        self,
        *,
        instruction: str,
        scoring_points: str,
        need_time: bool,
        k_fail: int,
        **kwargs: Any,
    ) -> Optional[TimeSolveResult]:
        return None

    def solve_reason(
        self,
        *,
        instruction: str,
        scoring_points: str,
        need_reason: bool,
        **kwargs: Any,
    ) -> Optional[ReasonSolveResult]:
        return None

    def solve_component(
        self,
        *,
        spec: Any,
        instruction: str,
        scoring_points: str,
        need_comp: bool,
        need_time: bool,
        need_reason: bool,
        center_time_s: Optional[int],
        det_reason: str,
        det_reason_other: str,
        det_reason_ready: bool,
        ev2: Any,
        **kwargs: Any,
    ) -> Optional[ComponentSolveResult]:
        return None

    def assemble_prediction(
        self,
        *,
        spec: Any,
        instruction: str,
        scoring_points: str,
        need_time: bool,
        need_comp: bool,
        need_reason: bool,
        k_fail: int,
        root_dt: str,
        root_comp: str,
        root_reason: str,
        det_reason: str,
        det_reason_other: str,
        det_reason_ready: bool,
        **kwargs: Any,
    ) -> Optional[dict]:
        return None

    def get_component_delta_s(
        self,
        *,
        spec: Any,
        need_comp: bool,
        need_time: bool,
        center_time_s: Optional[int],
        **kwargs: Any,
    ) -> Any:
        return 300

    def should_use_light_agent(
        self,
        *,
        spec: Any,
        need_comp: bool,
        ev3_gap_norm: float,
        no_llm_fallback: bool,
        args: Any,
        **kwargs: Any,
    ) -> bool:
        return False

    def get_time_dt_override(
        self,
        *,
        need_time: bool,
        comp: str,
        ev3: Any,
        **kwargs: Any,
    ) -> Optional[str]:
        return None

    def parse_component_id(self, raw: str) -> str:
        return str(raw).strip()

    def parse_root_cause_label(
        self,
        raw_reason: str,
    ) -> str:
        return str(raw_reason).strip()

    def extract_reason_hints(self, *, scoring_points: str) -> List[str]:
        return []

    def coerce_closed_set_lists(self) -> bool:
        return False

    def enable_structured_time_candidates(self) -> bool:
        return False

    def should_collect_component_evidence_for_time(
        self,
        *,
        task_id: int,
        need_time: bool,
        need_comp: bool,
        need_reason: bool,
    ) -> bool:
        return False

    def prefer_deterministic_time_only(self) -> bool:
        return False

    def get_time_candidate_top_k(self, *, task_id: int, default_top_k: int) -> int:
        return int(default_top_k)

    def extract_anomaly_segments(
        self,
        *,
        instruction: str,
        center_time_s: Optional[int],
        delta_s: int,
        severity_floor: float = 0.0,
        **kwargs: Any,
    ) -> List[AnomalySegment]:
        return []

    def extract_topology_edges(
        self,
        *,
        instruction: str,
        expanded_start_s: int,
        expanded_end_s: int,
        max_rows: Optional[int] = None,
        **kwargs: Any,
    ) -> List[TopologyEdge]:
        return []

    def build_candidate_schema(
        self,
        *,
        candidates: Sequence[Any],
        scores: Dict[str, float],
        ev3: Any,
        top_k: int = 12,
        **kwargs: Any,
    ) -> List[CandidateSchema]:
        if not candidates:
            return []
        mx = max(abs(float(v)) for v in scores.values()) if scores else 1.0
        if mx <= 1e-9:
            mx = 1.0
        out: List[CandidateSchema] = []
        for c in candidates[: int(top_k)]:
            comp = str(getattr(c, "component", "") or getattr(c, "component_id", "") or "").strip()
            if not comp:
                continue
            out.append(
                CandidateSchema(
                    component=comp,
                    score=float(scores.get(comp, 0.0)) / mx,
                    reason=str(getattr(c, "reason", "") or "").strip(),
                    occurrence_datetime=str(getattr(c, "occurrence_datetime", "") or "").strip() or None,
                    evidence={},
                )
            )
        return out

    def supports_multi_root_joint_selection(self) -> bool:
        return False

    def extract_component_level(self, *, scoring_points: str) -> str:
        return ""

    def canonicalize_component(self, *, component: str, component_level: str) -> str:
        return str(component or "").strip()

    def normalize_evidence_groups(
        self,
        *,
        groups: Sequence[Any],
        component_level: str,
    ) -> List[Any]:
        return list(groups or [])

    def filter_candidate_hypotheses(
        self,
        *,
        candidates: Sequence[Any],
        component_level: str,
        need_comp: bool,
        need_reason: bool,
        need_time: bool,
    ) -> List[Any]:
        return list(candidates or [])

    def rerank_candidate_hypotheses(
        self,
        *,
        candidates: Sequence[Any],
        context: Dict[str, Any],
        component_level: str,
        need_comp: bool,
        need_reason: bool,
        need_time: bool,
    ) -> Optional[List[Any]]:
        return None

    def expand_index_candidate_tuples(
        self,
        *,
        selected: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        task_id: int,
        k: int,
    ) -> List[Dict[str, Any]]:
        return [dict(x) for x in list(selected or []) if isinstance(x, dict)]

    def select_time_candidate_override(
        self,
        *,
        time_candidates: Sequence[Dict[str, Any]],
        component: str,
        reason: str,
        need_comp: bool,
        need_reason: bool,
        need_time: bool,
        fallback: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        return None

    def constrain_llm_to_index_candidates(self) -> bool:
        return False

    def augment_index_reasons(
        self,
        *,
        reasons: Sequence[str],
        result: Any,
        instruction: str,
        scoring_points: str,
        task_id: int,
        args: Any,
        need_reason: bool,
    ) -> Tuple[List[str], Dict[str, float], Dict[str, Any]]:
        return list(reasons or []), {}, {}

    def augment_index_time_candidates(
        self,
        *,
        time_candidates: Sequence[Dict[str, Any]],
        result: Any,
        instruction: str,
        scoring_points: str,
        task_id: int,
        args: Any,
        need_time: bool,
        component_scores: Dict[str, float],
        component_first_times: Dict[str, str],
        component_peak_times: Dict[str, str],
        context_overrides: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return [dict(x) for x in list(time_candidates or []) if isinstance(x, dict)]

    def postprocess_prediction_json_multi(
        self,
        *,
        prediction_json: str,
        selected_groups: Sequence[Dict[str, Any]],
        required_root_count: int,
        component_level: str,
    ) -> str:
        return str(prediction_json or "")

    def enable_trace_evidence(self) -> bool:
        return True

    def basil_query_config_overrides(
        self,
        *,
        task_id: int,
        need_time: bool,
        need_comp: bool,
        need_reason: bool,
        component_level: str,
    ) -> Dict[str, Any]:
        return {}

    def fallback_component_candidates_from_basil(self, evidence: Any) -> List[str]:
        return []
