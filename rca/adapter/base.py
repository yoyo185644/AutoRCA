from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rca.common.text import norm_token


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
        need_time: bool,
        k_fail: int,
        **kwargs: Any,
    ) -> Optional[TimeSolveResult]:
        return None

    def solve_reason(
        self,
        *,
        instruction: str,
        need_reason: bool,
        **kwargs: Any,
    ) -> Optional[ReasonSolveResult]:
        return None

    def solve_component(
        self,
        *,
        spec: Any,
        instruction: str,
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
        if not bool(need_comp):
            return False
        if bool(no_llm_fallback):
            return False
        if not bool(getattr(args, "light_agent", False)):
            return False
        if not self._allow_light_agent(spec=spec, need_comp=bool(need_comp)):
            return False
        gap_thr = float(getattr(args, "task3_agent_gap", 0.15) or 0.15)
        return float(ev3_gap_norm or 0.0) < gap_thr

    def _allow_light_agent(self, *, spec: Any, need_comp: bool) -> bool:
        """Return True if the task type permits LLM-assisted component selection.

        Subclasses override this to restrict or expand the allowed task set.
        Default: allow tasks 3, 5, 6, 7 (tasks that require component prediction).
        """
        return int(getattr(spec, "task_id", 0) or 0) in self._light_agent_task_ids()

    def _light_agent_task_ids(self) -> set:
        """Task IDs for which light-agent LLM calls are permitted."""
        return {3, 5, 6, 7}

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

    def extract_reason_hints(self) -> List[str]:
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

    def extract_component_level(self) -> str:
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
        """Unified reranking framework shared by all adapters.

        Score = 0.50×base_norm + 0.25×comp_norm + 0.25×reason_norm
                + _type_consistency()   # component-reason alignment
                + _hint_bonus()         # scoring-point hint alignment
        The result is multiplied by _component_prior() to filter junk components.
        Subclasses override the hook methods rather than this method.
        """
        if not candidates:
            return None
        try:
            from rca.counterfactor_agent.schemas import CandidateHypothesis
        except Exception:
            return None

        comp_scores = dict((context or {}).get("component_scores", {}) or {})
        reason_scores = dict((context or {}).get("reason_scores", {}) or {})
        top_metrics = dict((context or {}).get("top_metrics", {}) or {})

        hint_components: List[str] = []
        hint_reasons: List[str] = []
        global_reasons = [
            self.parse_root_cause_label(x)
            for x in list((context or {}).get("allowed_reasons") or [])
            if str(x).strip()
        ]
        global_reasons = [x for x in dict.fromkeys(global_reasons) if x]

        max_base = max([abs(float(getattr(c, "score", 0.0) or 0.0)) for c in candidates] + [1e-9])
        max_comp = max([abs(float(v)) for v in comp_scores.values()] + [1e-9])
        max_reason = max([abs(float(v)) for v in reason_scores.values()] + [1e-9])
        component_only = bool(need_comp) and not bool(need_reason) and not bool(need_time)

        scored: List[Tuple[float, Any]] = []
        for cand in list(candidates or []):
            comp = self.canonicalize_component(
                component=str(getattr(cand, "component", "") or ""),
                component_level=str(component_level or ""),
            )
            reason = self.parse_root_cause_label(str(getattr(cand, "reason", "") or ""))
            if bool(need_reason) and not reason:
                reason = hint_reasons[0] if hint_reasons else (global_reasons[0] if global_reasons else "")
            if bool(need_comp) and not comp:
                continue
            if bool(need_reason) and not reason:
                continue

            prior = self._component_prior(comp)
            if prior <= 0.0:
                continue

            base_norm = float(getattr(cand, "score", 0.0) or 0.0) / max_base
            comp_norm = float(comp_scores.get(comp, 0.0)) / max_comp if comp else 0.0
            reason_norm = float(reason_scores.get(reason, 0.0)) / max_reason if reason else 0.0
            metrics = [str(x) for x in list(top_metrics.get(comp, []) or [])]

            rootness_features: Dict[str, float] = {}
            if component_only:
                local_ev = self.component_local_metric_evidence(component=comp, context=context)
                local_metrics = [str(x) for x in list(local_ev.get("top_metrics", []) or []) if str(x).strip()]
                if local_metrics and not metrics:
                    metrics = local_metrics
                metric_ev = self.metric_rank_evidence(component=comp, metrics=metrics)
                latent_ev = self.latent_reason_metric_score(component=comp, metrics=metrics)
                score = self.component_evidence_score(
                    component=comp,
                    base_score=float(getattr(cand, "score", 0.0) or 0.0),
                    context=context,
                    metric_score=max(
                        float(metric_ev.get("metric_score", 0.0) or 0.0),
                        float(latent_ev.get("score", 0.0) or 0.0),
                        float(local_ev.get("metric_score", 0.0) or 0.0),
                    ),
                    ownership_score=max(
                        float(metric_ev.get("ownership", 0.0) or 0.0),
                        float(local_ev.get("ownership", 0.0) or 0.0),
                    ),
                    symptomness=max(
                        float(metric_ev.get("symptomness", 0.0) or 0.0),
                        float(local_ev.get("symptomness", 0.0) or 0.0),
                    ),
                    locality=float(local_ev.get("locality", 0.0) or 0.0),
                    persistence=float(local_ev.get("persistence", 0.0) or 0.0),
                    instance_owned_score=float(local_ev.get("instance_owned_score", 0.0) or 0.0),
                    sibling_contrast=float(local_ev.get("sibling_contrast", 0.0) or 0.0),
                )
                rootness_features = self._component_rootness_features(
                    component=comp,
                    base_score=float(getattr(cand, "score", 0.0) or 0.0),
                    context=context,
                    metric_score=max(
                        float(metric_ev.get("metric_score", 0.0) or 0.0),
                        float(latent_ev.get("score", 0.0) or 0.0),
                        float(local_ev.get("metric_score", 0.0) or 0.0),
                    ),
                    ownership_score=max(
                        float(metric_ev.get("ownership", 0.0) or 0.0),
                        float(local_ev.get("ownership", 0.0) or 0.0),
                    ),
                    symptomness=max(
                        float(metric_ev.get("symptomness", 0.0) or 0.0),
                        float(local_ev.get("symptomness", 0.0) or 0.0),
                    ),
                    locality=float(local_ev.get("locality", 0.0) or 0.0),
                    persistence=float(local_ev.get("persistence", 0.0) or 0.0),
                    instance_owned_score=float(local_ev.get("instance_owned_score", 0.0) or 0.0),
                    sibling_contrast=float(local_ev.get("sibling_contrast", 0.0) or 0.0),
                )
                try:
                    cand.source = str(getattr(cand, "source", "") or "adapter_rerank")
                    if latent_ev.get("reason"):
                        old_ev = [str(x) for x in list(getattr(cand, "evidence", []) or [])]
                        old_ev.append(f"latent_reason={latent_ev.get('reason')}")
                        if local_ev.get("metric_score", 0.0):
                            old_ev.append(
                                "local_change="
                                f"{float(local_ev.get('metric_score', 0.0) or 0.0):.3f}"
                            )
                        cand.evidence = list(dict.fromkeys(old_ev))
                except Exception:
                    pass
            else:
                score = 0.50 * base_norm
            if bool(need_comp) and not (bool(need_comp) and not bool(need_reason) and not bool(need_time)):
                score += 0.25 * comp_norm
            if bool(need_reason):
                score += 0.25 * reason_norm
            score += self._type_consistency(comp, reason, metrics)
            score += self._hint_bonus(
                comp=comp, reason=reason,
                hint_components=hint_components, hint_reasons=hint_reasons,
                need_comp=bool(need_comp), need_reason=bool(need_reason),
            )
            score *= prior

            try:
                cand.component = comp
                cand.reason = reason
                cand.score = float(score)
                if rootness_features:
                    setattr(cand, "rootness_features", dict(rootness_features))
                if metrics:
                    old_ev = [str(x) for x in list(getattr(cand, "evidence", []) or [])]
                    cand.evidence = list(dict.fromkeys(old_ev + metrics[:6]))
            except Exception:
                pass
            scored.append((float(score), cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        ordered = [c for _, c in scored]
        return self._postprocess_reranked(ordered)

    def _type_consistency(self, comp: str, reason: str, metrics: List[str]) -> float:
        """Additive bonus/penalty for component–reason type alignment.

        Override per dataset. Positive = good alignment; negative = mismatch.
        """
        return 0.0

    # ------------------------------------------------------------------
    # Data-driven component-type prior (lazy, computed once per adapter)
    # ------------------------------------------------------------------

    _COMP_RE = re.compile(
        r"root\s+cause\s+component\s+is\s+([A-Za-z0-9_\-]+)", re.IGNORECASE
    )
    _TELEMETRY_COMP_COLS = (
        "cmdb_id", "component", "component_id", "container",
        "instance", "node", "service", "pod", "host",
    )

    def _compute_component_type_priors(
        self,
        *,
        smoothing: float = 1.0,
        min_rc_count: int = 2,
        max_telemetry_files: int = 30,
        clip: float = 4.0,
    ) -> Dict[str, float]:
        """
        Return neutral priors by default.

        Earlier experimental versions estimated priors from evaluation labels.
        That leaks ground truth and is intentionally disabled. Dataset-specific
        adapters may provide non-leaking priors from telemetry-only profiles,
        but the shared path must stay neutral.
        """
        return {}

    def _component_prior(self, comp: str) -> float:
        """Multiplicative weight applied after scoring.

        The base implementation computes data-driven per-type priors once and
        caches them. Subclasses may override for fully custom behaviour.
        Return 0.0 to exclude a component entirely.
        """
        if not hasattr(self, "_cached_component_type_priors"):
            self._cached_component_type_priors: Dict[str, float] = (
                self._compute_component_type_priors()
            )
        priors = self._cached_component_type_priors
        if not priors:
            return 1.0
        ctype = self.component_semantic_type(comp)
        return priors.get(ctype, 1.0)

    def _hint_bonus(
        self,
        *,
        comp: str,
        reason: str,
        hint_components: List[str],
        hint_reasons: List[str],
        need_comp: bool,
        need_reason: bool,
    ) -> float:
        """Additive bonus for alignment with scoring-point hint components/reasons."""
        bonus = 0.0
        if need_comp and hint_components and comp in hint_components:
            bonus += 1.10
        if need_reason and hint_reasons and reason in hint_reasons:
            bonus += 0.95
        return bonus

    def _postprocess_reranked(self, candidates: List[Any]) -> List[Any]:
        """Optional post-processing after sorting (e.g. diversity filtering)."""
        return candidates

    def postprocess_output_tuples(
        self,
        tuples: Sequence[Dict[str, Any]],
        *,
        context: Dict[str, Any],
        need_comp: bool,
        need_reason: bool,
        need_time: bool,
    ) -> List[Dict[str, Any]]:
        """Optional final tuple-level post-processing after datetime binding."""
        return [dict(x) for x in list(tuples or []) if isinstance(x, Mapping)]

    # ------------------------------------------------------------------
    # Candidate-pool supplement: metric anomaly scan (universal recall fix)
    # ------------------------------------------------------------------

    def supplement_candidate_pool(
        self,
        existing_tuples: List[Dict[str, Any]],
        context: Dict[str, Any],
        *,
        date_dir: str = "",
        max_new_components: int = 8,
        min_breach_ratio: float = 0.08,
        supplement_score: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """Return additional candidate dicts to merge into the pool before reranking.

        Base implementation scans metric files for the incident time window and
        nominates components with significant breach anomalies that are absent
        from the existing BASIL pool.  Subclasses may extend (e.g. closed-set
        injection) by calling super() and appending their own entries.

        Parameters
        ----------
        existing_tuples:
            Candidate dicts already in the pool (from BASIL / prior steps).
        context:
            Agent context dict; must contain ``query_start_local``,
            ``query_end_local``, and ``dataset``.
        date_dir:
            Telemetry date-folder name (e.g. ``"2021_03_04"``).
        max_new_components:
            Hard cap on how many new distinct components to add.
        min_breach_ratio:
            Normalised breach score threshold (0–1) below which components
            are ignored.
        supplement_score:
            Raw score assigned to supplementary candidates before normalisation
            in reranking.  Should be below typical BASIL scores (1.0–2.0) so
            BASIL wins when it is correct, but high enough to compete when it
            is wrong.
        """
        import os
        import pandas as pd
        from rca.adapter.utils.component_candidates import _metric_component_scores

        # ── resolve time window ──────────────────────────────────────────────
        start_local = str(context.get("query_start_local", "") or "")
        end_local = str(context.get("query_end_local", "") or "")
        dataset = str(context.get("dataset", "") or "") or self.dataset
        if not start_local or not end_local:
            return []

        try:
            tz = str(self.config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
            start_s = int(pd.Timestamp(start_local, tz=tz).tz_convert("UTC").timestamp())
            end_s = int(pd.Timestamp(end_local, tz=tz).tz_convert("UTC").timestamp())
        except Exception:
            return []

        # ── locate metric directory ──────────────────────────────────────────
        if not date_dir:
            return []
        metric_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "metric")
        if not os.path.isdir(metric_dir):
            return []

        # ── run breach scan ──────────────────────────────────────────────────
        try:
            raw_scores, _, first_times, top_metrics = _metric_component_scores(
                metric_dir, start_s, end_s
            )
        except Exception:
            return []
        if not raw_scores:
            return []

        max_score = max(raw_scores.values())
        if max_score <= 1e-9:
            return []
        norm: Dict[str, float] = {c: v / max_score for c, v in raw_scores.items()}

        # ── components already represented in pool ───────────────────────────
        pool_comps: set = {
            self.parse_component_id(str(t.get("component", "") or ""))
            for t in existing_tuples
            if str(t.get("component", "") or "").strip()
        }

        # ── reasons to pair with new components (from existing pool) ─────────
        pool_reasons: List[str] = list(
            dict.fromkeys(
                str(t.get("reason", "") or "").strip()
                for t in existing_tuples
                if str(t.get("reason", "") or "").strip()
            )
        )
        # Use at most 2 distinct reasons to avoid combinatorial explosion
        pair_reasons: List[str] = pool_reasons[:2] if pool_reasons else [""]

        # ── build supplement entries ─────────────────────────────────────────
        ranked = sorted(norm.items(), key=lambda kv: kv[1], reverse=True)
        supplements: List[Dict[str, Any]] = []
        added_comps = 0

        for raw_comp, nscore in ranked:
            if nscore < min_breach_ratio:
                break
            comp_id = self.parse_component_id(str(raw_comp))
            if comp_id in pool_comps:
                continue

            first_dt = str(first_times.get(raw_comp, "") or "")
            metrics = list(top_metrics.get(raw_comp, []) or [])[:6]

            for reason in pair_reasons:
                supplements.append(
                    {
                        "component": comp_id,
                        "reason": reason,
                        "datetime": first_dt,
                        "score": float(supplement_score * nscore),
                        "source": "metric_supplement",
                        "supporting_metrics": metrics,
                        "component_type": "",
                        "first_anomaly_time": first_dt,
                        "symptomness": 0.0,
                    }
                )

            pool_comps.add(comp_id)
            added_comps += 1
            if added_comps >= max_new_components:
                break

        return supplements

    def expand_index_candidate_tuples(
        self,
        *,
        selected: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        task_id: int,
        k: int,
    ) -> List[Dict[str, Any]]:
        return [dict(x) for x in list(selected or []) if isinstance(x, dict)]

    _DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

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
        """Unified time-candidate selection.

        Scores all candidates, returns the highest-scoring valid datetime.
        Subclasses customise via _preprocess_time_candidates() and
        _score_time_candidate() rather than overriding this method.
        Telecom is the exception — it delegates to a specialised utility.
        """
        if not bool(need_time):
            return None
        candidates = [dict(x) for x in list(time_candidates or []) if isinstance(x, Mapping)]
        if not candidates:
            return (str(fallback), {"source": f"{self.dataset}_time_fallback"}) if str(fallback or "").strip() else None

        candidates = self._preprocess_time_candidates(candidates)
        comp = self.parse_component_id(str(component or ""))
        reason_norm = self.parse_root_cause_label(str(reason or ""))

        scored = sorted(
            candidates,
            key=lambda x: self._score_time_candidate(x, comp, reason_norm),
            reverse=True,
        )
        for item in scored:
            dt = str(item.get("datetime", "") or "").strip()
            if self._DATETIME_RE.match(dt):
                return dt, {**item, "source": str(item.get("source", "") or f"{self.dataset}_time_selected")}

        return (str(fallback), {"source": f"{self.dataset}_time_fallback"}) if str(fallback or "").strip() else None

    def _preprocess_time_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optional filtering/sorting before scoring. Override in subclass."""
        return candidates

    def _score_time_candidate(self, item: Dict[str, Any], comp: str, reason: str) -> float:
        """Score a single time candidate given the resolved component and reason.

        Default: base score + bonus for exact component/reason match.
        Subclasses override for dataset-specific signals (semantic family,
        source type, etc.).
        """
        score = float(item.get("score", 0.0) or 0.0)
        c0 = self.parse_component_id(str(item.get("component", "") or ""))
        r0 = self.parse_root_cause_label(str(item.get("reason", "") or ""))
        if comp and c0:
            score += 0.45 if c0 == comp else -0.12
        if reason and r0:
            score += 0.28 if r0 == reason else -0.08
        return score

    def constrain_llm_to_index_candidates(self) -> bool:
        return False

    def index_candidate_pool_k(
        self,
        *,
        task_id: int,
        k: int,
        default_pool_k: int,
        context: Dict[str, Any],
    ) -> int:
        return int(default_pool_k)

    def refine_index_llm_gate(
        self,
        *,
        gate: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        task_id: int,
    ) -> Dict[str, Any]:
        return dict(gate or {})

    def component_semantic_type(self, component: str) -> str:
        return ""

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _clip01(value: Any) -> float:
        try:
            v = float(value)
        except Exception:
            return 0.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def component_evidence_score(
        self,
        *,
        component: str,
        base_score: float,
        context: Mapping[str, Any],
        metric_score: float = 0.0,
        ownership_score: float = 0.0,
        symptomness: float = 0.0,
        locality: float = 0.0,
        persistence: float = 0.0,
        instance_owned_score: float = 0.0,
        sibling_contrast: float = 0.0,
    ) -> float:
        """Shared component ranking score for indexed evidence candidates.

        The formula is dataset-agnostic. Dataset adapters only define semantic
        type and metric ownership; this helper handles global vs typed-channel
        normalization, component-local metric strength, and symptom discount.
        """
        comp = self.parse_component_id(str(component or ""))
        comp_scores = dict((context or {}).get("component_scores", {}) or {})
        max_global = max([abs(self._safe_float(v)) for v in comp_scores.values()] + [abs(self._safe_float(base_score)), 1.0])
        raw_norm = self._safe_float(comp_scores.get(comp, base_score)) / max_global
        raw_norm = max(0.0, min(1.0, raw_norm))

        by_channel: Dict[str, List[Tuple[str, float]]] = {}
        for c, score in comp_scores.items():
            ch = str(self.component_semantic_type(str(c)) or "other")
            by_channel.setdefault(ch, []).append((str(c), self._safe_float(score)))
        for values in by_channel.values():
            values.sort(key=lambda kv: kv[1], reverse=True)

        channel = str(self.component_semantic_type(comp) or "other")
        channel_values = by_channel.get(channel, [])
        max_channel = max([abs(v) for _, v in channel_values] + [abs(self._safe_float(base_score)), 1.0])
        local_norm = self._safe_float(comp_scores.get(comp, base_score)) / max_channel
        local_norm = max(0.0, min(1.0, local_norm))
        rank_prior = 0.0
        for idx, (c, _score) in enumerate(channel_values, start=1):
            if c == comp:
                rank_prior = 1.0 / float(idx)
                break
        propagation = self._component_propagation_features(component=comp, context=context)

        try:
            from rca.adapter.ranker.rootness import RootnessFeatures, RootnessRanker

            return float(
                RootnessRanker().score(
                    RootnessFeatures.from_mapping(
                        {
                            "global_score": raw_norm,
                            "channel_score": local_norm,
                            "channel_rank_score": rank_prior,
                            "local_change": self._safe_float(metric_score),
                            "ownership": self._safe_float(ownership_score),
                            "locality": self._safe_float(locality),
                            "persistence": self._safe_float(persistence),
                            "instance_owned_score": self._safe_float(instance_owned_score),
                            "sibling_contrast": self._safe_float(sibling_contrast),
                            "symptomness": self._safe_float(symptomness),
                            **propagation,
                        }
                    )
                )
            )
        except Exception:
            score = (
                0.50 * max(0.0, raw_norm)
                + 0.35 * max(0.0, local_norm)
                + 0.10 * max(0.0, rank_prior)
                + 0.20 * max(0.0, self._safe_float(metric_score))
                + 0.10 * max(0.0, self._safe_float(ownership_score))
            )
            sym = max(0.0, min(1.0, self._safe_float(symptomness)))
            if sym > 0.0:
                score *= max(0.55, 1.0 - 0.35 * sym)
            score += 0.12 * max(0.0, propagation.get("causal_source_score", 0.0))
            score += 0.08 * max(0.0, propagation.get("outgoing_propagation", 0.0))
            incoming = max(0.0, propagation.get("incoming_symptomness", 0.0))
            if incoming > 0.0:
                score *= max(0.50, 1.0 - 0.40 * incoming)
            return float(max(0.0, score))

    def _edge_value(self, edge: Any, key: str, default: Any = None) -> Any:
        if isinstance(edge, Mapping):
            return edge.get(key, default)
        return getattr(edge, key, default)

    def _parse_context_time_s(self, value: Any, *, context: Mapping[str, Any]) -> int:
        text = str(value or "").strip()
        if not text:
            return 0
        try:
            import pandas as pd

            tz = str(self.config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
            ts = pd.Timestamp(text)
            if ts.tzinfo is None:
                ts = ts.tz_localize(tz)
            return int(ts.tz_convert("UTC").timestamp())
        except Exception:
            return 0

    def _component_propagation_features(
        self,
        *,
        component: str,
        context: Mapping[str, Any],
    ) -> Dict[str, float]:
        """Trace-direction evidence shared by all dataset adapters.

        Causal edges in the BASIL index use ``src -> dst`` to denote a
        propagation candidate: the source component can explain later symptoms
        at the destination.  A root-like component should have strong outgoing
        edges whose destination anomalies occur later; a symptom-like component
        often has strong incoming edges from earlier upstream anomalies.
        """
        comp = self.parse_component_id(str(component or ""))
        if not comp:
            return {"causal_source_score": 0.0, "outgoing_propagation": 0.0, "incoming_symptomness": 0.0}

        causal_scores = dict((context or {}).get("causal_scores", {}) or {})
        max_causal = max([abs(self._safe_float(v)) for v in causal_scores.values()] + [1.0])
        causal_source = self._clip01(self._safe_float(causal_scores.get(comp, 0.0)) / max_causal)

        first_times = dict((context or {}).get("component_first_times", {}) or {})
        comp_t = self._parse_context_time_s(first_times.get(comp), context=context)

        edges = list((context or {}).get("causal_edges", []) or [])
        max_edge_score = 0.0
        for edge in edges:
            max_edge_score = max(
                max_edge_score,
                self._safe_float(self._edge_value(edge, "score", 0.0)),
                self._safe_float(self._edge_value(edge, "transfer_max", 0.0)),
                self._safe_float(self._edge_value(edge, "lift_max", 0.0)),
            )
        max_edge_score = max(max_edge_score, 1e-9)

        outgoing = 0.0
        incoming = 0.0
        for edge in edges:
            src = self.parse_component_id(str(self._edge_value(edge, "src", "") or ""))
            dst = self.parse_component_id(str(self._edge_value(edge, "dst", "") or ""))
            if not src or not dst:
                continue
            score = max(
                self._safe_float(self._edge_value(edge, "score", 0.0)),
                self._safe_float(self._edge_value(edge, "transfer_max", 0.0)),
                self._safe_float(self._edge_value(edge, "lift_max", 0.0)),
            )
            score = self._clip01(score / max_edge_score)
            if score <= 0.0:
                continue

            if src == comp:
                dst_t = self._parse_context_time_s(first_times.get(dst), context=context)
                if comp_t > 0 and dst_t > 0:
                    temporal = 1.0 if comp_t <= dst_t else 0.25
                else:
                    temporal = 0.60
                outgoing += score * temporal
            if dst == comp:
                src_t = self._parse_context_time_s(first_times.get(src), context=context)
                if comp_t > 0 and src_t > 0:
                    temporal = 1.0 if src_t <= comp_t else 0.25
                else:
                    temporal = 0.60
                incoming += score * temporal

        return {
            "causal_source_score": self._clip01(causal_source),
            "outgoing_propagation": self._clip01(outgoing),
            "incoming_symptomness": self._clip01(incoming),
        }

    def _component_rootness_features(
        self,
        *,
        component: str,
        base_score: float,
        context: Mapping[str, Any],
        metric_score: float = 0.0,
        ownership_score: float = 0.0,
        symptomness: float = 0.0,
        locality: float = 0.0,
        persistence: float = 0.0,
        instance_owned_score: float = 0.0,
        sibling_contrast: float = 0.0,
    ) -> Dict[str, float]:
        comp = self.parse_component_id(str(component or ""))
        comp_scores = dict((context or {}).get("component_scores", {}) or {})
        max_global = max([abs(self._safe_float(v)) for v in comp_scores.values()] + [abs(self._safe_float(base_score)), 1.0])
        raw_norm = max(0.0, min(1.0, self._safe_float(comp_scores.get(comp, base_score)) / max_global))

        by_channel: Dict[str, List[Tuple[str, float]]] = {}
        for c, score in comp_scores.items():
            ch = str(self.component_semantic_type(str(c)) or "other")
            by_channel.setdefault(ch, []).append((str(c), self._safe_float(score)))
        for values in by_channel.values():
            values.sort(key=lambda kv: kv[1], reverse=True)

        channel = str(self.component_semantic_type(comp) or "other")
        channel_values = by_channel.get(channel, [])
        max_channel = max([abs(v) for _, v in channel_values] + [abs(self._safe_float(base_score)), 1.0])
        local_norm = max(0.0, min(1.0, self._safe_float(comp_scores.get(comp, base_score)) / max_channel))
        rank_prior = 0.0
        for idx, (c, _score) in enumerate(channel_values, start=1):
            if c == comp:
                rank_prior = 1.0 / float(idx)
                break
        propagation = self._component_propagation_features(component=comp, context=context)
        return {
            "global_score": raw_norm,
            "channel_score": local_norm,
            "channel_rank_score": max(0.0, min(1.0, rank_prior)),
            "local_change": max(0.0, min(1.0, self._safe_float(metric_score))),
            "ownership": max(0.0, min(1.0, self._safe_float(ownership_score))),
            "locality": max(0.0, min(1.0, self._safe_float(locality))),
            "persistence": max(0.0, min(1.0, self._safe_float(persistence))),
            "instance_owned_score": max(0.0, min(1.0, self._safe_float(instance_owned_score))),
            "sibling_contrast": max(0.0, min(1.0, self._safe_float(sibling_contrast))),
            "symptomness": max(0.0, min(1.0, self._safe_float(symptomness))),
            **propagation,
        }

    def _pairwise_rerank_component_only(self, candidates: Sequence[Any]) -> List[Any]:
        try:
            from rca.adapter.ranker.rootness import RootnessRanker
        except Exception:
            return list(candidates or [])
        rows: List[Dict[str, Any]] = []
        by_id: Dict[int, Any] = {}
        for idx, cand in enumerate(list(candidates or [])):
            row = {
                "idx": idx,
                "score": float(getattr(cand, "score", 0.0) or 0.0),
                "rootness_features": dict(getattr(cand, "rootness_features", {}) or {}),
            }
            rows.append(row)
            by_id[idx] = cand
        ranked = RootnessRanker().rerank_pairwise(rows)
        return [by_id[int(row["idx"])] for row in ranked if int(row.get("idx", -1)) in by_id]

    def component_local_metric_evidence(
        self,
        *,
        component: str,
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Telemetry-only local change evidence for component ranking.

        This shared scorer does not read query labels or historical records. It
        asks a causal-change-point question for every component: were this
        component's own metrics weak before the incident window, stronger in the
        window, persistent, and relatively local to a small set of components?
        Dataset adapters only provide canonical component names and metric
        ownership via ``metric_rank_evidence``.
        """
        comp = self.parse_component_id(str(component or ""))
        if not comp:
            return {"metric_score": 0.0, "ownership": 0.0, "symptomness": 0.0, "top_metrics": []}
        evidence = self._component_local_metric_evidence_map(context=context)
        return dict(evidence.get(comp, {}) or {"metric_score": 0.0, "ownership": 0.0, "symptomness": 0.0, "top_metrics": []})

    def _component_local_metric_evidence_map(self, *, context: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        import math
        import os

        import pandas as pd

        dataset = str((context or {}).get("dataset", "") or "") or self.dataset
        date_dir = str((context or {}).get("date_dir", "") or "")
        start_local = str((context or {}).get("query_start_local", "") or "")
        end_local = str((context or {}).get("query_end_local", "") or "")
        if not dataset or not date_dir or not start_local or not end_local:
            return {}

        cache_key = (dataset, date_dir, start_local, end_local)
        cache = getattr(self, "_component_local_metric_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_component_local_metric_cache", cache)
        if cache_key in cache:
            return dict(cache[cache_key])

        metric_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "metric")
        if not os.path.isdir(metric_dir):
            cache[cache_key] = {}
            return {}

        try:
            tz = str(self.config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
            start_s = int(pd.Timestamp(start_local, tz=tz).tz_convert("UTC").timestamp())
            end_s = int(pd.Timestamp(end_local, tz=tz).tz_convert("UTC").timestamp())
        except Exception:
            cache[cache_key] = {}
            return {}
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        window_s = max(60, int(end_s - start_s))
        before_start_s = int(start_s - max(600, window_s))

        try:
            from rca.adapter.utils.component_candidates import _normalize_metric_file
        except Exception:
            cache[cache_key] = {}
            return {}

        comp_metric_sum: Dict[str, Dict[str, float]] = {}
        comp_minute_sum: Dict[str, Dict[int, float]] = {}
        comp_before_sum: Dict[str, float] = {}
        metric_components: Dict[str, set[str]] = {}

        paths = [
            os.path.join(metric_dir, name)
            for name in os.listdir(metric_dir)
            if str(name).endswith(".csv")
        ]
        for path in paths:
            try:
                long = _normalize_metric_file(path)
            except Exception:
                continue
            if long.empty:
                continue
            try:
                p95 = (
                    long.groupby(["component", "metric"], as_index=False)["value"]
                    .quantile(0.95)
                    .rename(columns={"value": "p95"})
                )
                use = long[
                    (long["ts_s"] >= before_start_s)
                    & (long["ts_s"] <= end_s)
                ].copy()
                if use.empty:
                    continue
                use = use.merge(p95, on=["component", "metric"], how="left").dropna(subset=["p95"]).copy()
                denom = use["p95"].abs().where(use["p95"].abs() > 1e-6, 1.0)
                use["breach_ratio"] = ((use["value"] - use["p95"]) / denom).clip(lower=0.0, upper=10.0)
                use = use[use["breach_ratio"] > 0].copy()
                if use.empty:
                    continue
                use["component"] = use["component"].map(lambda x: self.parse_component_id(str(x)))
                use = use[use["component"].astype(str).str.len() > 0].copy()
                if use.empty:
                    continue
                use["metric"] = use["metric"].astype(str)
                use["minute"] = (use["ts_s"].astype(int) // 60) * 60
            except Exception:
                continue

            win = use[(use["ts_s"] >= start_s) & (use["ts_s"] <= end_s)].copy()
            before = use[(use["ts_s"] >= before_start_s) & (use["ts_s"] < start_s)].copy()

            if not before.empty:
                for comp, value in before.groupby("component")["breach_ratio"].sum().items():
                    comp_before_sum[str(comp)] = comp_before_sum.get(str(comp), 0.0) + float(value)

            if win.empty:
                continue
            by_cm = win.groupby(["component", "metric"], as_index=False)["breach_ratio"].sum()
            for _, row in by_cm.iterrows():
                comp = str(row["component"])
                metric = str(row["metric"])
                value = float(row["breach_ratio"])
                comp_metric_sum.setdefault(comp, {})
                comp_metric_sum[comp][metric] = comp_metric_sum[comp].get(metric, 0.0) + value
                metric_components.setdefault(metric, set()).add(comp)

            by_min = win.groupby(["component", "minute"], as_index=False)["breach_ratio"].sum()
            for _, row in by_min.iterrows():
                comp = str(row["component"])
                minute = int(row["minute"])
                value = float(row["breach_ratio"])
                comp_minute_sum.setdefault(comp, {})
                comp_minute_sum[comp][minute] = comp_minute_sum[comp].get(minute, 0.0) + value

        if not comp_metric_sum:
            cache[cache_key] = {}
            return {}

        comps = sorted(comp_metric_sum.keys())
        n_comp = max(1, len(comps))
        raw: Dict[str, Dict[str, Any]] = {}
        max_peak = 1e-9
        max_contrast = 1e-9
        for comp in comps:
            metric_scores = comp_metric_sum.get(comp, {})
            top = sorted(metric_scores.items(), key=lambda kv: kv[1], reverse=True)[:8]
            top_metrics = [str(m) for m, _ in top]
            minute_values = list(comp_minute_sum.get(comp, {}).values())
            peak = max(minute_values) if minute_values else 0.0
            after_mean = float(sum(minute_values)) / float(max(1, len(minute_values)))
            before_total = float(comp_before_sum.get(comp, 0.0) or 0.0)
            before_minutes = max(1, window_s // 60)
            before_mean = before_total / float(before_minutes)
            contrast = max(0.0, after_mean - before_mean)
            threshold = max(0.05, 0.25 * peak)
            persistence = (
                float(sum(1 for v in minute_values if v >= threshold)) / float(max(1, len(minute_values)))
                if minute_values
                else 0.0
            )
            locality_values = []
            for metric in top_metrics:
                df = len(metric_components.get(metric, set()))
                if n_comp > 1:
                    locality_values.append(math.log((1.0 + n_comp) / (1.0 + df)) / math.log(1.0 + n_comp))
            locality = float(sum(locality_values)) / float(max(1, len(locality_values))) if locality_values else 0.0
            rank_ev = self.metric_rank_evidence(component=comp, metrics=top_metrics)
            raw[comp] = {
                "peak": float(peak),
                "contrast": float(contrast),
                "persistence": float(max(0.0, min(1.0, persistence))),
                "locality": float(max(0.0, min(1.0, locality))),
                "ownership": float(rank_ev.get("ownership", 0.0) or 0.0),
                "symptomness": float(rank_ev.get("symptomness", 0.0) or 0.0),
                "top_metrics": top_metrics,
            }
            max_peak = max(max_peak, float(peak))
            max_contrast = max(max_contrast, float(contrast))

        preliminary: Dict[str, float] = {}
        for comp, ev in raw.items():
            peak_norm = float(ev["peak"]) / max_peak
            contrast_norm = float(ev["contrast"]) / max_contrast
            persistence = float(ev["persistence"])
            locality = float(ev["locality"])
            ownership = float(ev["ownership"])
            symptomness = max(0.0, min(1.0, float(ev["symptomness"])))
            metric_score = (
                0.40 * contrast_norm
                + 0.25 * peak_norm
                + 0.20 * persistence
                + 0.15 * locality
            )
            if ownership > 0:
                metric_score = 0.75 * metric_score + 0.25 * ownership
            if symptomness > 0:
                metric_score *= max(0.60, 1.0 - 0.30 * symptomness)
            preliminary[comp] = float(max(0.0, min(1.0, metric_score)))

        by_sibling_type: Dict[str, List[float]] = {}
        for comp, score in preliminary.items():
            ctype = str(self.component_semantic_type(comp) or "other")
            by_sibling_type.setdefault(ctype, []).append(float(score))
        for values in by_sibling_type.values():
            values.sort()

        out: Dict[str, Dict[str, Any]] = {}
        for comp, ev in raw.items():
            persistence = float(ev["persistence"])
            locality = float(ev["locality"])
            ownership = float(ev["ownership"])
            symptomness = max(0.0, min(1.0, float(ev["symptomness"])))
            metric_score = preliminary.get(comp, 0.0)
            siblings = by_sibling_type.get(str(self.component_semantic_type(comp) or "other"), [])
            if siblings:
                mid = len(siblings) // 2
                median = siblings[mid] if len(siblings) % 2 else 0.5 * (siblings[mid - 1] + siblings[mid])
                max_sibling = max(siblings) if siblings else 1.0
                denom = max(1e-9, float(max_sibling) - float(median))
                sibling_contrast = max(0.0, (float(metric_score) - float(median)) / denom) if max_sibling > median else 0.0
            else:
                sibling_contrast = 0.0
            instance_owned_score = max(0.0, min(1.0, ownership * (0.50 + 0.50 * float(metric_score)) * (1.0 - 0.50 * symptomness)))
            out[comp] = {
                "metric_score": float(max(0.0, min(1.0, metric_score))),
                "ownership": ownership,
                "symptomness": symptomness,
                "top_metrics": list(ev.get("top_metrics", []) or []),
                "contrast": float(ev["contrast"]),
                "persistence": persistence,
                "locality": locality,
                "instance_owned_score": float(max(0.0, min(1.0, instance_owned_score))),
                "sibling_contrast": float(max(0.0, min(1.0, sibling_contrast))),
            }
        cache[cache_key] = dict(out)
        return out

    def metric_rank_evidence(self, *, component: str, metrics: Sequence[str]) -> Dict[str, float]:
        """Dataset hook for component-local metric ownership.

        Returns values in [0, 1]:
        - metric_score: strength of component-owned metric evidence
        - ownership: fraction/quality of metrics owned by this component type
        - symptomness: how much evidence looks like shared downstream symptoms
        """
        return {"metric_score": 0.0, "ownership": 0.0, "symptomness": 0.0}

    def latent_reason_metric_score(self, *, component: str, metrics: Sequence[str]) -> Dict[str, Any]:
        """Infer a non-output latent reason score from taxonomy and metrics.

        This is used for component-only tasks: the final answer remains a
        component, but component ranking can still benefit from reason-like
        evidence without reading evaluation labels.
        """
        rows = [str(m or "").lower() for m in list(metrics or []) if str(m or "").strip()]
        if not rows:
            return {"score": 0.0, "reason": ""}
        blob = " ".join(rows)
        best_reason = ""
        best_score = 0.0
        for reason in list(self.basil_reason_taxonomy() or []):
            label = self.parse_root_cause_label(str(reason or ""))
            keywords = [str(x).lower() for x in list(self.reason_metric_keywords(label) or []) if str(x).strip()]
            if not keywords:
                continue
            hits = sum(1 for tok in keywords if tok in blob)
            score = float(hits) / float(max(1, min(4, len(keywords))))
            expected = set(self.reason_compatible_component_types(label) or [])
            ctype = str(self.component_semantic_type(component) or "")
            if expected and ctype in expected:
                score += 0.25
            elif expected and ctype and ctype not in expected:
                score *= 0.60
            if score > best_score:
                best_score = float(score)
                best_reason = str(label)
        return {"score": float(max(0.0, min(1.0, best_score))), "reason": best_reason}

    def reason_semantic_type(self, reason: str) -> str:
        return ""

    def reason_metric_keywords(self, reason: str) -> List[str]:
        return []

    def reason_compatible_component_types(self, reason: str) -> List[str]:
        return []

    def basil_reason_taxonomy(self) -> List[str]:
        data = self.config.get("reason_taxonomy", {})
        if isinstance(data, Mapping):
            return [str(k) for k in data.keys() if str(k).strip()]
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return [str(x) for x in data if str(x).strip()]
        return []

    def llm_reason_rules(self) -> List[str]:
        return [
            "Prefer candidates whose reason is compatible with the component type.",
            "Prefer candidates whose anomalous metrics directly support the reason.",
            "Do not override deterministic top-1 unless the alternative has stronger direct evidence.",
        ]

    def _surface_reason_scores(
        self,
        *,
        top_metrics: Mapping[str, Sequence[str]],
        comp_scores: Mapping[str, float],
    ) -> Dict[str, float]:
        """Telemetry-only reason evidence from indexed component metrics.

        This is intentionally independent of ``scoring_points`` and historical
        labels.  It asks a simple dataset-agnostic question: which taxonomy
        reason is best supported by the metrics already surfaced by the index?
        Dataset adapters provide only taxonomy, canonicalization, component
        type and metric-keyword hooks.
        """
        taxonomy = [
            self.parse_root_cause_label(str(x))
            for x in list(self.basil_reason_taxonomy() or [])
            if str(x).strip()
        ]
        taxonomy = [x for x in dict.fromkeys(taxonomy) if x]
        if not taxonomy:
            return {}

        ranked_components = [
            str(c)
            for c, _score in sorted(
                dict(comp_scores or {}).items(),
                key=lambda kv: self._safe_float(kv[1]),
                reverse=True,
            )
            if str(c or "").strip()
        ]
        if not ranked_components:
            ranked_components = [
                str(c)
                for c in dict(top_metrics or {}).keys()
                if str(c or "").strip()
            ]

        max_comp = max([abs(self._safe_float(v)) for v in dict(comp_scores or {}).values()] + [1.0])
        scores: Dict[str, float] = {r: 0.0 for r in taxonomy}
        for rank, comp in enumerate(ranked_components[:16], start=1):
            metrics = [str(x).lower() for x in list(dict(top_metrics or {}).get(comp, []) or []) if str(x).strip()]
            if not metrics:
                continue
            joined = " ".join(metrics)
            comp_norm = self._safe_float(dict(comp_scores or {}).get(comp, 0.0)) / max(max_comp, 1e-9)
            rank_weight = 1.0 / (1.0 + 0.25 * float(rank - 1))
            component_weight = max(0.20, comp_norm) * rank_weight
            ctype = str(self.component_semantic_type(comp) or "")
            for reason in taxonomy:
                keywords = [
                    str(k).lower()
                    for k in list(self.reason_metric_keywords(reason) or [])
                    if str(k).strip()
                ]
                if keywords:
                    matched = {k for k in keywords if k in joined}
                    support = float(len(matched)) / float(max(1, min(4, len(set(keywords)))))
                else:
                    r_tokens = [x for x in re.split(r"\s+", norm_token(reason)) if len(x) >= 3]
                    matched = {t for t in r_tokens if t in joined}
                    support = float(len(matched)) / float(max(1, min(3, len(set(r_tokens)))))
                if support <= 0.0:
                    continue
                expected = set(self.reason_compatible_component_types(reason) or [])
                if expected and ctype:
                    support *= 1.18 if ctype in expected else 0.72
                scores[reason] = max(scores.get(reason, 0.0), component_weight * min(1.0, support))

        max_score = max([abs(v) for v in scores.values()] + [0.0])
        if max_score <= 1e-12:
            return scores
        return {k: max(0.0, float(v) / max_score) for k, v in scores.items()}

    def validate_llm_candidate_adoption(
        self,
        *,
        proposed_component: str,
        proposed_reason: str,
        deterministic_component: str,
        deterministic_reason: str,
        context: Dict[str, Any],
        need_comp: bool,
        need_reason: bool,
    ) -> Tuple[bool, str]:
        comp = self.parse_component_id(str(proposed_component or ""))
        reason = self.parse_root_cause_label(str(proposed_reason or ""))
        ctype = str(self.component_semantic_type(comp) or "")
        rtype = str(self.reason_semantic_type(reason) or "")
        if bool(need_comp) and bool(need_reason) and ctype and rtype and ctype != rtype:
            return False, f"component/reason semantic mismatch: component_type={ctype}, reason_type={rtype}"

        keywords = [str(x).lower() for x in list(self.reason_metric_keywords(reason) or []) if str(x).strip()]
        top_metrics = dict((context or {}).get("top_metrics", {}) or {})
        metrics = [str(x).lower() for x in list(top_metrics.get(comp, []) or [])]
        if bool(need_reason) and keywords and metrics and not any(k in " ".join(metrics) for k in keywords):
            return False, f"reason metrics unsupported: reason={reason}, expected_keywords={keywords[:6]}"
        return True, ""

    def augment_index_reasons(
        self,
        *,
        reasons: Sequence[str],
        result: Any,
        instruction: str,
        task_id: int,
        args: Any,
        need_reason: bool,
    ) -> Tuple[List[str], Dict[str, float], Dict[str, Any]]:
        if not need_reason:
            return list(reasons or []), {}, {}

        taxonomy = [
            self.parse_root_cause_label(str(x))
            for x in list(self.basil_reason_taxonomy() or [])
            if str(x).strip()
        ]
        taxonomy = [x for x in dict.fromkeys(taxonomy) if x]
        if not taxonomy:
            return list(reasons or []), {}, {}

        tuple_scores: Dict[str, float] = {}
        for item in [dict(x) for x in list(getattr(result, "tuples", []) or []) if isinstance(x, Mapping)]:
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            if reason in taxonomy:
                tuple_scores[reason] = max(tuple_scores.get(reason, 0.0), self._safe_float(item.get("score", 0.0)))

        comp_ev = getattr(result, "component_evidence", None)
        top_metrics = dict(getattr(comp_ev, "top_metrics", {}) or {})
        comp_scores = dict(getattr(comp_ev, "combined_scores", {}) or {})
        if not top_metrics:
            top_metrics = dict(getattr(result, "top_metrics", {}) or {})
        if not comp_scores:
            comp_scores = dict(getattr(result, "combined_scores", {}) or {})

        surface_scores = self._surface_reason_scores(top_metrics=top_metrics, comp_scores=comp_scores)
        merged_scores: Dict[str, float] = {r: 0.0 for r in taxonomy}
        for reason in taxonomy:
            merged_scores[reason] = max(
                self._safe_float(tuple_scores.get(reason, 0.0)),
                self._safe_float(surface_scores.get(reason, 0.0)),
            )

        for raw in list(reasons or []):
            reason = self.parse_root_cause_label(str(raw))
            if reason in taxonomy:
                merged_scores.setdefault(reason, 0.0)

        max_score = max([abs(v) for v in merged_scores.values()] + [0.0])
        if max_score > 1e-12:
            norm_scores = {k: max(0.0, float(v) / max_score) for k, v in merged_scores.items()}
        else:
            norm_scores = {k: 0.0 for k in taxonomy}

        ordered = [
            r
            for r, _score in sorted(
                norm_scores.items(),
                key=lambda kv: float(kv[1]),
                reverse=True,
            )
        ]
        reason_metrics: List[str] = []
        for comp, _score in sorted(dict(comp_scores or {}).items(), key=lambda kv: self._safe_float(kv[1]), reverse=True)[:12]:
            reason_metrics.extend([str(x) for x in list(dict(top_metrics or {}).get(comp, []) or []) if str(x).strip()])
        return ordered, norm_scores, {
            "reason_surface_scores": surface_scores,
            "reason_metrics": list(dict.fromkeys(reason_metrics))[:30],
        }

    def augment_index_time_candidates(
        self,
        *,
        time_candidates: Sequence[Dict[str, Any]],
        result: Any,
        instruction: str,
        task_id: int,
        args: Any,
        need_time: bool,
        component_scores: Dict[str, float],
        component_first_times: Dict[str, str],
        component_peak_times: Dict[str, str],
        context_overrides: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Base implementation: merge with estimate_occurrence_datetime_candidates so that
        # Market/Bank get the same multi-signal fusion (±1min neighbors, reason-aware
        # metric onset, component metric onset) that TelecomAdapter already provides.
        merged = [dict(x) for x in list(time_candidates or []) if isinstance(x, dict)]
        if need_time:
            try:
                from rca.adapter.utils.time_candidates import estimate_occurrence_datetime_candidates
                reasons = list(self.basil_reason_taxonomy() or [])
                generated = estimate_occurrence_datetime_candidates(
                    dataset=str(getattr(self, "dataset", "") or ""),
                    instruction=str(instruction or ""),
                    component_first_times=dict(component_first_times or {}),
                    component_peak_times=dict(component_peak_times or {}),
                    component_scores=dict(component_scores or {}),
                    reasons=reasons,
                    top_k=max(16, int(getattr(args, "time_candidate_top_k", 8) or 8)),
                )
                merged.extend([cand.to_dict() for cand in generated])
                # Deduplicate by (datetime, component, source), keep highest score
                best: dict = {}
                for item in merged:
                    dt = str(item.get("datetime", "") or "").strip()
                    if not dt:
                        continue
                    key = (dt, str(item.get("component", "") or ""), str(item.get("source", "") or ""))
                    if key not in best or float(item.get("score", 0.0) or 0.0) > float(best[key].get("score", 0.0) or 0.0):
                        best[key] = dict(item)
                merged = sorted(best.values(), key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
                merged = merged[:max(16, int(getattr(args, "time_candidate_top_k", 8) or 8))]
            except Exception:
                pass
        return merged

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
