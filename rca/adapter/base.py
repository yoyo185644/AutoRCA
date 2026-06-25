from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


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

        hint_components = [
            self.parse_component_id(x)
            for x in list((context or {}).get("scoring_point_components") or [])
            if str(x).strip()
        ]
        hint_components = [x for x in dict.fromkeys(hint_components) if x]
        hint_reasons = [
            self.parse_root_cause_label(x)
            for x in list((context or {}).get("scoring_point_reasons") or [])
            if str(x).strip()
        ]
        hint_reasons = [x for x in dict.fromkeys(hint_reasons) if x]
        global_reasons = [
            self.parse_root_cause_label(x)
            for x in list((context or {}).get("allowed_reasons") or [])
            if str(x).strip()
        ]
        global_reasons = [x for x in dict.fromkeys(global_reasons) if x]

        max_base = max([abs(float(getattr(c, "score", 0.0) or 0.0)) for c in candidates] + [1e-9])
        max_comp = max([abs(float(v)) for v in comp_scores.values()] + [1e-9])
        max_reason = max([abs(float(v)) for v in reason_scores.values()] + [1e-9])

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

            score = 0.50 * base_norm
            if bool(need_comp):
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
                if metrics:
                    old_ev = [str(x) for x in list(getattr(cand, "evidence", []) or [])]
                    cand.evidence = list(dict.fromkeys(old_ev + metrics[:6]))
            except Exception:
                pass
            scored.append((float(score), cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return self._postprocess_reranked([c for _, c in scored])

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
        Derive component-type prior weights from dataset telemetry and labels.

        lift(type) = P(root-cause is this type) / P(type exists in metric data)

        Returns a dict mapping semantic type → multiplier (normalised to mean=1.0).
        Falls back to an empty dict (all priors = 1.0) on any I/O error.
        """
        from collections import Counter

        import pandas as pd

        # ── Step 1: RC type frequencies from scoring_points in query.csv ────
        query_path = f"dataset/{self.dataset}/query.csv"
        rc_counter: Counter[str] = Counter()
        try:
            df_q = pd.read_csv(query_path)
            for sp in df_q.get("scoring_points", pd.Series(dtype=str)).dropna():
                for m in self._COMP_RE.finditer(str(sp)):
                    raw = m.group(1).strip(" .")
                    ctype = self.component_semantic_type(self.parse_component_id(raw))
                    if ctype:
                        rc_counter[ctype] += 1
        except Exception:
            return {}

        if not rc_counter or sum(rc_counter.values()) < min_rc_count:
            return {}

        # ── Step 2: Base-rate from telemetry component column ────────────────
        paths = dict(self.config.get("paths", {}) or {})
        telemetry_root = str(
            paths.get("telemetry_root", f"dataset/{self.dataset}/telemetry")
        )
        data_counter: Counter[str] = Counter()
        file_count = 0
        col_lower = {c.lower(): c for c in self._TELEMETRY_COMP_COLS}

        if os.path.isdir(telemetry_root):
            for dirpath, _dirs, names in os.walk(telemetry_root):
                for name in sorted(names):
                    if not name.lower().endswith((".csv", ".tsv")):
                        continue
                    if file_count >= max_telemetry_files:
                        break
                    path = os.path.join(dirpath, name)
                    try:
                        sep = "\t" if name.endswith(".tsv") else ","
                        hdr = pd.read_csv(path, nrows=0, sep=sep)
                        matched = [
                            str(c)
                            for c in hdr.columns
                            if str(c).lower() in col_lower
                        ]
                        if not matched:
                            continue
                        df_t = pd.read_csv(
                            path, usecols=[matched[0]], nrows=50_000, sep=sep
                        )
                        for val in df_t[matched[0]].dropna().unique():
                            ctype = self.component_semantic_type(
                                self.parse_component_id(str(val))
                            )
                            if ctype:
                                data_counter[ctype] += 1
                        file_count += 1
                    except Exception:
                        continue
                if file_count >= max_telemetry_files:
                    break

        # ── Step 3: Compute lift and normalise ───────────────────────────────
        all_types = set(rc_counter.keys()) | set(data_counter.keys())
        n = len(all_types) or 1
        total_rc = sum(rc_counter.values())
        total_data = sum(data_counter.values()) or 1

        priors: Dict[str, float] = {}
        for ctype in all_types:
            rc_cnt = rc_counter.get(ctype, 0)
            if rc_cnt < min_rc_count:
                continue
            data_cnt = max(data_counter.get(ctype, 0), 1)
            p_rc = (rc_cnt + smoothing) / (total_rc + smoothing * n)
            p_data = (data_cnt + smoothing) / (total_data + smoothing * n)
            lift = float(max(1.0 / clip, min(clip, p_rc / p_data)))
            priors[ctype] = lift

        if not priors:
            return {}

        # Normalise so mean = 1.0 (preserves relative weights, unbiased overall)
        mean_v = sum(priors.values()) / len(priors)
        if mean_v > 0:
            priors = {k: v / mean_v for k, v in priors.items()}
        return priors

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

    def reason_semantic_type(self, reason: str) -> str:
        return ""

    def reason_metric_keywords(self, reason: str) -> List[str]:
        return []

    def llm_reason_rules(self) -> List[str]:
        return [
            "Prefer candidates whose reason is compatible with the component type.",
            "Prefer candidates whose anomalous metrics directly support the reason.",
            "Do not override deterministic top-1 unless the alternative has stronger direct evidence.",
        ]

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
        # Base implementation: merge with estimate_occurrence_datetime_candidates so that
        # Market/Bank get the same multi-signal fusion (±1min neighbors, reason-aware
        # metric onset, component metric onset) that TelecomAdapter already provides.
        merged = [dict(x) for x in list(time_candidates or []) if isinstance(x, dict)]
        if need_time:
            try:
                from rca.adapter.utils.time_candidates import estimate_occurrence_datetime_candidates
                from rca.adapter.utils.query_parsing import extract_reason_candidates
                reasons = extract_reason_candidates(str(scoring_points or ""))
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
