from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from rca.index.config import infer_basil_task_type
from rca.index.query import BasilQueryConfig, basil_query_config_from_mapping, query_basil_components
from rca.adapter.utils.time_candidates import TimeCandidate


class TaskType(str, Enum):
    TIME_ONLY = "TIME_ONLY"
    REASON_ONLY = "REASON_ONLY"
    COMPONENT_ONLY = "COMPONENT_ONLY"
    REASON_TIME = "REASON_TIME"
    COMPONENT_TIME = "COMPONENT_TIME"
    COMPONENT_REASON = "COMPONENT_REASON"
    COMPONENT_REASON_TIME = "COMPONENT_REASON_TIME"


def get_task_type(task_id: int) -> TaskType:
    return TaskType(infer_basil_task_type(task_id=int(task_id)))


@dataclass(frozen=True)
class BasilEvidence:
    incident_id: str
    query_window: Tuple[datetime, datetime]
    task_type: TaskType
    component_scores: Dict[str, float]
    time_candidates: Dict[str, List[datetime]]
    reason_scores: Dict[str, Dict[str, float]]
    supporting_metrics: Dict[str, List[str]]
    causal_scores: Dict[str, float]
    low_confidence: bool
    confidence_reason: Optional[str] = None
    first_times: Dict[str, str] = field(default_factory=dict)
    peak_times: Dict[str, str] = field(default_factory=dict)
    estimated_failure_time: str = ""
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BasilComponentCandidates:
    candidates: List[str]
    scores: Dict[str, float]
    causal_scores: Dict[str, float]
    low_confidence: bool
    confidence_reason: Optional[str] = None


@dataclass(frozen=True)
class BasilTimeCandidates:
    candidates: List[TimeCandidate]
    by_component: Dict[str, List[datetime]]
    low_confidence: bool
    confidence_reason: Optional[str] = None


@dataclass(frozen=True)
class BasilReasonCandidates:
    candidates: List[str]
    scores: Dict[str, float]
    by_component: Dict[str, Dict[str, float]]
    fallback_candidates: List[str] = field(default_factory=list)
    low_confidence: bool = False
    confidence_reason: Optional[str] = None


def _to_utc_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        ts = pd.Timestamp(text)
        if ts.tzinfo is None:
            ts = pd.Timestamp(text, tz="Asia/Shanghai")
        return ts.tz_convert("UTC").to_pydatetime()
    except Exception:
        return None


def _to_local_string(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")


def _reason_scores_from_hypotheses(result: Any, components: Iterable[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {str(c): {} for c in components or [] if str(c).strip()}
    for hyp in getattr(result, "hypotheses", []) or []:
        comp = str(getattr(hyp, "component", "") or "").strip()
        reason = str(getattr(hyp, "reason", "") or "").strip()
        if not comp or not reason:
            continue
        score = float(getattr(hyp, "reason_score", 0.0) or 0.0)
        if score <= 0:
            score = float(getattr(hyp, "score", 0.0) or 0.0)
        out.setdefault(comp, {})
        out[comp][reason] = max(float(out[comp].get(reason, 0.0)), float(score))
    return out


def query_basil_evidence(
    *,
    dataset: str,
    instruction: str,
    task_id: int,
    top_k: int,
    reasons: Optional[Sequence[str]] = None,
    center_time_s: Optional[int] = None,
    delta_s: int = 1800,
    config: Optional[Any] = None,
) -> BasilEvidence:
    task_type = get_task_type(int(task_id))
    cfg = basil_query_config_from_mapping(config, top_k=top_k) if isinstance(config, Mapping) else (config or BasilQueryConfig(top_k=top_k, task_type=task_type.value))
    if str(cfg.task_type) != task_type.value or int(cfg.top_k) != int(top_k):
        cfg = BasilQueryConfig(**{**cfg.__dict__, "top_k": int(top_k), "task_type": task_type.value})

    result = query_basil_components(
        dataset=dataset,
        instruction=instruction,
        top_k=top_k,
        center_time_s=center_time_s,
        delta_s=delta_s,
        reasons=reasons or [],
        config=cfg,
    )
    if int(getattr(cfg, "time_tolerance_s", 300) or 300) <= 120 and not list(getattr(result, "hypotheses", []) or []):
        try:
            relaxed = BasilQueryConfig(**{**cfg.__dict__, "time_tolerance_s": 600})
            result = query_basil_components(
                dataset=dataset,
                instruction=instruction,
                top_k=top_k,
                center_time_s=center_time_s,
                delta_s=delta_s,
                reasons=reasons or [],
                config=relaxed,
            )
            logging.getLogger(__name__).warning(
                "basil_time_tolerance_relaxed task_id=%s tolerance_s=%s relaxed_s=%s",
                str(int(task_id)),
                str(int(getattr(cfg, "time_tolerance_s", 300) or 300)),
                "600",
            )
        except Exception:
            pass

    start_dt = _to_utc_datetime(result.start_local) or datetime.now(timezone.utc)
    end_dt = _to_utc_datetime(result.end_local) or start_dt
    incident_id = f"{dataset}:{result.date_dir}:{int(start_dt.timestamp())}:{int(end_dt.timestamp())}"

    component_scores = {str(k): float(v) for k, v in (result.combined_scores or {}).items()}
    time_by_component: Dict[str, List[datetime]] = {}
    for comp in set(result.first_times.keys()) | set(result.peak_times.keys()) | set(component_scores.keys()):
        times: List[datetime] = []
        for raw in [result.first_times.get(comp), result.peak_times.get(comp)]:
            parsed = _to_utc_datetime(raw)
            if parsed is not None and parsed not in times:
                times.append(parsed)
        if times:
            time_by_component[str(comp)] = times
    hyp_components: List[str] = []
    for hyp in getattr(result, "hypotheses", []) or []:
        comp = str(getattr(hyp, "component", "") or "").strip()
        if not comp:
            continue
        onset_s = getattr(hyp, "onset_s", None)
        if onset_s is None:
            onset_s = getattr(hyp, "occurrence_time", None)
        if onset_s is None:
            continue
        try:
            dt = datetime.fromtimestamp(int(onset_s), tz=timezone.utc)
        except Exception:
            continue
        cur = list(time_by_component.get(comp, []))
        if dt not in cur:
            cur.append(dt)
        time_by_component[comp] = cur
        hyp_components.append(comp)
    if hyp_components:
        logging.getLogger(__name__).info(
            "basil_hypothesis_times_merged n_components=%s sample_components=%s",
            str(len(set(hyp_components))),
            str(list(dict.fromkeys(hyp_components))[:5]),
        )

    reason_by_component = _reason_scores_from_hypotheses(result, component_scores.keys())
    low_confidence = bool((result.stats.ub_slack or {}).get("low_confidence", 0.0))
    confidence_reason = "insufficient segments" if low_confidence else None

    return BasilEvidence(
        incident_id=incident_id,
        query_window=(start_dt, end_dt),
        task_type=task_type,
        component_scores=component_scores,
        time_candidates=time_by_component,
        reason_scores=reason_by_component,
        supporting_metrics={str(k): list(v or []) for k, v in (result.top_metrics or {}).items()},
        causal_scores={str(k): float(v) for k, v in (result.causal_scores or {}).items()},
        low_confidence=low_confidence,
        confidence_reason=confidence_reason,
        first_times=dict(result.first_times or {}),
        peak_times=dict(result.peak_times or {}),
        estimated_failure_time=str(result.estimated_failure_time or ""),
        timings=dict(result.stats.timings or {}),
    )


def build_component_candidates(evidence: BasilEvidence, *, top_k: int = 12) -> BasilComponentCandidates:
    scored = []
    for comp, score in evidence.component_scores.items():
        causal = float(evidence.causal_scores.get(comp, 0.0))
        scored.append((str(comp), float(score) + 0.05 * causal))
    scored.sort(key=lambda x: x[1], reverse=True)
    candidates = [c for c, _ in scored[: max(1, int(top_k))]]
    return BasilComponentCandidates(
        candidates=candidates,
        scores={c: float(s) for c, s in scored},
        causal_scores=dict(evidence.causal_scores),
        low_confidence=bool(evidence.low_confidence),
        confidence_reason=evidence.confidence_reason,
    )


def build_time_candidates(
    evidence: BasilEvidence,
    *,
    top_k: int = 8,
    focus_components: Optional[Sequence[str]] = None,
) -> BasilTimeCandidates:
    def _compute_impact_times(first_dt: datetime, peak_dt: Optional[datetime]) -> List[datetime]:
        if first_dt.tzinfo is None:
            first_dt = first_dt.replace(tzinfo=timezone.utc)
        out: List[datetime] = []
        seen: set[str] = set()

        def add(dt: datetime) -> None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
            if dt < first_dt.astimezone(timezone.utc):
                return
            key = dt.isoformat()
            if key in seen:
                return
            seen.add(key)
            out.append(dt)

        evolution_s = 0.0
        if peak_dt is not None:
            if peak_dt.tzinfo is None:
                peak_dt = peak_dt.replace(tzinfo=timezone.utc)
            evolution_s = max(0.0, float((peak_dt - first_dt).total_seconds()))

        for minutes in [5, 8, 10, 15, 20]:
            add(first_dt + timedelta(minutes=int(minutes)))

        if evolution_s >= 180.0:
            for factor in [0.6, 0.8, 1.0, 1.2]:
                offset_s = min(evolution_s * float(factor), 1200.0)
                add(first_dt + timedelta(seconds=float(offset_s)))
        return out

    out: List[TimeCandidate] = []
    ranked_components = sorted(evidence.component_scores.items(), key=lambda x: float(x[1]), reverse=True)
    if not ranked_components:
        ranked_components = [(c, 0.0) for c in evidence.time_candidates.keys()]
    try:
        max_comp_s = max([float(s) for _, s in ranked_components] + [0.0])
    except Exception:
        max_comp_s = 0.0
    if max_comp_s <= 1e-12:
        max_comp_s = 1.0

    for comp, comp_score in ranked_components:
        comp_key = str(comp)
        comp_norm = max(0.0, float(comp_score)) / float(max_comp_s)
        first_dt = _to_utc_datetime((evidence.first_times or {}).get(comp_key)) if evidence.first_times else None
        peak_dt = _to_utc_datetime((evidence.peak_times or {}).get(comp_key)) if evidence.peak_times else None
        extra_times = list((evidence.time_candidates or {}).get(comp_key, []) or [])
        if first_dt is None and extra_times:
            first_dt = extra_times[0]
        if peak_dt is None and len(extra_times) >= 2:
            peak_dt = extra_times[1]
        if first_dt is None:
            continue

        base_ev = {
            "incident_id": evidence.incident_id,
            "low_confidence": bool(evidence.low_confidence),
            "confidence_reason": evidence.confidence_reason or "",
            "supporting_metrics": evidence.supporting_metrics.get(comp_key, []),
        }

        evolution_s = 0.0
        if peak_dt is not None:
            try:
                evolution_s = max(0.0, float((peak_dt - first_dt).total_seconds()))
            except Exception:
                evolution_s = 0.0

        out.append(
            TimeCandidate(
                datetime=_to_local_string(first_dt),
                source="basil_first_anomaly_time",
                score=float(comp_norm * 1.0),
                component=comp_key,
                reason="",
                evidence=dict(base_ev) | {"evolution_s": int(evolution_s)},
            )
        )
        if peak_dt is not None:
            out.append(
                TimeCandidate(
                    datetime=_to_local_string(peak_dt),
                    source="basil_peak_time",
                    score=float(comp_norm * 0.85),
                    component=comp_key,
                    reason="",
                    evidence=dict(base_ev) | {"evolution_s": int(evolution_s)},
                )
            )

        if peak_dt is not None:
            if evolution_s >= 180.0:
                ramp_dt = (first_dt + timedelta(seconds=float(min(0.60 * evolution_s, 1200.0)))).replace(second=0, microsecond=0)
                confirm_dt = (first_dt + timedelta(seconds=float(min(0.40 * evolution_s, 1200.0)))).replace(second=0, microsecond=0)
            else:
                ramp_dt = (first_dt + timedelta(minutes=3)).replace(second=0, microsecond=0)
                confirm_dt = (first_dt + timedelta(minutes=2)).replace(second=0, microsecond=0)
            out.append(
                TimeCandidate(
                    datetime=_to_local_string(ramp_dt),
                    source="basil_ramp_time",
                    score=float(comp_norm * 0.90),
                    component=comp_key,
                    reason="",
                    evidence=dict(base_ev) | {"evolution_s": int(evolution_s)},
                )
            )
            out.append(
                TimeCandidate(
                    datetime=_to_local_string(confirm_dt),
                    source="basil_confirm_time",
                    score=float(comp_norm * 0.88),
                    component=comp_key,
                    reason="",
                    evidence=dict(base_ev) | {"evolution_s": int(evolution_s)},
                )
            )

        impact_times = _compute_impact_times(first_dt, peak_dt)
        for t in impact_times:
            out.append(
                TimeCandidate(
                    datetime=_to_local_string(t),
                    source="basil_impact_time",
                    score=float(comp_norm * 1.02),
                    component=comp_key,
                    reason="",
                    evidence=dict(base_ev) | {"evolution_s": int(evolution_s)},
                )
            )
        if impact_times:
            logging.getLogger(__name__).info(
                "basil_impact_time_added component=%s evolution_s=%s n_impact_candidates=%s impact_times=%s",
                comp_key,
                str(int(evolution_s)),
                str(len(impact_times)),
                str([x.isoformat() for x in impact_times[:4]]),
            )

        hyp_times = list(extra_times[2:]) if len(extra_times) > 2 else []
        for t in hyp_times:
            if t is None:
                continue
            out.append(
                TimeCandidate(
                    datetime=_to_local_string(t),
                    source="basil_hypothesis_time",
                    score=float(comp_norm * 1.01),
                    component=comp_key,
                    reason="",
                    evidence=dict(base_ev) | {"origin": "hypothesis"},
                )
            )
    out.sort(key=lambda c: float(c.score), reverse=True)
    focus = list(dict.fromkeys([str(x) for x in (focus_components or []) if str(x).strip()]))
    if focus:
        picked: List[TimeCandidate] = []
        seen_dt: set[str] = set()
        for comp in focus:
            comp_best = None
            for c in out:
                if str(getattr(c, "component", "") or "") == comp:
                    comp_best = c
                    break
            if comp_best is not None:
                dt = str(getattr(comp_best, "datetime", "") or "")
                if dt and dt not in seen_dt:
                    picked.append(comp_best)
                    seen_dt.add(dt)
        for c in out:
            if len(picked) >= int(max(1, int(top_k))):
                break
            dt = str(getattr(c, "datetime", "") or "")
            if not dt or dt in seen_dt:
                continue
            picked.append(c)
            seen_dt.add(dt)
        out = picked
    return BasilTimeCandidates(
        candidates=out[: max(1, int(top_k))],
        by_component=dict(evidence.time_candidates),
        low_confidence=bool(evidence.low_confidence),
        confidence_reason=evidence.confidence_reason,
    )


def build_reason_candidates(
    evidence: BasilEvidence,
    *,
    fallback_candidates: Optional[Sequence[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> BasilReasonCandidates:
    flat: Dict[str, float] = {}
    for comp_scores in (evidence.reason_scores or {}).values():
        for reason, score in (comp_scores or {}).items():
            reason_s = str(reason or "").strip()
            if not reason_s:
                continue
            flat[reason_s] = max(float(flat.get(reason_s, 0.0)), float(score or 0.0))

    fallback = [str(x) for x in (fallback_candidates or []) if str(x).strip()]
    if logger is not None and fallback:
        basil_set = set(flat.keys())
        fallback_set = set(fallback)
        if basil_set != fallback_set:
            logger.info(
                "BASIL reason candidate diff: "
                f"basil={sorted(basil_set)}, fallback={sorted(fallback_set)}, "
                f"only_basil={sorted(basil_set - fallback_set)}, only_fallback={sorted(fallback_set - basil_set)}"
            )

    if not flat:
        n = max(1, len(fallback))
        flat = {reason: float(1.0 - (i / float(max(1, n - 1)))) if n > 1 else 1.0 for i, reason in enumerate(fallback)}

    ranked = sorted(flat.items(), key=lambda x: float(x[1]), reverse=True)
    return BasilReasonCandidates(
        candidates=[r for r, _ in ranked],
        scores={r: float(s) for r, s in ranked},
        by_component={k: dict(v) for k, v in (evidence.reason_scores or {}).items()},
        fallback_candidates=fallback,
        low_confidence=bool(evidence.low_confidence),
        confidence_reason=evidence.confidence_reason,
    )
