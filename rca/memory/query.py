from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from rca.component_candidates import _to_epoch_seconds, parse_time_window_generic
from rca.memory.extractor import extract_anomaly_segments, extract_trace_topology_edges
from rca.memory.indexes import AnomalySegmentIndex, CausalPropagationIndex, REASON_TOKENS
from rca.memory.schemas import (
    AnomalySegment,
    BasilComponentResult,
    BasilQueryStats,
    CausalEdgeSketch,
    RCAHypothesis,
    TraceTopologyEdge,
)


@dataclass(frozen=True)
class BasilQueryConfig:
    top_k: int = 12
    bucket_size_s: int = 300
    block_size: int = 32
    max_lag_s: int = 900
    severity_floor: float = 0.0
    w_local: float = 1.0
    w_causal: float = 0.45
    w_reason: float = 0.25
    approximate_epsilon: float = 0.0
    max_trace_rows: Optional[int] = None
    persist_memory: bool = True
    memory_cache_dir: str = "test/cache/basil_index"
    similarity_ts_weight: float = 0.45
    similarity_topology_weight: float = 0.30
    similarity_pattern_weight: float = 0.25
    history_decay_lambda: float = 0.03
    historical_top_k: int = 5
    min_evidence_threshold: int = 2
    task_type: str = "COMPONENT_REASON_TIME"
    rerank_top_n: int = 20
    candidate_reason_top_r: int = 3
    time_tolerance_s: int = 300


def basil_query_config_from_mapping(values: Mapping[str, Any], *, top_k: Optional[int] = None) -> BasilQueryConfig:
    """Build config from a loose caller dict while ignoring unknown keys."""

    allowed = {f.name for f in fields(BasilQueryConfig)}
    data = {str(k): v for k, v in dict(values or {}).items() if str(k) in allowed}
    if top_k is not None:
        data["top_k"] = int(top_k)
    return BasilQueryConfig(**data)


@dataclass(frozen=True)
class ComponentEvidenceSummary:
    component_id: str
    local_score: float
    causal_score: float
    reason_scores: Dict[str, float]
    first_anomaly_time: datetime
    peak_time: datetime
    supporting_metrics: List[str]
    incident_id: str


@dataclass(frozen=True)
class ExpandedCandidate:
    component_id: str
    reason: str = ""
    occurrence_time: Optional[datetime] = None
    score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BasilPipelineResult:
    summaries: List[ComponentEvidenceSummary]
    fused_components: List[Tuple[str, float]]
    expanded_candidates: List[ExpandedCandidate]
    reranked_candidates: List[ExpandedCandidate]
    low_confidence: bool
    timings: Dict[str, float]


def _to_utc_dt(ts_s: int) -> datetime:
    return datetime.fromtimestamp(int(ts_s), tz=timezone.utc)


def _to_local_string(ts_s: int) -> str:
    return pd.to_datetime(int(ts_s), unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize(scores: Mapping[str, float]) -> Dict[str, float]:
    vals = [abs(_safe_float(v)) for v in (scores or {}).values()]
    mx = max(vals + [0.0])
    if mx <= 1e-12:
        return {str(k): 0.0 for k in (scores or {}).keys()}
    return {str(k): max(0.0, _safe_float(v) / mx) for k, v in (scores or {}).items()}


def _reason_score_from_metrics(metrics: Iterable[str], hints: Iterable[str], reason: str) -> float:
    reason_s = str(reason or "")
    if not reason_s:
        return 0.0
    if reason_s in set(str(x) for x in hints or []):
        return 1.0
    toks = REASON_TOKENS.get(reason_s, [])
    if not toks:
        toks = [x for x in reason_s.lower().replace("-", " ").split() if x]
    if not toks:
        return 0.0
    blob = " ".join(str(m).lower() for m in metrics or [])
    hits = sum(1 for t in toks if t and t.lower() in blob)
    return min(1.0, hits / max(1.0, min(3.0, float(len(toks)))))


def _segment_pattern(seg: AnomalySegment) -> str:
    p = str((seg.metadata or {}).get("pattern", "") or "").strip()
    if p:
        return p
    if float(seg.severity or 0.0) >= 2.0:
        return "spike"
    return "drift"


class BasilEvidenceIndex:
    """Thread-safe append-only evidence store plus retrieval utilities."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._segments: Dict[str, List[AnomalySegment]] = {}
        self._edges: Dict[str, List[TraceTopologyEdge]] = {}
        self._incident_times: Dict[str, datetime] = {}

    def append_incident(
        self,
        *,
        incident_id: str,
        segments: Sequence[AnomalySegment],
        edges: Sequence[TraceTopologyEdge],
        incident_time: datetime,
    ) -> None:
        with self._lock:
            if incident_id not in self._segments:
                self._segments[incident_id] = []
            if incident_id not in self._edges:
                self._edges[incident_id] = []
            seen_seg = {s.segment_id for s in self._segments[incident_id]}
            for seg in segments or []:
                if isinstance(seg, AnomalySegment) and seg.segment_id not in seen_seg:
                    self._segments[incident_id].append(seg)
                    seen_seg.add(seg.segment_id)
            seen_edge = {(e.src, e.dst, e.edge_type) for e in self._edges[incident_id]}
            for edge in edges or []:
                if not isinstance(edge, TraceTopologyEdge):
                    continue
                key = (edge.src, edge.dst, edge.edge_type)
                if key not in seen_edge:
                    self._edges[incident_id].append(edge)
                    seen_edge.add(key)
            self._incident_times[incident_id] = incident_time.astimezone(timezone.utc)

    def get_incident(self, incident_id: str) -> Tuple[List[AnomalySegment], List[TraceTopologyEdge]]:
        with self._lock:
            return list(self._segments.get(str(incident_id), [])), list(self._edges.get(str(incident_id), []))

    def retrieve_similar_incidents(
        self,
        *,
        current_incident_id: str,
        current_segments: Sequence[AnomalySegment],
        current_edges: Sequence[TraceTopologyEdge],
        config: BasilQueryConfig,
    ) -> List[Tuple[str, float]]:
        current_components = {s.component for s in current_segments or []}
        current_patterns = {_segment_pattern(s) for s in current_segments or []}
        current_metrics = {s.signal_name for s in current_segments or []}
        now_dt = datetime.now(timezone.utc)
        out: List[Tuple[str, float]] = []
        with self._lock:
            incident_ids = [x for x in self._segments.keys() if x != str(current_incident_id)]
            for incident_id in incident_ids:
                segs = list(self._segments.get(incident_id, []))
                comps = {s.component for s in segs}
                pats = {_segment_pattern(s) for s in segs}
                mets = {s.signal_name for s in segs}
                topo = {(e.src, e.dst) for e in self._edges.get(incident_id, [])}
                cur_topo = {(e.src, e.dst) for e in current_edges or []}

                ts_sim = _jaccard(current_metrics, mets)
                topo_sim = _jaccard(current_components | {x for pair in cur_topo for x in pair}, comps | {x for pair in topo for x in pair})
                pattern_sim = _jaccard(current_patterns, pats)
                sim = (
                    float(config.similarity_ts_weight) * ts_sim
                    + float(config.similarity_topology_weight) * topo_sim
                    + float(config.similarity_pattern_weight) * pattern_sim
                )
                incident_time = self._incident_times.get(incident_id, now_dt)
                days = max(0.0, (now_dt - incident_time).total_seconds() / 86400.0)
                sim *= math.exp(-float(config.history_decay_lambda) * days)
                if sim > 0:
                    out.append((incident_id, float(sim)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[: max(0, int(config.historical_top_k))]


def _jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    aa = {str(x) for x in a or [] if str(x).strip()}
    bb = {str(x) for x in b or [] if str(x).strip()}
    if not aa and not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


class TaskAwareScoreFusion:
    DEFAULT_WEIGHTS = {
        "TIME_ONLY": (0.80, 0.20, 0.00),
        "REASON_ONLY": (0.35, 0.10, 0.55),
        "COMPONENT_ONLY": (0.75, 0.20, 0.05),
        "REASON_TIME": (0.45, 0.15, 0.40),
        "COMPONENT_TIME": (0.65, 0.30, 0.05),
        "COMPONENT_REASON": (0.55, 0.20, 0.25),
        "COMPONENT_REASON_TIME": (0.55, 0.25, 0.20),
    }

    def __init__(self, weights: Optional[Mapping[str, Tuple[float, float, float]]] = None) -> None:
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({str(k): tuple(v) for k, v in weights.items()})

    def fuse(
        self,
        summaries: Sequence[ComponentEvidenceSummary],
        *,
        task_type: str,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        alpha, beta, gamma = self.weights.get(str(task_type), self.weights["COMPONENT_REASON_TIME"])
        scored: List[Tuple[str, float]] = []
        for item in summaries or []:
            reason_max = max([_safe_float(v) for v in item.reason_scores.values()] + [0.0])
            score = float(alpha) * item.local_score + float(beta) * item.causal_score + float(gamma) * reason_max
            scored.append((item.component_id, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, int(top_k))]


class CandidateExpansion:
    def expand(
        self,
        summaries: Sequence[ComponentEvidenceSummary],
        fused_components: Sequence[Tuple[str, float]],
        *,
        task_type: str,
        reason_top_r: int,
    ) -> List[ExpandedCandidate]:
        by_comp = {s.component_id: s for s in summaries or []}
        out: List[ExpandedCandidate] = []
        for comp, score in fused_components or []:
            summary = by_comp.get(comp)
            if summary is None:
                continue
            if task_type == "COMPONENT_ONLY":
                out.append(ExpandedCandidate(component_id=comp, score=float(score), evidence=_summary_evidence(summary)))
                continue
            times = [summary.first_anomaly_time, summary.peak_time]
            times = list(dict.fromkeys([t for t in times if isinstance(t, datetime)]))
            if task_type == "TIME_ONLY":
                for t in times:
                    out.append(ExpandedCandidate(component_id=comp, occurrence_time=t, score=float(score), evidence=_summary_evidence(summary)))
                continue
            if task_type == "COMPONENT_TIME":
                for t in times:
                    out.append(ExpandedCandidate(component_id=comp, occurrence_time=t, score=float(score), evidence=_summary_evidence(summary)))
                continue
            reasons = sorted(summary.reason_scores.items(), key=lambda x: float(x[1]), reverse=True)[: max(1, int(reason_top_r))]
            if not reasons:
                reasons = [("", 0.0)]
            if task_type == "REASON_ONLY":
                for reason, reason_score in reasons:
                    out.append(
                        ExpandedCandidate(
                            component_id=comp,
                            reason=str(reason),
                            score=float(score) + 0.10 * float(reason_score),
                            evidence=_summary_evidence(summary),
                        )
                    )
                continue
            if task_type == "COMPONENT_REASON":
                for reason, reason_score in reasons:
                    out.append(
                        ExpandedCandidate(
                            component_id=comp,
                            reason=str(reason),
                            score=float(score) + 0.10 * float(reason_score),
                            evidence=_summary_evidence(summary),
                        )
                    )
                continue
            for reason, reason_score in reasons:
                for t in times:
                    out.append(
                        ExpandedCandidate(
                            component_id=comp,
                            reason=str(reason),
                            occurrence_time=t,
                            score=float(score) + 0.10 * float(reason_score),
                            evidence=_summary_evidence(summary),
                        )
                    )
        out.sort(key=lambda c: c.score, reverse=True)
        return out


class TwoStageReranker:
    def __init__(self, llm_ranker: Optional[Callable[[List[ExpandedCandidate]], List[ExpandedCandidate]]] = None) -> None:
        self.llm_ranker = llm_ranker

    def rerank(
        self,
        candidates: Sequence[ExpandedCandidate],
        *,
        query_start: datetime,
        query_end: datetime,
        tolerance_s: int,
        top_n: int,
    ) -> List[ExpandedCandidate]:
        kept: List[ExpandedCandidate] = []
        lo = query_start.timestamp() - int(tolerance_s)
        hi = query_end.timestamp() + int(tolerance_s)
        for cand in candidates or []:
            if cand.occurrence_time is not None:
                ts = cand.occurrence_time.timestamp()
                if ts < lo or ts > hi:
                    continue
            score = float(cand.score)
            metrics = [str(x).lower() for x in cand.evidence.get("supporting_metrics", []) or []]
            if cand.reason:
                reason_score = _reason_score_from_metrics(metrics, [], cand.reason)
                score += 0.15 * reason_score
                if reason_score <= 0 and cand.reason != "unknown":
                    score -= 0.10
            if cand.evidence.get("upstream_count", 0) > cand.evidence.get("incoming_count", 0):
                score += 0.10
            kept.append(ExpandedCandidate(cand.component_id, cand.reason, cand.occurrence_time, score, cand.evidence))
        kept.sort(key=lambda c: c.score, reverse=True)
        kept = kept[: max(1, int(top_n))]
        if self.llm_ranker:
            try:
                ranked = self.llm_ranker(list(kept))
                if ranked:
                    return ranked[: max(1, int(top_n))]
            except Exception:
                pass
        return kept


def _summary_evidence(summary: ComponentEvidenceSummary) -> Dict[str, Any]:
    return {
        "local_score": float(summary.local_score),
        "causal_score": float(summary.causal_score),
        "reason_scores": dict(summary.reason_scores),
        "supporting_metrics": list(summary.supporting_metrics),
        "first_anomaly_time": summary.first_anomaly_time.isoformat(),
        "peak_time": summary.peak_time.isoformat(),
    }


def _summaries_from_segments(
    *,
    segments: Sequence[AnomalySegment],
    edges: Sequence[TraceTopologyEdge],
    reasons: Sequence[str],
    start_s: int,
    end_s: int,
    incident_id: str,
    config: BasilQueryConfig,
) -> Tuple[List[ComponentEvidenceSummary], List[CausalEdgeSketch], Dict[str, float], Dict[str, float], Dict[str, float]]:
    seg_index = AnomalySegmentIndex(segments, bucket_size_s=config.bucket_size_s, block_size=config.block_size)
    local_scores = seg_index.local_scores(start_s, end_s)
    metric_scores = seg_index.metric_scores(start_s, end_s)
    trace_scores = seg_index.trace_scores(start_s, end_s)
    affected = [c for c, s in local_scores.items() if float(s) > 0.05]
    causal_index = CausalPropagationIndex(segments, edges, max_lag_s=config.max_lag_s, block_size=config.block_size)
    causal_scores = causal_index.causal_scores(affected_components=affected)

    comps = sorted(set(local_scores.keys()) | set(causal_scores.keys()))
    summaries: List[ComponentEvidenceSummary] = []
    for comp in comps:
        comp_segs = sorted(seg_index.by_component.get(comp, []), key=lambda s: int(s.start_s))
        if not comp_segs:
            continue
        reason_scores = {}
        metric_blob: List[str] = []
        hints: List[str] = []
        for seg in comp_segs:
            metric_blob.extend(seg.top_metrics or [])
            metric_blob.append(seg.signal_name)
            hints.extend(seg.reason_hints or [])
        for reason in reasons or []:
            reason_scores[str(reason)] = _reason_score_from_metrics(metric_blob, hints, str(reason))
        first_s = min(int(s.start_s) for s in comp_segs)
        peak_seg = max(comp_segs, key=lambda s: float(s.severity or 0.0))
        summaries.append(
            ComponentEvidenceSummary(
                component_id=comp,
                local_score=float(local_scores.get(comp, 0.0)),
                causal_score=float(causal_scores.get(comp, 0.0)),
                reason_scores=reason_scores,
                first_anomaly_time=_to_utc_dt(first_s),
                peak_time=_to_utc_dt(int(peak_seg.peak_s)),
                supporting_metrics=list(dict.fromkeys(metric_blob))[:8],
                incident_id=incident_id,
            )
        )
    return summaries, causal_index.edge_sketches, metric_scores, trace_scores, causal_scores


def _merge_historical_boosts(
    summaries: Sequence[ComponentEvidenceSummary],
    historical: Sequence[Tuple[str, float]],
    index: BasilEvidenceIndex,
) -> List[ComponentEvidenceSummary]:
    by_comp = {s.component_id: s for s in summaries or []}
    for incident_id, sim in historical or []:
        segs, edges = index.get_incident(incident_id)
        if not segs:
            continue
        hist_summaries, _, _, _, _ = _summaries_from_segments(
            segments=segs,
            edges=edges,
            reasons=[],
            start_s=min(int(s.start_s) for s in segs),
            end_s=max(int(s.end_s) for s in segs),
            incident_id=incident_id,
            config=BasilQueryConfig(),
        )
        for hist in hist_summaries:
            old = by_comp.get(hist.component_id)
            if old is None:
                by_comp[hist.component_id] = ComponentEvidenceSummary(
                    component_id=hist.component_id,
                    local_score=0.25 * float(sim) * hist.local_score,
                    causal_score=0.25 * float(sim) * hist.causal_score,
                    reason_scores={k: 0.25 * float(sim) * v for k, v in hist.reason_scores.items()},
                    first_anomaly_time=hist.first_anomaly_time,
                    peak_time=hist.peak_time,
                    supporting_metrics=hist.supporting_metrics,
                    incident_id=hist.incident_id,
                )
            else:
                by_comp[hist.component_id] = ComponentEvidenceSummary(
                    component_id=old.component_id,
                    local_score=max(old.local_score, old.local_score + 0.10 * float(sim) * hist.local_score),
                    causal_score=max(old.causal_score, old.causal_score + 0.10 * float(sim) * hist.causal_score),
                    reason_scores={**old.reason_scores, **{k: max(old.reason_scores.get(k, 0.0), 0.10 * float(sim) * v) for k, v in hist.reason_scores.items()}},
                    first_anomaly_time=old.first_anomaly_time,
                    peak_time=old.peak_time,
                    supporting_metrics=list(dict.fromkeys(list(old.supporting_metrics) + list(hist.supporting_metrics)))[:8],
                    incident_id=old.incident_id,
                )
    return list(by_comp.values())


_GLOBAL_EVIDENCE_INDEX = BasilEvidenceIndex()


def query_basil_components(
    dataset: str,
    instruction: str,
    top_k: int = 12,
    center_time_s: Optional[int] = None,
    delta_s: int = 300,
    reasons: Optional[Sequence[str]] = None,
    config: Optional[BasilQueryConfig] = None,
) -> BasilComponentResult:
    """Run the BASIL four-stage online pipeline.

    Stage 1 writes current evidence into a thread-safe append-only index and
    retrieves similar historical incidents. Stages 2-4 fuse, expand, and rerank
    candidates. This function keeps the legacy return type so existing RCA code
    can consume the new pipeline without a second memory implementation.
    """

    t_total = time.perf_counter()
    timings: Dict[str, float] = {}
    cfg = config or BasilQueryConfig(top_k=top_k)
    if int(cfg.top_k) != int(top_k):
        cfg = BasilQueryConfig(**{**cfg.__dict__, "top_k": int(top_k)})

    date_dir, start, end = parse_time_window_generic(instruction)
    start_s, end_s = _to_epoch_seconds(start), _to_epoch_seconds(end)
    incident_id = f"{dataset}:{date_dir}:{start_s}:{end_s}"

    t0 = time.perf_counter()
    segments, ctx = extract_anomaly_segments(
        dataset=dataset,
        instruction=instruction,
        center_time_s=center_time_s,
        delta_s=delta_s,
        severity_floor=cfg.severity_floor,
    )
    timings["basil.evidence_extract_segments_seconds"] = time.perf_counter() - t0

    t_edges = time.perf_counter()
    expanded_start_s = max(0, start_s - int(cfg.max_lag_s))
    expanded_end_s = end_s + int(cfg.max_lag_s)
    edges = extract_trace_topology_edges(
        dataset=dataset,
        date_dir=date_dir,
        start_s=expanded_start_s,
        end_s=expanded_end_s,
        max_rows=cfg.max_trace_rows,
    )
    timings["basil.evidence_extract_topology_seconds"] = time.perf_counter() - t_edges

    t_write = time.perf_counter()
    incident_time = _to_utc_dt(start_s)
    _GLOBAL_EVIDENCE_INDEX.append_incident(
        incident_id=incident_id,
        segments=segments,
        edges=edges,
        incident_time=incident_time,
    )
    timings["basil.evidence_index_write_seconds"] = time.perf_counter() - t_write

    t_query = time.perf_counter()
    historical = _GLOBAL_EVIDENCE_INDEX.retrieve_similar_incidents(
        current_incident_id=incident_id,
        current_segments=segments,
        current_edges=edges,
        config=cfg,
    )
    timings["basil.evidence_index_query_seconds"] = time.perf_counter() - t_query
    low_confidence = len(segments) < int(cfg.min_evidence_threshold)
    timings["basil.stage1_evidence_index_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    summaries, causal_edges, metric_scores, trace_scores, causal_scores = _summaries_from_segments(
        segments=segments,
        edges=edges,
        reasons=reasons or [],
        start_s=int(ctx.get("local_start_s", start_s)),
        end_s=int(ctx.get("local_end_s", end_s)),
        incident_id=incident_id,
        config=cfg,
    )
    summaries = _merge_historical_boosts(summaries, historical, _GLOBAL_EVIDENCE_INDEX)
    fused = TaskAwareScoreFusion().fuse(summaries, task_type=cfg.task_type, top_k=cfg.top_k)
    timings["basil.stage2_task_fusion_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    expanded = CandidateExpansion().expand(
        summaries,
        fused,
        task_type=cfg.task_type,
        reason_top_r=cfg.candidate_reason_top_r,
    )
    timings["basil.stage3_candidate_expansion_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reranked = TwoStageReranker().rerank(
        expanded,
        query_start=start.to_pydatetime().astimezone(timezone.utc),
        query_end=end.to_pydatetime().astimezone(timezone.utc),
        tolerance_s=cfg.time_tolerance_s,
        top_n=cfg.rerank_top_n,
    )
    timings["basil.stage4_two_stage_rerank_seconds"] = time.perf_counter() - t0
    timings["basil.candidate_decision_seconds"] = (
        timings.get("basil.stage2_task_fusion_seconds", 0.0)
        + timings.get("basil.stage3_candidate_expansion_seconds", 0.0)
        + timings.get("basil.stage4_two_stage_rerank_seconds", 0.0)
    )
    timings["basil.total_seconds"] = time.perf_counter() - t_total

    summary_by_comp = {s.component_id: s for s in summaries}
    fused_scores = dict(fused)
    candidates = [comp for comp, _ in fused[: int(cfg.top_k)]]
    first_times = {s.component_id: _to_local_string(int(s.first_anomaly_time.timestamp())) for s in summaries}
    peak_times = {s.component_id: _to_local_string(int(s.peak_time.timestamp())) for s in summaries}
    top_metrics = {s.component_id: list(s.supporting_metrics) for s in summaries}
    reason_scores = {s.component_id: max([float(v) for v in s.reason_scores.values()] + [0.0]) for s in summaries}

    hypotheses: List[RCAHypothesis] = []
    for cand in reranked[: int(cfg.top_k)]:
        summary = summary_by_comp.get(cand.component_id)
        hypotheses.append(
            RCAHypothesis(
                component=cand.component_id,
                reason=cand.reason,
                onset_s=int(cand.occurrence_time.timestamp()) if cand.occurrence_time else None,
                local_score=float(summary.local_score if summary else 0.0),
                causal_score=float(summary.causal_score if summary else 0.0),
                reason_score=float(max((summary.reason_scores or {}).values()) if summary and summary.reason_scores else 0.0),
                score=float(cand.score),
                upper_bound=float(fused_scores.get(cand.component_id, cand.score)),
                evidence={
                    "incident_id": incident_id,
                    "low_confidence": bool(low_confidence),
                    "historical_matches": [{"incident_id": i, "similarity": s} for i, s in historical],
                    "expanded_candidate": {
                        "component_id": cand.component_id,
                        "reason": cand.reason,
                        "occurrence_time": cand.occurrence_time.isoformat() if cand.occurrence_time else "",
                    },
                    **dict(cand.evidence or {}),
                },
            )
        )

    stats = BasilQueryStats(
        total_components=len(summaries),
        total_blocks=0,
        visited_blocks=0,
        refined_components=len(reranked),
        pruned_blocks=max(0, len(expanded) - len(reranked)),
        kth_score=float(reranked[min(len(reranked), int(cfg.top_k)) - 1].score) if reranked else 0.0,
        max_remaining_ub=0.0,
        epsilon=float(cfg.approximate_epsilon),
        exact_mode=True,
        ub_slack={
            "low_confidence": float(bool(low_confidence)),
            "historical_matches": float(len(historical)),
            "segments": float(len(segments)),
            "expanded_candidates": float(len(expanded)),
        },
        timings=timings,
    )

    expl = (
        "BASIL four-stage pipeline: "
        f"segments={len(segments)}, historical={len(historical)}, "
        f"fused={len(fused)}, expanded={len(expanded)}, reranked={len(reranked)}, "
        f"low_confidence={bool(low_confidence)}"
    )

    return BasilComponentResult(
        dataset=dataset,
        date_dir=date_dir,
        start_local=start.strftime("%Y-%m-%d %H:%M:%S"),
        end_local=end.strftime("%Y-%m-%d %H:%M:%S"),
        local_start_s=int(ctx.get("local_start_s", start_s)),
        local_end_s=int(ctx.get("local_end_s", end_s)),
        estimated_failure_time=str(ctx.get("t_star_local", start.strftime("%Y-%m-%d %H:%M:%S"))),
        candidates=candidates,
        metric_scores={k: float(v) for k, v in metric_scores.items()},
        trace_scores={k: float(v) for k, v in trace_scores.items()},
        causal_scores={k: float(v) for k, v in causal_scores.items()},
        reason_scores=reason_scores,
        combined_scores=fused_scores,
        peak_times=peak_times,
        first_times=first_times,
        top_metrics=top_metrics,
        hypotheses=hypotheses,
        causal_edges=causal_edges[: max(50, int(cfg.top_k) * 4)],
        stats=stats,
        explanation=expl,
    )
