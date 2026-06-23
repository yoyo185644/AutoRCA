from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _clamp01(x: Any) -> float:
    v = _safe_float(x, 0.0)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def _to_utc_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _minute_bucket(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return dt.isoformat()

def _minute_bucket_local(dt: Optional[datetime], tz_name: str) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        tz = ZoneInfo(str(tz_name or "UTC"))
    except Exception:
        tz = timezone.utc
    dt = dt.astimezone(tz).replace(second=0, microsecond=0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0
    union = len(a | b)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


@dataclass(frozen=True)
class EvidenceGroup:
    group_id: str
    candidate_component: str
    candidate_reason: str
    candidate_time: str
    candidate_score: float
    supporting_segments: List[Dict[str, Any]] = field(default_factory=list)
    supporting_metrics: List[str] = field(default_factory=list)
    covered_time_windows: List[str] = field(default_factory=list)
    covered_components: List[str] = field(default_factory=list)
    covered_symptom_groups: List[str] = field(default_factory=list)
    memory_evidence: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_score: float = 0.0
    raw_candidates: List[Dict[str, Any]] = field(default_factory=list)
    repeated_component_candidate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_evidence_groups_from_basil(
    *,
    basil_evidence: Any,
    need_reason: bool,
    allow_repeated_component_candidates: bool,
    tz_name: str = "UTC",
    repeated_time_gap_seconds: int = 1800,
    max_time_windows_per_group: int = 6,
) -> List[EvidenceGroup]:
    ev = basil_evidence
    if ev is None:
        return []

    comp_scores = dict(getattr(ev, "component_scores", {}) or {})
    causal_scores = dict(getattr(ev, "causal_scores", {}) or {})
    reason_by_comp = dict(getattr(ev, "reason_scores", {}) or {})
    time_by_comp = dict(getattr(ev, "time_candidates", {}) or {})
    metrics_by_comp = dict(getattr(ev, "supporting_metrics", {}) or {})
    if not comp_scores:
        comps = set()
        comps |= set(str(k) for k in (metrics_by_comp or {}).keys())
        comps |= set(str(k) for k in (reason_by_comp or {}).keys())
        comps |= set(str(k) for k in (time_by_comp or {}).keys())
        try:
            comps |= set(str(k) for k in (getattr(ev, "first_times", {}) or {}).keys())
            comps |= set(str(k) for k in (getattr(ev, "peak_times", {}) or {}).keys())
        except Exception:
            pass
        comp_scores = {str(c): 0.10 for c in comps if str(c).strip()}

    max_comp = max([_safe_float(s, 0.0) for s in comp_scores.values()] + [0.0])
    if max_comp <= 1e-12:
        max_comp = 1.0

    groups: List[EvidenceGroup] = []
    for comp, comp_s in comp_scores.items():
        comp_key = str(comp or "").strip()
        if not comp_key:
            continue
        comp_norm = _clamp01(_safe_float(comp_s, 0.0) / float(max_comp))
        reasons = dict(reason_by_comp.get(comp_key, {}) or {})
        best_reason = ""
        best_reason_norm = 0.0
        if need_reason and reasons:
            max_r = max([_safe_float(s, 0.0) for s in reasons.values()] + [0.0])
            if max_r <= 1e-12:
                max_r = 1.0
            best_reason, best_raw = max(reasons.items(), key=lambda kv: _safe_float(kv[1], 0.0))
            best_reason_norm = _clamp01(_safe_float(best_raw, 0.0) / float(max_r))

        if need_reason:
            cand_score = min(comp_norm, best_reason_norm) if reasons else comp_norm
        else:
            cand_score = comp_norm

        dt_str = str((getattr(ev, "first_times", {}) or {}).get(comp_key) or "").strip()
        dt_utc = _to_utc_dt(dt_str)
        if dt_utc is None:
            times = list(time_by_comp.get(comp_key, []) or [])
            if times:
                dt_utc = _to_utc_dt(times[0])
        cand_time = dt_str
        if not cand_time and dt_utc is not None:
            cand_time = dt_utc.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        buckets: List[str] = []
        extra_times = list(time_by_comp.get(comp_key, []) or [])
        for x in extra_times[: max(0, int(max_time_windows_per_group) - 1)]:
            b = _minute_bucket_local(_to_utc_dt(x), tz_name)
            if b:
                buckets.append(b)
        b0 = _minute_bucket_local(dt_utc, tz_name)
        if b0:
            buckets.append(b0)
        buckets = list(dict.fromkeys([str(x) for x in buckets if str(x).strip()]))
        buckets = buckets[: max(1, int(max_time_windows_per_group))]

        base_raw = {
            "source": "basil",
            "incident_id": str(getattr(ev, "incident_id", "") or ""),
            "component": comp_key,
            "component_score": float(_safe_float(comp_s, 0.0)),
            "component_score_norm": float(comp_norm),
            "causal_score": float(_safe_float(causal_scores.get(comp_key, 0.0), 0.0)),
            "best_reason": str(best_reason or ""),
            "best_reason_score_norm": float(best_reason_norm),
            "time_candidates_count": int(len(extra_times)),
        }

        group_id = f"{comp_key}"
        groups.append(
            EvidenceGroup(
                group_id=group_id,
                candidate_component=comp_key,
                candidate_reason=str(best_reason or ""),
                candidate_time=str(cand_time or ""),
                candidate_score=float(_clamp01(cand_score)),
                supporting_segments=[],
                supporting_metrics=[str(m) for m in list(metrics_by_comp.get(comp_key, []) or [])],
                covered_time_windows=list(buckets),
                covered_components=[comp_key],
                covered_symptom_groups=[str(best_reason or "")] if str(best_reason or "").strip() else [],
                memory_evidence=[
                    {
                        "source": "basil",
                        "low_confidence": bool(getattr(ev, "low_confidence", False)),
                        "confidence_reason": str(getattr(ev, "confidence_reason", "") or ""),
                        "timezone": str(tz_name or "UTC"),
                    }
                ],
                uncertainty_score=1.0 if bool(getattr(ev, "low_confidence", False)) else 0.0,
                raw_candidates=[base_raw],
                repeated_component_candidate=False,
            )
        )

        if allow_repeated_component_candidates and extra_times and len(extra_times) >= 2:
            ts = sorted([t for t in [_to_utc_dt(x) for x in extra_times] if t is not None])
            if ts:
                clusters: List[List[datetime]] = [[ts[0]]]
                for dt in ts[1:]:
                    prev = clusters[-1][-1]
                    gap = float((dt - prev).total_seconds())
                    if gap >= float(repeated_time_gap_seconds):
                        clusters.append([dt])
                    else:
                        clusters[-1].append(dt)
                if len(clusters) >= 2:
                    for j, cluster in enumerate(clusters[:2], start=1):
                        b = _minute_bucket_local(cluster[0], tz_name)
                        if not b:
                            continue
                        gid = f"{comp_key}::repeat{j}::{b}"
                        groups.append(
                            EvidenceGroup(
                                group_id=gid,
                                candidate_component=comp_key,
                                candidate_reason=str(best_reason or ""),
                                candidate_time=str(cand_time or ""),
                                candidate_score=float(_clamp01(cand_score)),
                                supporting_segments=[],
                                supporting_metrics=[str(m) for m in list(metrics_by_comp.get(comp_key, []) or [])],
                                covered_time_windows=[b],
                                covered_components=[comp_key],
                                covered_symptom_groups=[str(best_reason or "")] if str(best_reason or "").strip() else [],
                                memory_evidence=[
                                    {
                                        "source": "basil",
                                        "low_confidence": bool(getattr(ev, "low_confidence", False)),
                                        "confidence_reason": str(getattr(ev, "confidence_reason", "") or ""),
                                        "timezone": str(tz_name or "UTC"),
                                    }
                                ],
                                uncertainty_score=1.0 if bool(getattr(ev, "low_confidence", False)) else 0.0,
                                raw_candidates=[dict(base_raw) | {"repeated_component_cluster": int(j), "cluster_bucket": b}],
                                repeated_component_candidate=True,
                            )
                        )

    return groups


def deduplicate_groups(
    groups: Sequence[EvidenceGroup],
    *,
    target_k: int,
    time_near_seconds: int = 300,
    overlap_threshold: float = 0.85,
) -> List[EvidenceGroup]:
    items = [g for g in list(groups or []) if isinstance(g, EvidenceGroup)]
    if not items:
        return []

    def _t0(g: EvidenceGroup) -> Optional[datetime]:
        return _to_utc_dt(g.candidate_time)

    by_comp: Dict[str, List[EvidenceGroup]] = {}
    for g in items:
        c = str(g.candidate_component or "").strip()
        if not c:
            continue
        by_comp.setdefault(c, []).append(g)

    after_comp: List[EvidenceGroup] = []
    for comp, gs in by_comp.items():
        gs = sorted(gs, key=lambda x: float(x.candidate_score), reverse=True)
        if len(gs) <= 1:
            after_comp.extend(gs)
            continue
        keep: List[EvidenceGroup] = [gs[0]]
        if int(target_k) >= 3:
            repeats = [x for x in gs[1:] if bool(x.repeated_component_candidate)]
            if repeats:
                base_dt = _t0(keep[0])
                for x in repeats:
                    dt = _t0(x)
                    if base_dt is None or dt is None:
                        continue
                    if abs((dt - base_dt).total_seconds()) >= 1800:
                        keep.append(x)
                        break
        after_comp.extend(keep)

    after_comp = sorted(after_comp, key=lambda x: float(x.candidate_score), reverse=True)

    deduped: List[EvidenceGroup] = []
    for g in after_comp:
        g_tw = set([str(x) for x in (g.covered_time_windows or []) if str(x).strip()])
        g_met = set([str(x) for x in (g.supporting_metrics or []) if str(x).strip()])
        g_sym = set([str(x) for x in (g.covered_symptom_groups or []) if str(x).strip()])
        g_dt = _t0(g)
        is_dup = False
        for kept in deduped:
            k_tw = set([str(x) for x in (kept.covered_time_windows or []) if str(x).strip()])
            k_met = set([str(x) for x in (kept.supporting_metrics or []) if str(x).strip()])
            k_sym = set([str(x) for x in (kept.covered_symptom_groups or []) if str(x).strip()])

            overlaps = []
            if g_tw and k_tw:
                overlaps.append(_jaccard(g_tw, k_tw))
            if g_met and k_met:
                overlaps.append(_jaccard(g_met, k_met))
            if g_sym and k_sym:
                overlaps.append(_jaccard(g_sym, k_sym))
            if overlaps and (sum(overlaps) / float(len(overlaps))) >= float(overlap_threshold):
                is_dup = True
                break

            if g_dt is not None:
                k_dt = _t0(kept)
                if k_dt is not None:
                    close = abs((g_dt - k_dt).total_seconds()) <= float(time_near_seconds)
                    if close and str(g.candidate_component) == str(kept.candidate_component):
                        met_overlap = _jaccard(g_met, k_met) if (g_met and k_met) else 0.0
                        if met_overlap >= 0.50:
                            is_dup = True
                            break

        if not is_dup:
            deduped.append(g)
    return deduped


def coverage_gain(selected: Sequence[EvidenceGroup], all_groups: Sequence[EvidenceGroup]) -> float:
    sel = list(selected or [])
    universe = list(all_groups or [])
    if not sel or not universe:
        return 0.0

    def _uniq(attr: str, xs: Sequence[EvidenceGroup]) -> Set[str]:
        out: Set[str] = set()
        for g in xs:
            vals = getattr(g, attr, None)
            if not vals:
                continue
            for v in vals:
                s = str(v or "").strip()
                if s:
                    out.add(s)
        return out

    uni_tw = _uniq("covered_time_windows", universe)
    uni_comp = _uniq("covered_components", universe)
    uni_sym = _uniq("covered_symptom_groups", universe)

    dims: List[Tuple[Set[str], Set[str]]] = []
    sel_tw = _uniq("covered_time_windows", sel)
    sel_comp = _uniq("covered_components", sel)
    sel_sym = _uniq("covered_symptom_groups", sel)
    if uni_tw:
        dims.append((sel_tw, uni_tw))
    if uni_comp:
        dims.append((sel_comp, uni_comp))
    if uni_sym:
        dims.append((sel_sym, uni_sym))
    if not dims:
        return 0.0

    covs = []
    for s, u in dims:
        covs.append(float(len(s)) / float(max(1, len(u))))
    return _clamp01(sum(covs) / float(len(covs)))


def mean_candidate_score(selected: Sequence[EvidenceGroup]) -> float:
    sel = list(selected or [])
    if not sel:
        return 0.0
    vals = [_clamp01(g.candidate_score) for g in sel]
    return _clamp01(sum(vals) / float(len(vals)))


def redundancy_penalty(selected: Sequence[EvidenceGroup]) -> float:
    sel = list(selected or [])
    if len(sel) <= 1:
        return 0.0

    def _set(g: EvidenceGroup, attr: str) -> Set[str]:
        vals = getattr(g, attr, None)
        if not vals:
            return set()
        return set([str(v) for v in vals if str(v).strip()])

    def _segments_set(g: EvidenceGroup) -> Set[str]:
        out: Set[str] = set()
        for s in list(g.supporting_segments or [])[:64]:
            if not isinstance(s, dict):
                continue
            comp = str(s.get("component", "") or "").strip()
            st = str(s.get("start", "") or "").strip()
            ed = str(s.get("end", "") or "").strip()
            key = f"{comp}|{st}|{ed}".strip("|")
            if key and key != "||":
                out.add(key)
        return out

    pairs = list(itertools.combinations(sel, 2))
    if not pairs:
        return 0.0
    overlaps: List[float] = []
    for a, b in pairs:
        dims = []
        sa = _segments_set(a)
        sb = _segments_set(b)
        if sa and sb:
            dims.append(_jaccard(sa, sb))
        for attr in ["covered_time_windows", "covered_symptom_groups", "supporting_metrics"]:
            sa = _set(a, attr)
            sb = _set(b, attr)
            if sa and sb:
                dims.append(_jaccard(sa, sb))
        if dims:
            overlaps.append(sum(dims) / float(len(dims)))
        else:
            overlaps.append(0.0)
    return _clamp01(sum(overlaps) / float(len(overlaps)))


def select_multi_root_set(
    groups: Sequence[EvidenceGroup],
    *,
    target_k: int,
    top_n: int = 10,
) -> Tuple[List[EvidenceGroup], Dict[str, Any]]:
    k = int(target_k) if int(target_k) >= 1 else 1
    top_n = max(1, int(top_n))

    deduped = deduplicate_groups(groups, target_k=k)
    ranked = sorted(deduped, key=lambda g: float(g.candidate_score), reverse=True)
    ranked = ranked[:top_n]

    details: Dict[str, Any] = {
        "target_k": int(k),
        "top_n": int(top_n),
        "input_groups": int(len(list(groups or []))),
        "after_dedup_groups": int(len(deduped)),
        "ranked_groups": int(len(ranked)),
        "insufficient_candidates": False,
        "selected_group_ids": [],
        "best_score_tuple": (0.0, 0.0, 0.0),
    }

    if len(ranked) <= k:
        selected = list(ranked)
        details["insufficient_candidates"] = bool(len(ranked) < k)
        details["selected_group_ids"] = [g.group_id for g in selected]
        return selected, details

    best_set: List[EvidenceGroup] = []
    best_tuple: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    for comb in itertools.combinations(ranked, k):
        S = list(comb)
        tup = (
            float(coverage_gain(S, ranked)),
            float(mean_candidate_score(S)),
            float(-redundancy_penalty(S)),
        )
        if tup > best_tuple:
            best_tuple = tup
            best_set = S

    best_set = sorted(best_set, key=lambda g: float(g.candidate_score), reverse=True)
    details["selected_group_ids"] = [g.group_id for g in best_set]
    details["best_score_tuple"] = tuple(float(x) for x in best_tuple)
    return best_set, details
