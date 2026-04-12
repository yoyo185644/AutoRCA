"""
Adaptive Layer (rule-based, deterministic).

This module provides a lightweight post-hoc refinement layer for WindowGrangerTrace:
- calibrate component scores/ranking using small, explainable adjustments
- refine reasons using explicit keyword/pattern tables
- mildly refine failure times (optional)

No heavy ML models. No LLM required.

Example
-------
```python
import json
from rca.window_granger_solver import run_window_granger_pipeline
from rca.adaptive_layer import AdaptiveConfig, apply_adaptive_layer

trace = run_window_granger_pipeline(dataset="Telecom", instruction=instr, task_id=task_id)
cfg_obj = json.load(open("rca/adaptive_config.json", "r", encoding="utf-8"))
cfg = AdaptiveConfig.from_dict_for_dataset(cfg_obj, dataset="Telecom")
result = apply_adaptive_layer(trace, config=cfg)

print(result.final_predictions_after_adaptive)
print([r.name for r in result.applied_rules])
```
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from rca.window_granger_solver import WindowGrangerTrace
except Exception:
    WindowGrangerTrace = Any


@dataclass(frozen=True)
class AdaptiveConfig:
    enabled: bool = True
    enable_component_calibration: bool = True
    enable_reason_refine: bool = True
    enable_failure_time_refine: bool = False
    top_k: int = 10
    component_rules: Dict[str, Any] = field(
        default_factory=lambda: {
            "granger_rootness_boost_weight": 0.10,
            "downstream_penalty_weight": 0.08,
            "sink_gamma": 0.5,
            "lead_lag_boost_weight": 0.06,
            "lead_lag_threshold_seconds": 120,
            "confidence_gap_threshold": 0.05,
            "gap_delta_scale": 0.35,
            "prefix_boost": {},
            "apply_to_prediction": True,
        }
    )
    reason_rules: Dict[str, Any] = field(
        default_factory=lambda: {
            "allow_override": False,
            "keyword_map": {},
            "patterns": [],
            "min_pattern_score": 1.0,
            "override_policy": "pattern_only"
        }
    )
    failure_time_rules: Dict[str, Any] = field(
        default_factory=lambda: {
            "merge_close_peaks": True,
            "min_separation_seconds": 300,
            "preserve_if_weak_evidence": True,
            "align_to_component_peak": False,
            "max_adjust_seconds": 300,
            "task5_use_component_peak_time": False
        }
    )

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "AdaptiveConfig":
        """
        Parse a resolved (flat) config dict into AdaptiveConfig.

        For dataset-specific configs (with top-level "datasets"), prefer `from_dict_for_dataset`.
        """
        if not isinstance(obj, Mapping):
            return AdaptiveConfig()
        return AdaptiveConfig(
            enabled=bool(obj.get("enabled", True)),
            enable_component_calibration=bool(obj.get("enable_component_calibration", True)),
            enable_reason_refine=bool(obj.get("enable_reason_refine", True)),
            enable_failure_time_refine=bool(obj.get("enable_failure_time_refine", False)),
            top_k=int(obj.get("top_k", 10)),
            component_rules=dict(obj.get("component_rules", {}) or {}),
            reason_rules=dict(obj.get("reason_rules", {}) or {}),
            failure_time_rules=dict(obj.get("failure_time_rules", {}) or {}),
        )

    @staticmethod
    def from_dict_for_dataset(obj: Mapping[str, Any], dataset: str) -> "AdaptiveConfig":
        """
        Parse a config template which may contain:
        - global / defaults
        - datasets.<DatasetKey> overrides, where DatasetKey is one of: Market, Bank, Telecom
        - optional external rule table paths in reason_rules.rule_table_path
        """
        if not isinstance(obj, Mapping):
            return AdaptiveConfig()

        ds_key = "Market" if str(dataset).startswith("Market/") else str(dataset)
        base = dict(obj.get("global", obj) or {})
        datasets = obj.get("datasets", {})
        override = {}
        if isinstance(datasets, Mapping):
            override = dict(datasets.get(ds_key, {}) or {})

        merged = _deep_merge_dict(base, override)
        rr = dict(merged.get("reason_rules", {}) or {})
        rule_path = rr.get("rule_table_path") or rr.get("rule_table_paths")
        ext = _load_rule_tables(rule_path)
        if ext:
            rr = _deep_merge_dict(ext, rr)
            merged["reason_rules"] = rr
        return AdaptiveConfig.from_dict(merged)


@dataclass(frozen=True)
class AppliedRule:
    name: str
    target: str
    before: Any
    after: Any
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptiveResult:
    trace: WindowGrangerTrace
    config: AdaptiveConfig
    calibrated_component_scores: Dict[str, float]
    ranked_components_after_adaptive: List[str]
    predicted_reasons_after_adaptive: List[str]
    estimated_failure_times_after_adaptive: List[str]
    final_predictions_after_adaptive: str
    applied_rules: List[AppliedRule]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        return out


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _compute_rootness(edges) -> Dict[str, float]:
    out_score: Dict[str, float] = {}
    in_score: Dict[str, float] = {}
    for e in edges or []:
        src = getattr(e, "src", None) if not isinstance(e, dict) else e.get("src")
        tgt = getattr(e, "tgt", None) if not isinstance(e, dict) else e.get("tgt")
        w = getattr(e, "w", None) if not isinstance(e, dict) else e.get("w")
        src = str(src or "")
        tgt = str(tgt or "")
        if not src or not tgt:
            continue
        ww = _safe_float(w, 0.0)
        out_score[src] = out_score.get(src, 0.0) + ww
        in_score[tgt] = in_score.get(tgt, 0.0) + ww
    comps = set(list(out_score.keys()) + list(in_score.keys()))
    rootness = {}
    for c in comps:
        rootness[c] = out_score.get(c, 0.0) - 0.35 * in_score.get(c, 0.0)
    return rootness


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _load_rule_table(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _load_rule_tables(path_or_paths: Any) -> Dict[str, Any]:
    if not path_or_paths:
        return {}
    if isinstance(path_or_paths, str):
        return _load_rule_table(path_or_paths)
    if isinstance(path_or_paths, list):
        merged: Dict[str, Any] = {}
        for p in path_or_paths:
            if not isinstance(p, str):
                continue
            merged = _deep_merge_dict(merged, _load_rule_table(p))
        return merged
    return {}


def _normalize_map(values: Mapping[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    max_abs = max([abs(v) for v in values.values()], default=0.0)
    if max_abs <= 1e-12:
        return {k: 0.0 for k in values.keys()}
    return {k: float(v) / max_abs for k, v in values.items()}


def _graph_degrees(edges) -> Tuple[Dict[str, float], Dict[str, float]]:
    out_score: Dict[str, float] = {}
    in_score: Dict[str, float] = {}
    for e in edges or []:
        src = getattr(e, "src", None) if not isinstance(e, dict) else e.get("src")
        tgt = getattr(e, "tgt", None) if not isinstance(e, dict) else e.get("tgt")
        w = getattr(e, "w", None) if not isinstance(e, dict) else e.get("w")
        src = str(src or "")
        tgt = str(tgt or "")
        if not src or not tgt:
            continue
        ww = _safe_float(w, 0.0)
        out_score[src] = out_score.get(src, 0.0) + ww
        in_score[tgt] = in_score.get(tgt, 0.0) + ww
    return out_score, in_score


def _lead_lag_boost(
    trace: WindowGrangerTrace, ranked_components: Sequence[str], threshold_seconds: int
) -> Dict[str, float]:
    peak_map = getattr(trace, "component_peak_minutes_s", None)
    if not isinstance(peak_map, Mapping):
        return {}
    peaks: Dict[str, int] = {}
    for k, v in peak_map.items():
        try:
            peaks[str(k)] = int(v)
        except Exception:
            continue
    if not peaks:
        return {}
    top = [c for c in ranked_components[:10] if c in peaks]
    if len(top) < 2:
        return {}
    times = [peaks[c] for c in top]
    med = sorted(times)[len(times) // 2]
    out: Dict[str, float] = {}
    for c in top:
        lead = med - peaks[c]
        if lead >= int(threshold_seconds):
            out[c] = float(lead)
    if not out:
        return {}
    max_lead = max(out.values())
    if max_lead <= 0:
        return {}
    return {c: v / max_lead for c, v in out.items()}


def calibrate_component_scores(
    trace: WindowGrangerTrace,
    config: AdaptiveConfig,
    memory: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, float], List[str], List[AppliedRule]]:
    rules: List[AppliedRule] = []
    base_scores = dict(trace.component_anomaly_scores or {})
    if not config.enabled or not config.enable_component_calibration or not base_scores:
        ranked = list(trace.ranked_components_before_adaptive or [])
        return base_scores, ranked, rules

    mem = dict(memory or {})
    prefix_boost = dict(config.component_rules.get("prefix_boost", {}) or {})
    w_root = _safe_float(config.component_rules.get("granger_rootness_boost_weight", 0.10), 0.10)
    w_sink = _safe_float(config.component_rules.get("downstream_penalty_weight", 0.08), 0.08)
    sink_gamma = _safe_float(config.component_rules.get("sink_gamma", 0.5), 0.5)
    w_lead = _safe_float(config.component_rules.get("lead_lag_boost_weight", 0.06), 0.06)
    lead_thresh_s = int(config.component_rules.get("lead_lag_threshold_seconds", 120))
    gap_thr = _safe_float(config.component_rules.get("confidence_gap_threshold", 0.05), 0.05)
    gap_scale = _safe_float(config.component_rules.get("gap_delta_scale", 0.35), 0.35)
    component_bias = dict(mem.get("component_bias", {}) or {})

    out_deg, in_deg = _graph_degrees(trace.granger_edges)
    rootness = {c: out_deg.get(c, 0.0) - 0.35 * in_deg.get(c, 0.0) for c in set(list(out_deg.keys()) + list(in_deg.keys()))}
    sinkness = {c: in_deg.get(c, 0.0) - sink_gamma * out_deg.get(c, 0.0) for c in set(list(out_deg.keys()) + list(in_deg.keys()))}
    root_norm = _normalize_map(rootness)
    sink_norm = _normalize_map(sinkness)

    before_rank = sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)
    before_top = [c for c, _ in before_rank[: max(1, int(config.top_k))]]

    new_scores: Dict[str, float] = {}
    for comp, s0 in base_scores.items():
        comp_s = str(comp)
        s = _safe_float(s0, 0.0)
        if w_root != 0.0:
            s = s * (1.0 + w_root * _safe_float(root_norm.get(comp_s, 0.0), 0.0))
        if w_sink != 0.0 and _safe_float(sink_norm.get(comp_s, 0.0), 0.0) > 0:
            s = s * (1.0 - w_sink * _safe_float(sink_norm.get(comp_s, 0.0), 0.0))
        for pref, b in prefix_boost.items():
            if comp_s.startswith(str(pref)):
                s += _safe_float(b, 0.0)
        if comp_s in component_bias:
            s += _safe_float(component_bias.get(comp_s), 0.0)
        new_scores[comp_s] = float(s)

    ranked = [c for c, _ in sorted(new_scores.items(), key=lambda kv: kv[1], reverse=True)]

    lead_boost = _lead_lag_boost(trace, ranked, threshold_seconds=lead_thresh_s)
    if lead_boost and w_lead != 0.0:
        for c, v in lead_boost.items():
            if c in new_scores:
                new_scores[c] = float(new_scores[c] * (1.0 + w_lead * _safe_float(v, 0.0)))
        ranked = [c for c, _ in sorted(new_scores.items(), key=lambda kv: kv[1], reverse=True)]
        rules.append(
            AppliedRule(
                name="lead_lag_preference",
                target="component_scores",
                before={"top": before_top[:5]},
                after={"top": ranked[:5]},
                details={"lead_lag_threshold_seconds": int(lead_thresh_s), "lead_lag_boost_weight": float(w_lead)},
            )
        )

    base_ranked = [c for c, _ in before_rank]
    if len(ranked) >= 2 and len(base_ranked) >= 2:
        top1, top2 = ranked[0], ranked[1]
        gap = _safe_float(new_scores.get(top1, 0.0)) - _safe_float(new_scores.get(top2, 0.0))
        if gap >= 0 and gap < gap_thr:
            stabilized: Dict[str, float] = {}
            for comp in new_scores.keys():
                s_base = _safe_float(base_scores.get(comp, 0.0), 0.0)
                s_new = _safe_float(new_scores.get(comp, 0.0), 0.0)
                stabilized[comp] = float(s_base + gap_scale * (s_new - s_base))
            new_scores = stabilized
            ranked = [c for c, _ in sorted(new_scores.items(), key=lambda kv: kv[1], reverse=True)]
            rules.append(
                AppliedRule(
                    name="confidence_gap_stabilization",
                    target="component_scores",
                    before={"gap": float(gap), "top2": [top1, top2]},
                    after={"gap": float(_safe_float(new_scores.get(ranked[0], 0.0)) - _safe_float(new_scores.get(ranked[1], 0.0))), "top2": ranked[:2]},
                    details={"threshold": float(gap_thr), "delta_scale": float(gap_scale), "uncertain": True},
                )
            )

    after_top = ranked[: max(1, int(config.top_k))]

    if after_top != before_top:
        rules.append(
            AppliedRule(
                name="component_score_calibration",
                target="component_scores",
                before={"top": before_top},
                after={"top": after_top},
                details={
                    "granger_rootness_boost_weight": w_root,
                    "downstream_penalty_weight": w_sink,
                    "sink_gamma": sink_gamma,
                    "prefix_boost": prefix_boost,
                    "memory_component_bias_keys": sorted(list(component_bias.keys()))[:20],
                },
            )
        )

    return new_scores, ranked, rules


def _keyword_reason(keyword_map: Mapping[str, Sequence[str]], top_metrics: Sequence[str]) -> Optional[str]:
    blob = " ".join([str(x).lower() for x in top_metrics])
    for reason, kws in keyword_map.items():
        if not kws:
            continue
        for k in kws:
            if k and re.search(re.escape(str(k).lower()), blob):
                return str(reason)
    return None


def refine_reasons(
    trace: WindowGrangerTrace,
    ranked_components: List[str],
    config: AdaptiveConfig,
    memory: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[str], List[AppliedRule]]:
    rules: List[AppliedRule] = []
    base = list(trace.predicted_reasons_before_adaptive or [])
    if not config.enabled or not config.enable_reason_refine:
        return base, rules

    allow_override = bool(config.reason_rules.get("allow_override", False))
    override_policy = str(config.reason_rules.get("override_policy", "pattern_only"))
    keyword_map = dict(config.reason_rules.get("keyword_map", {}) or {})
    patterns = list(config.reason_rules.get("patterns", []) or [])
    min_score = _safe_float(config.reason_rules.get("min_pattern_score", 1.0), 1.0)
    if not keyword_map and not patterns:
        return base, rules

    def component_type(comp: str) -> str:
        c = str(comp)
        if c.startswith("node-") or c.startswith("os_"):
            return "node"
        if c.startswith("db_"):
            return "db"
        if c.startswith("docker_") or c.startswith("container"):
            return "container"
        return "unknown"

    def score_patterns(comp: str, top_metrics: Sequence[str]) -> Tuple[str, float, Dict[str, Any]]:
        blob = " ".join([str(x).lower() for x in top_metrics])
        best_reason = ""
        best_score = 0.0
        best_details: Dict[str, Any] = {}
        ctype = component_type(comp)
        for p in patterns:
            if not isinstance(p, Mapping):
                continue
            reason = str(p.get("reason", ""))
            if not reason:
                continue
            prefixes = p.get("component_prefixes")
            if isinstance(prefixes, list) and prefixes:
                if not any(str(comp).startswith(str(pref)) for pref in prefixes):
                    continue
            types = p.get("component_types")
            if isinstance(types, list) and types:
                if ctype not in [str(t) for t in types]:
                    continue
            all_kws = [str(x).lower() for x in (p.get("all") or [])]
            any_kws = [str(x).lower() for x in (p.get("any") or [])]
            none_kws = [str(x).lower() for x in (p.get("none") or [])]
            if any(k and k in blob for k in none_kws):
                continue
            missing = [k for k in all_kws if k and k not in blob]
            if missing:
                continue
            score = 0.0
            score += float(len(all_kws))
            if any_kws:
                score += 0.5 * float(sum(1 for k in any_kws if k and k in blob))
            score += _safe_float(p.get("bias", 0.0), 0.0)
            score += _safe_float(p.get("score_delta", 0.0), 0.0)
            pri = int(p.get("priority", 0))
            score = score + 0.01 * float(pri)
            if score > best_score:
                best_score = score
                best_reason = reason
                best_details = {
                    "pattern": p.get("name", ""),
                    "priority": pri,
                    "score": score,
                    "explanation": p.get("explanation", ""),
                }
        return best_reason, best_score, best_details

    out: List[str] = []
    for i in range(int(trace.k_fail)):
        comp = trace.root_components[i] if i < len(trace.root_components) else (ranked_components[0] if ranked_components else "unknown")
        top_metrics = trace.top_metrics_map.get(comp, [])
        inferred_kw = _keyword_reason(keyword_map, top_metrics) or ""
        inferred_pat, pat_score, pat_details = score_patterns(comp, top_metrics)
        used_pattern = bool(inferred_pat and pat_score >= min_score)
        inferred = inferred_pat if used_pattern else inferred_kw
        if i < len(base):
            can_override = allow_override and (override_policy == "any" or used_pattern)
            if can_override and inferred and inferred != base[i]:
                rules.append(
                    AppliedRule(
                        name="reason_refine",
                        target="reason",
                        before={"idx": i + 1, "reason": base[i], "component": comp},
                        after={"idx": i + 1, "reason": inferred, "component": comp},
                        details={"top_metrics": top_metrics[:10], "pattern": pat_details, "used_pattern": used_pattern},
                    )
                )
                out.append(inferred)
            else:
                out.append(base[i])
        else:
            out.append(inferred or "unknown")

    return out, rules


def refine_failure_times(
    trace: WindowGrangerTrace,
    config: AdaptiveConfig,
    memory: Optional[Mapping[str, Any]] = None,
    components: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[AppliedRule]]:
    rules: List[AppliedRule] = []
    base = list(trace.estimated_failure_times or [])
    if not config.enabled or not config.enable_failure_time_refine:
        return base, rules

    if not base:
        return base, rules
    merge_close = bool(config.failure_time_rules.get("merge_close_peaks", True))
    min_sep = int(config.failure_time_rules.get("min_separation_seconds", 300))
    preserve_weak = bool(config.failure_time_rules.get("preserve_if_weak_evidence", True))
    align_peak = bool(config.failure_time_rules.get("align_to_component_peak", False))
    max_adjust = int(config.failure_time_rules.get("max_adjust_seconds", 300))
    task5_peak = bool(config.failure_time_rules.get("task5_use_component_peak_time", False))

    weak = False
    if preserve_weak:
        if not trace.component_anomaly_scores:
            weak = True
        if getattr(trace, "failure_source", "") == "trace_p95":
            weak = True

    def parse_dt(s: str) -> Optional[datetime]:
        try:
            return datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    dts = [parse_dt(x) for x in base]
    if any(x is None for x in dts):
        return base, rules

    out = list(base)
    if merge_close and len(dts) >= 2:
        for i in range(1, len(dts)):
            if dts[i] is None or dts[i - 1] is None:
                continue
            diff = (dts[i] - dts[i - 1]).total_seconds()
            if diff < 0:
                continue
            if weak:
                if diff == 0:
                    out[i] = out[i - 1]
                    rules.append(
                        AppliedRule(
                            name="merge_duplicate_peaks_weak",
                            target="failure_times",
                            before={"idx": i + 1, "time": base[i]},
                            after={"idx": i + 1, "time": out[i]},
                            details={"diff_seconds": diff},
                        )
                    )
            else:
                if diff < min_sep:
                    out[i] = out[i - 1]
                    rules.append(
                        AppliedRule(
                            name="merge_close_peaks",
                            target="failure_times",
                            before={"idx": i + 1, "time": base[i]},
                            after={"idx": i + 1, "time": out[i]},
                            details={"diff_seconds": diff, "min_separation_seconds": min_sep},
                        )
                    )

    if task5_peak and int(trace.task_id) == 5:
        peak_map = getattr(trace, "component_peak_times", None)
        if isinstance(peak_map, Mapping) and peak_map:
            comps = list(components or trace.root_components or [])
            for i in range(min(len(out), len(comps))):
                comp = str(comps[i])
                peak_s = peak_map.get(comp)
                if isinstance(peak_s, str) and peak_s:
                    before = out[i]
                    out[i] = peak_s
                    rules.append(
                        AppliedRule(
                            name="task5_use_component_peak_time",
                            target="failure_times",
                            before={"idx": i + 1, "time": before, "component": comp},
                            after={"idx": i + 1, "time": out[i], "component": comp},
                            details={"source": "component_peak_times"},
                        )
                    )

    if align_peak and not weak and int(trace.task_id) in (1, 4, 5, 7):
        peak_map = getattr(trace, "component_peak_times", None)
        if isinstance(peak_map, Mapping) and peak_map:
            comps = list(components or trace.root_components or [])
            if comps:
                for i in range(min(len(out), len(comps))):
                    comp = str(comps[i])
                    peak_s = peak_map.get(comp)
                    if not isinstance(peak_s, str) or not peak_s:
                        continue
                    t0 = parse_dt(out[i])
                    tp = parse_dt(peak_s)
                    if t0 is None or tp is None:
                        continue
                    diff = abs((tp - t0).total_seconds())
                    if diff <= max(0, max_adjust):
                        before = out[i]
                        out[i] = peak_s
                        rules.append(
                            AppliedRule(
                                name="align_time_to_component_peak",
                                target="failure_times",
                                before={"idx": i + 1, "time": before, "component": comp},
                                after={"idx": i + 1, "time": out[i], "component": comp},
                                details={"abs_adjust_seconds": float(diff), "max_adjust_seconds": int(max_adjust)},
                            )
                        )

    return out, rules


def _format_prediction(task_id: int, times: List[str], components: List[str], reasons: List[str], k_fail: int) -> str:
    if task_id == 1:
        parts = []
        for idx in range(1, k_fail + 1):
            dt = times[idx - 1] if idx - 1 < len(times) else (times[-1] if times else "")
            parts.append(f"\"{idx}\": {{ \"root cause occurrence datetime\": \"{dt}\" }}")
        return "{ " + ", ".join(parts) + " }"
    if task_id == 2:
        parts = []
        for idx in range(1, k_fail + 1):
            r = reasons[idx - 1] if idx - 1 < len(reasons) else "unknown"
            parts.append(f"\"{idx}\": {{ \"root cause reason\": \"{r}\" }}")
        return "{ " + ", ".join(parts) + " }"

    objs = []
    for i in range(k_fail):
        obj: Dict[str, str] = {}
        if task_id in (1, 4, 5, 7):
            obj["root cause occurrence datetime"] = times[i] if i < len(times) else (times[-1] if times else "")
        if task_id in (3, 5, 6, 7):
            obj["root cause component"] = components[i] if i < len(components) else (components[-1] if components else "unknown")
        if task_id in (2, 4, 6, 7):
            obj["root cause reason"] = reasons[i] if i < len(reasons) else (reasons[-1] if reasons else "unknown")
        objs.append(obj)

    parts = []
    for idx, obj in enumerate(objs, start=1):
        fields = []
        if "root cause occurrence datetime" in obj:
            fields.append(f"\"root cause occurrence datetime\": \"{obj['root cause occurrence datetime']}\"")
        if "root cause component" in obj:
            fields.append(f"\"root cause component\": \"{obj['root cause component']}\"")
        if "root cause reason" in obj:
            fields.append(f"\"root cause reason\": \"{obj['root cause reason']}\"")
        inner = "{ " + ", ".join(fields) + " }"
        parts.append(f"\"{idx}\": {inner}")
    return "{ " + ", ".join(parts) + " }"


def apply_adaptive_layer(
    trace: WindowGrangerTrace,
    config: Optional[AdaptiveConfig] = None,
    memory: Optional[Mapping[str, Any]] = None,
) -> AdaptiveResult:
    cfg = config or AdaptiveConfig()
    applied: List[AppliedRule] = []

    calibrated_scores, ranked_after, rules1 = calibrate_component_scores(trace, cfg, memory=memory)
    applied.extend(rules1)

    reasons_after, rules2 = refine_reasons(trace, ranked_after, cfg, memory=memory)
    applied.extend(rules2)

    components = list(trace.root_components or [])
    if ranked_after:
        components = (ranked_after + ["unknown"] * int(trace.k_fail))[: int(trace.k_fail)]

    times_after, rules3 = refine_failure_times(trace, cfg, memory=memory, components=components)
    applied.extend(rules3)

    pred_after = _format_prediction(
        task_id=int(trace.task_id),
        times=times_after,
        components=components,
        reasons=reasons_after,
        k_fail=int(trace.k_fail),
    )

    return AdaptiveResult(
        trace=trace,
        config=cfg,
        calibrated_component_scores=calibrated_scores,
        ranked_components_after_adaptive=ranked_after[: max(1, int(cfg.top_k))] if ranked_after else [],
        predicted_reasons_after_adaptive=reasons_after,
        estimated_failure_times_after_adaptive=times_after,
        final_predictions_after_adaptive=pred_after,
        applied_rules=applied,
    )
