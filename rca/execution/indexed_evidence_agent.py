from __future__ import annotations

import importlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from rca.counterfactor_agent.agent import CounterfactualRCAAgent
from rca.common.text import service_base
from rca.common.tasks import required_root_count_from_instruction, task_need_flags
from rca.adapter.registry import get_dataset_adapter
from rca.core import BasilCorePipeline


@dataclass(frozen=True)
class IndexedEvidenceDecision:
    selected_tuples: List[Dict[str, Any]]
    agent_called: bool = False
    llm_called: bool = False
    llm_used: bool = False
    uncertainty_reasons: List[str] = field(default_factory=list)
    candidate_pool_diagnostics: Dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    elapsed_seconds: float = 0.0
    prediction: str = ""
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    prompt: List[Any] = field(default_factory=list)


def _required_root_count(result: Any, instruction: str) -> int:
    evidence = dict(getattr(result, "evidence", {}) or {})
    try:
        count = int(evidence.get("required_root_count") or 0)
    except Exception:
        count = 0
    if count > 0:
        return int(count)
    return max(1, int(required_root_count_from_instruction(str(instruction or ""))))


def _need_flags(task_id: int) -> tuple[bool, bool, bool]:
    return task_need_flags(int(task_id))


def _difficulty(task_id: int) -> str:
    t = int(task_id)
    if t <= 3:
        return "easy"
    if t <= 6:
        return "middle"
    return "hard"


def _task_focus(task_id: int) -> str:
    return {
        1: "Only the root cause occurrence datetime is evaluated.",
        2: "Only the root cause reason is evaluated.",
        3: "Only the root cause component is evaluated.",
        4: "The root cause occurrence datetime and root cause reason are evaluated.",
        5: "The root cause occurrence datetime and root cause component are evaluated.",
        6: "The root cause component and root cause reason are evaluated.",
        7: "The root cause occurrence datetime, component, and reason are all evaluated.",
    }.get(int(task_id), "Follow the instruction and identify the required root cause fields.")


def _unique(items: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items or []:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _load_counterfactual_prompts(dataset: str) -> tuple[str, str]:
    ap = importlib.import_module("rca.baseline.rca_agent.prompt.agent_prompt")
    ds = str(dataset or "")
    if ds == "Bank":
        bp_name = "rca.baseline.rca_agent.prompt.basic_prompt_Bank"
    else:
        bp_name = "rca.baseline.rca_agent.prompt.basic_prompt_Telecom"
    bp = importlib.import_module(bp_name)
    return str(getattr(ap, "rules", "") or ""), str(getattr(bp, "schema", getattr(bp, "cand", "")) or "")


def _safe_prediction_to_tuples(prediction: str, fallback: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(str(prediction or ""))
    except Exception:
        return list(fallback or [])[: max(1, int(k))]
    if not isinstance(obj, Mapping):
        return list(fallback or [])[: max(1, int(k))]

    selected: List[Dict[str, Any]] = []
    for i in range(1, max(1, int(k)) + 1):
        item = obj.get(str(i))
        if not isinstance(item, Mapping):
            continue
        fallback_item = dict(fallback[i - 1]) if i - 1 < len(fallback) else {}
        selected.append(
            {
                "component": str(item.get("root cause component") or fallback_item.get("component") or ""),
                "reason": str(item.get("root cause reason") or fallback_item.get("reason") or ""),
                "datetime": str(item.get("root cause occurrence datetime") or fallback_item.get("datetime") or ""),
                "score": float(fallback_item.get("score", 0.0) or 0.0),
                "source": "counterfactual_agent",
            }
        )
    return selected or list(fallback or [])[: max(1, int(k))]


def _candidate_component_family(component: str) -> str:
    comp = str(component or "").strip().lower()
    if not comp:
        return ""
    if comp.startswith("db_"):
        return "db"
    if comp.startswith(("docker_", "container")):
        return "container"
    if comp.startswith(("os_", "node-")):
        return "node"
    if comp.startswith("redis"):
        return "cache"
    return service_base(comp) or comp


def _candidate_time_cluster(dt: str) -> str:
    text = str(dt or "").strip()
    if not text:
        return ""
    match = re.search(r"(\d{2}):(\d{2})(?::(\d{2}))?", text)
    if not match:
        return text
    hour = int(match.group(1))
    minute = int(match.group(2))
    cluster_minute = (minute // 5) * 5
    return f"{hour:02d}:{cluster_minute:02d}"


def _candidate_epoch_hint(dt: str) -> int:
    text = str(dt or "").strip()
    if not text:
        return 10**18
    try:
        return int(time.mktime(time.strptime(text[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")))
    except Exception:
        pass
    match = re.search(r"(\d{2}):(\d{2})(?::(\d{2}))?", text)
    if not match:
        return 10**18
    try:
        return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3) or 0)
    except Exception:
        return 10**18


def _root_likelihood_order(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Order candidates by root-likelihood without dataset-specific weights.

    The ordering keeps coarse global evidence as the first signal, while using
    channel-local rank and first-anomaly time to break ties inside the same
    score band. Channel preservation expands recall; it should not let a very
    weak channel representative outrank strong current evidence.
    """

    rows = [dict(x) for x in list(items or []) if isinstance(x, Mapping)]
    if not rows:
        return []
    by_channel: Dict[str, List[Dict[str, Any]]] = {}
    for item in rows:
        comp = str(item.get("component", "") or "")
        ch = str(item.get("component_type") or item.get("channel") or _candidate_component_family(comp) or "default")
        by_channel.setdefault(ch, []).append(item)

    channel_rank: Dict[int, int] = {}
    for ch_items in by_channel.values():
        ranked = sorted(ch_items, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        for rank, item in enumerate(ranked, start=1):
            channel_rank[id(item)] = int(rank)

    max_score = max([float(x.get("score", 0.0) or 0.0) for x in rows] + [1.0])

    def _feature(item: Mapping[str, Any], name: str, default: float = 0.0) -> float:
        features = item.get("rootness_features", {})
        if isinstance(features, Mapping) and name in features:
            try:
                return float(features.get(name, default) or default)
            except Exception:
                return float(default)
        try:
            return float(item.get(name, default) or default)
        except Exception:
            return float(default)

    def effective_score(item: Mapping[str, Any]) -> float:
        raw = float(item.get("score", 0.0) or 0.0)
        local_support = max(
            0.0,
            min(
                1.0,
                max(
                    _feature(item, "instance_owned_score"),
                    _feature(item, "ownership"),
                    _feature(item, "locality"),
                ),
            ),
        )
        sibling_contrast = max(0.0, min(1.0, _feature(item, "sibling_contrast")))
        symptomness = max(
            0.0,
            min(
                1.0,
                max(
                    _feature(item, "symptomness"),
                    _feature(item, "incoming_symptomness") * max(0.0, 1.0 - local_support),
                ),
            ),
        )
        # Keep current evidence primary, but prefer candidates whose anomaly is
        # owned by the candidate instance/channel and discount broad downstream
        # symptoms.  This is intentionally based on generic rootness features,
        # not dataset labels or task-specific rules.
        return (
            raw * max(0.20, 1.0 - 0.65 * symptomness)
            + 0.20 * local_support
            + 0.08 * sibling_contrast
        )

    def score_band(item: Mapping[str, Any]) -> int:
        # Bucket scores coarsely so tiny numeric differences do not dominate
        # earlier onset/channel-rank evidence.
        s = effective_score(item) / max(max_score, 1e-9)
        return int(max(0.0, min(1.0, s)) * 10)

    def key(item: Dict[str, Any]) -> tuple:
        dt = str(item.get("datetime", "") or item.get("first_anomaly_time", "") or "")
        symptomness = max(_feature(item, "symptomness"), _feature(item, "incoming_symptomness"))
        local_support = max(
            _feature(item, "instance_owned_score"),
            _feature(item, "ownership"),
            _feature(item, "locality"),
        )
        sibling_contrast = _feature(item, "sibling_contrast")
        return (
            -score_band(item),
            -local_support,
            -sibling_contrast,
            symptomness,
            channel_rank.get(id(item), 10**6),
            _candidate_epoch_hint(dt),
            -effective_score(item),
            -float(item.get("score", 0.0) or 0.0),
        )

    return sorted(rows, key=key)


def _candidate_pool_diagnostics(
    candidates: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    need_time: bool,
    need_comp: bool,
    need_reason: bool,
    preview_n: int = 8,
) -> Dict[str, Any]:
    rows = [dict(x) for x in list(candidates or []) if isinstance(x, Mapping)]

    components = {
        str(x.get("component", "") or "").strip()
        for x in rows
        if str(x.get("component", "") or "").strip()
    }
    reasons = {
        str(x.get("reason", "") or "").strip()
        for x in rows
        if str(x.get("reason", "") or "").strip()
    }
    datetimes = {
        str(x.get("datetime", "") or x.get("occurrence_datetime", "") or "").strip()
        for x in rows
        if str(x.get("datetime", "") or x.get("occurrence_datetime", "") or "").strip()
    }
    time_clusters = {_candidate_time_cluster(x) for x in datetimes if x}
    component_families = {_candidate_component_family(x) for x in components if x}

    preview: List[Dict[str, Any]] = []
    for item in rows[: max(0, int(preview_n))]:
        rootness_features = dict(item.get("rootness_features", {}) or {}) if isinstance(item.get("rootness_features", {}), Mapping) else {}
        preview.append(
            {
                "component": str(item.get("component", "") or ""),
                "reason": str(item.get("reason", "") or ""),
                "datetime": str(item.get("datetime", "") or item.get("occurrence_datetime", "") or ""),
                "first_anomaly_time": str(item.get("first_anomaly_time", "") or ""),
                "component_type": str(item.get("component_type", "") or item.get("channel", "") or ""),
                "supporting_metrics": [str(x) for x in list(item.get("supporting_metrics", []) or [])[:6]],
                "symptomness": float(item.get("symptomness", 0.0) or 0.0),
                "instance_owned_score": float(rootness_features.get("instance_owned_score", 0.0) or 0.0),
                "sibling_contrast": float(rootness_features.get("sibling_contrast", 0.0) or 0.0),
                "causal_source_score": float(rootness_features.get("causal_source_score", 0.0) or 0.0),
                "outgoing_propagation": float(rootness_features.get("outgoing_propagation", 0.0) or 0.0),
                "incoming_symptomness": float(rootness_features.get("incoming_symptomness", 0.0) or 0.0),
                "score": float(item.get("score", 0.0) or 0.0),
                "source": str(item.get("source", "") or ""),
            }
        )

    return {
        "candidate_pool_count": int(len(rows)),
        "candidate_component_count": int(len(components)),
        "candidate_reason_count": int(len(reasons)),
        "candidate_time_count": int(len(datetimes)),
        "candidate_component_family_count": int(len(component_families)),
        "candidate_time_cluster_count": int(len(time_clusters)),
        "llm_max_prompt_candidates": int((context or {}).get("llm_max_prompt_candidates", 0) or 0),
        "needs_component": bool(need_comp),
        "needs_reason": bool(need_reason),
        "needs_time": bool(need_time),
        "candidate_pool_preview": preview,
    }


def _candidate_component_type(item: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    ctype = str(item.get("component_type", "") or item.get("channel", "") or "").strip()
    if ctype:
        return ctype
    comp = str(item.get("component", "") or "").strip()
    adapter = get_dataset_adapter(str((context or {}).get("dataset", "") or ""))
    if adapter is not None and comp:
        try:
            return str(adapter.component_semantic_type(comp) or "")
        except Exception:
            return ""
    return ""


def _reason_family(reason: str) -> str:
    text = str(reason or "").lower()
    if any(tok in text for tok in ("network", "packet", "latency", "loss", "retrans", "corrupt")):
        return "network"
    if "cpu" in text:
        return "cpu"
    if "memory" in text or "mem" in text:
        return "memory"
    if any(tok in text for tok in ("read", "write", "disk", "i/o", "io", "space")):
        return "io"
    if any(tok in text for tok in ("process", "termination", "restart", "kill")):
        return "process"
    return text.strip()


def _tuple_key(item: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "component": str(item.get("component", "") or "").strip(),
        "reason": str(item.get("reason", "") or "").strip(),
        "datetime": str(item.get("datetime", "") or "").strip(),
    }


def _local_evidence_strength(item: Mapping[str, Any], context: Mapping[str, Any]) -> float:
    features = dict(item.get("rootness_features", {}) or {}) if isinstance(item.get("rootness_features", {}), Mapping) else {}
    metrics = [str(x) for x in list(item.get("supporting_metrics", []) or []) if str(x or "").strip()]
    comp = str(item.get("component", "") or "").strip()
    if not metrics:
        metrics = [str(x) for x in list((context or {}).get("top_metrics", {}).get(comp, []) or []) if str(x or "").strip()]
    owned = max(
        float(features.get("instance_owned_score", 0.0) or 0.0),
        float(features.get("ownership", 0.0) or 0.0),
        float(item.get("instance_owned_score", 0.0) or 0.0),
    )
    sibling = float(features.get("sibling_contrast", 0.0) or item.get("sibling_contrast", 0.0) or 0.0)
    metric_bonus = min(1.0, len(metrics) / 6.0)
    symptomness = max(
        float(item.get("symptomness", 0.0) or 0.0),
        float(features.get("incoming_symptomness", 0.0) or 0.0),
    )
    return float(owned + 0.35 * sibling + 0.25 * metric_bonus - 0.35 * symptomness)


def _reason_metric_match_score(item: Mapping[str, Any], context: Mapping[str, Any]) -> float:
    reason = str(item.get("reason", "") or "").strip().lower()
    if not reason:
        return 0.0
    comp = str(item.get("component", "") or "").strip()
    metrics = [str(x).lower() for x in list(item.get("supporting_metrics", []) or []) if str(x or "").strip()]
    metrics.extend(str(x).lower() for x in list((context or {}).get("top_metrics", {}).get(comp, []) or []) if str(x or "").strip())
    blob = " ".join(metrics + [str(item.get("source", "") or "").lower()])
    if not blob:
        return 0.0
    if "cpu" in reason:
        return 1.0 if "cpu" in blob or "load" in blob else 0.0
    if any(tok in reason for tok in ("memory", "mem")):
        return 1.0 if any(tok in blob for tok in ("mem", "memory", "rss", "heap", "gc")) else 0.0
    if any(tok in reason for tok in ("network", "packet", "latency", "loss", "retrans")):
        return 1.0 if any(tok in blob for tok in ("network", "packet", "tcp", "retrans", "latency", "loss")) else 0.0
    if any(tok in reason for tok in ("read", "write", "disk", "io", "i/o")):
        return 1.0 if any(tok in blob for tok in ("read", "write", "disk", "io", "i/o", "fs_")) else 0.0
    return 0.3


def _temporal_change_score(item: Mapping[str, Any], context: Mapping[str, Any], *, need_time: bool) -> float:
    if not bool(need_time):
        return 0.0
    dt = str(item.get("datetime", "") or item.get("occurrence_datetime", "") or "").strip()
    if not dt:
        return 0.0
    source = str(item.get("source", "") or "").lower()
    score = 0.15
    if any(tok in source for tok in ("change_point", "strongest", "first", "onset", "impact")):
        score += 0.45
    if "peak" in source:
        score += 0.25
    comp = str(item.get("component", "") or "").strip()
    first_times = dict((context or {}).get("component_first_times", {}) or {})
    peak_times = dict((context or {}).get("component_peak_times", {}) or {})
    if comp and dt[:16] in {str(first_times.get(comp, ""))[:16], str(peak_times.get(comp, ""))[:16]}:
        score += 0.25
    return float(min(1.0, score))


def _propagation_score(item: Mapping[str, Any]) -> float:
    features = dict(item.get("rootness_features", {}) or {}) if isinstance(item.get("rootness_features", {}), Mapping) else {}
    return float(
        min(
            1.0,
            max(
                float(features.get("causal_source_score", 0.0) or 0.0),
                float(features.get("outgoing_propagation", 0.0) or 0.0),
                float(item.get("causal_source_score", 0.0) or 0.0),
                float(item.get("outgoing_propagation", 0.0) or 0.0),
            ),
        )
    )


def _symptom_risk_score(item: Mapping[str, Any], context: Mapping[str, Any]) -> float:
    features = dict(item.get("rootness_features", {}) or {}) if isinstance(item.get("rootness_features", {}), Mapping) else {}
    risk = max(
        float(item.get("symptomness", 0.0) or 0.0),
        float(features.get("incoming_symptomness", 0.0) or 0.0),
    )
    if _candidate_component_type(item, context) == "node":
        risk = max(risk, 0.35)
    source = str(item.get("source", "") or "").lower()
    if "trace" in source and not item.get("supporting_metrics"):
        risk = max(risk, 0.45)
    return float(min(1.0, risk))


def _intervention_profile(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    need_time: bool,
    need_comp: bool,
    need_reason: bool,
) -> Dict[str, Any]:
    local = max(0.0, min(1.0, _local_evidence_strength(item, context)))
    reason_match = _reason_metric_match_score(item, context) if bool(need_reason) else 0.0
    temporal = _temporal_change_score(item, context, need_time=bool(need_time))
    propagation = _propagation_score(item)
    symptom = _symptom_risk_score(item, context)
    has_required = True
    if bool(need_comp) and not str(item.get("component", "") or "").strip():
        has_required = False
    if bool(need_reason) and not str(item.get("reason", "") or "").strip():
        has_required = False
    if bool(need_time) and not str(item.get("datetime", "") or "").strip():
        has_required = False
    score = 0.38 * local + 0.20 * reason_match + 0.20 * temporal + 0.14 * propagation - 0.24 * symptom
    if not has_required:
        score -= 0.40
    return {
        "component": str(item.get("component", "") or ""),
        "reason": str(item.get("reason", "") or ""),
        "datetime": str(item.get("datetime", "") or ""),
        "component_type": _candidate_component_type(item, context),
        "local_evidence": float(local),
        "reason_metric_match": float(reason_match),
        "temporal_change": float(temporal),
        "propagation_fit": float(propagation),
        "symptom_risk": float(symptom),
        "intervention_score": float(score),
        "has_required_fields": bool(has_required),
    }


def _intervention_challenger_plan(
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    need_time: bool,
    need_comp: bool,
    need_reason: bool,
    max_keys: int = 6,
) -> Dict[str, Any]:
    rows = [dict(x) for x in list(candidates or []) if isinstance(x, Mapping)]
    if len(rows) < 2:
        return {"keys": [], "profiles": [], "call_llm": False, "reason": "not_enough_candidates"}
    top1 = rows[0]
    top1_prof = _intervention_profile(top1, context, need_time=need_time, need_comp=need_comp, need_reason=need_reason)
    try:
        top1_score = float(top1.get("score", 0.0) or 0.0)
    except Exception:
        top1_score = 0.0
    candidates_with_profiles = []
    for item in rows[1: min(len(rows), 24)]:
        prof = _intervention_profile(item, context, need_time=need_time, need_comp=need_comp, need_reason=need_reason)
        if not bool(prof.get("has_required_fields")):
            continue
        gain = float(prof.get("intervention_score", 0.0)) - float(top1_prof.get("intervention_score", 0.0))
        different = False
        if need_comp and str(prof.get("component", "")) != str(top1_prof.get("component", "")):
            different = True
        if need_reason and str(prof.get("reason", "")) != str(top1_prof.get("reason", "")):
            different = True
        if need_time and _candidate_time_cluster(str(prof.get("datetime", ""))) != _candidate_time_cluster(str(top1_prof.get("datetime", ""))):
            different = True
        if not different:
            continue
        try:
            item_score = float(item.get("score", 0.0) or 0.0)
        except Exception:
            item_score = 0.0
        near_score = bool(item_score >= max(0.0, 0.72 * top1_score) or (top1_score - item_score) <= 0.18)
        top1_symptom = bool(float(top1_prof.get("symptom_risk", 0.0) or 0.0) >= 0.35)
        # The challenger must be actionable in at least one required field, not
        # merely different.  This keeps the LLM for cases where there is a real
        # intervention opportunity.
        actionable = False
        if need_comp:
            actionable = actionable or (
                str(prof.get("component", "")) != str(top1_prof.get("component", ""))
                and float(prof.get("local_evidence", 0.0)) >= float(top1_prof.get("local_evidence", 0.0)) + 0.12
                and float(prof.get("symptom_risk", 0.0)) <= float(top1_prof.get("symptom_risk", 0.0)) + 0.10
            )
            actionable = actionable or (
                top1_symptom
                and near_score
                and str(prof.get("component_type", "")) != str(top1_prof.get("component_type", ""))
                and float(prof.get("symptom_risk", 0.0)) <= float(top1_prof.get("symptom_risk", 0.0)) + 0.20
            )
        if need_reason:
            actionable = actionable or (
                str(prof.get("reason", "")) != str(top1_prof.get("reason", ""))
                and float(prof.get("reason_metric_match", 0.0)) >= float(top1_prof.get("reason_metric_match", 0.0)) + 0.25
            )
            actionable = actionable or (
                near_score
                and str(prof.get("reason", "")) != str(top1_prof.get("reason", ""))
                and _reason_family(str(prof.get("reason", ""))) != _reason_family(str(top1_prof.get("reason", "")))
                and float(prof.get("reason_metric_match", 0.0)) >= float(top1_prof.get("reason_metric_match", 0.0))
            )
        if need_time:
            actionable = actionable or (
                _candidate_time_cluster(str(prof.get("datetime", ""))) != _candidate_time_cluster(str(top1_prof.get("datetime", "")))
                and float(prof.get("temporal_change", 0.0)) >= float(top1_prof.get("temporal_change", 0.0)) + 0.20
            )
            actionable = actionable or (
                near_score
                and _candidate_time_cluster(str(prof.get("datetime", ""))) != _candidate_time_cluster(str(top1_prof.get("datetime", "")))
                and float(prof.get("temporal_change", 0.0)) >= max(0.35, float(top1_prof.get("temporal_change", 0.0)) - 0.05)
            )
        if not actionable:
            continue
        prof = dict(prof)
        prof["near_score_challenger"] = bool(near_score)
        prof["top1_symptom_challenge"] = bool(top1_symptom)
        candidates_with_profiles.append((gain, prof, item))

    candidates_with_profiles.sort(
        key=lambda x: (
            float(x[0]),
            float(x[1].get("local_evidence", 0.0)),
            float(x[1].get("reason_metric_match", 0.0)),
            float(x[1].get("temporal_change", 0.0)),
        ),
        reverse=True,
    )
    selected = []
    selected_profiles = []
    seen: set[tuple[str, str, str]] = set()
    for gain, prof, item in candidates_with_profiles:
        if gain < 0.10 and float(prof.get("intervention_score", 0.0)) < 0.40:
            continue
        sig = (str(item.get("component", "") or ""), str(item.get("reason", "") or ""), str(item.get("datetime", "") or ""))
        if sig in seen:
            continue
        selected.append(_tuple_key(item))
        prof2 = dict(prof)
        prof2["gain_vs_top1"] = float(gain)
        selected_profiles.append(prof2)
        seen.add(sig)
        if len(selected) >= max(1, int(max_keys) - 1):
            break
    call_llm = bool(selected)
    keys = [_tuple_key(top1)] + selected if call_llm else []
    return {
        "keys": keys,
        "profiles": [top1_prof] + selected_profiles if call_llm else [top1_prof],
        "call_llm": bool(call_llm),
        "reason": "actionable_intervention_challenger" if call_llm else "no_actionable_intervention_challenger",
        "top1_profile": dict(top1_prof),
    }


def _attach_topology_context_to_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    adapter: Any,
    context: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in list(rows or []):
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        if not isinstance(item.get("topology_context", {}), Mapping):
            item.pop("topology_context", None)
        if not item.get("topology_context") and adapter is not None:
            comp = str(item.get("component", "") or "").strip()
            if comp:
                try:
                    topo = dict(adapter.topology_context_for_component(component=comp, context=context) or {})
                except Exception:
                    topo = {}
                if topo:
                    item["topology_context"] = topo
        out.append(item)
    return out


def _coverage_aware_challenger_keys(
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    need_time: bool,
    need_comp: bool,
    need_reason: bool,
    max_keys: int = 6,
) -> Dict[str, Any]:
    """Select representative challengers from observed evidence dimensions.

    This does not use labels.  It asks whether the current prompt candidates
    cover component types, reason families, and time clusters that are present
    in indexed evidence.  Missing dimensions produce retrieval/rerank keys for
    the LLM prompt.
    """
    rows = [dict(x) for x in list(candidates or []) if isinstance(x, Mapping)]
    if not rows:
        return {"keys": [], "coverage": {"complete": False, "reason": "empty_pool"}}

    adapter = get_dataset_adapter(str((context or {}).get("dataset", "") or ""))
    observed_types: set[str] = set()
    if need_comp:
        for comp in list((context or {}).get("allowed_components") or []):
            c = str(comp or "").strip()
            if not c:
                continue
            if adapter is not None:
                try:
                    ct = str(adapter.component_semantic_type(c) or "")
                except Exception:
                    ct = ""
            else:
                ct = ""
            if ct:
                observed_types.add(ct)
        for comp in list(((context or {}).get("top_metrics") or {}).keys()):
            c = str(comp or "").strip()
            if not c or adapter is None:
                continue
            try:
                ct = str(adapter.component_semantic_type(c) or "")
            except Exception:
                ct = ""
            if ct:
                observed_types.add(ct)

    observed_reason_families: set[str] = set()
    if need_reason:
        for reason in list((context or {}).get("allowed_reasons") or []):
            fam = _reason_family(str(reason or ""))
            if fam:
                observed_reason_families.add(fam)
        for reason, score in dict((context or {}).get("reason_scores") or {}).items():
            try:
                if float(score or 0.0) <= 0:
                    continue
            except Exception:
                pass
            fam = _reason_family(str(reason or ""))
            if fam:
                observed_reason_families.add(fam)

    observed_time_clusters: set[str] = set()
    if need_time:
        for item in list((context or {}).get("time_candidates") or []):
            if not isinstance(item, Mapping):
                continue
            cluster = _candidate_time_cluster(str(item.get("datetime", "") or ""))
            if cluster:
                observed_time_clusters.add(cluster)

    prompt_window = rows[: min(len(rows), 8)]
    candidate_types = {_candidate_component_type(x, context) for x in prompt_window if _candidate_component_type(x, context)}
    candidate_reason_families = {_reason_family(str(x.get("reason", "") or "")) for x in prompt_window if str(x.get("reason", "") or "").strip()}
    candidate_time_clusters = {_candidate_time_cluster(str(x.get("datetime", "") or "")) for x in prompt_window if str(x.get("datetime", "") or "").strip()}

    missing_types = sorted(observed_types - candidate_types)
    missing_reasons = sorted(observed_reason_families - candidate_reason_families)
    missing_times = sorted(observed_time_clusters - candidate_time_clusters)

    selected: List[Dict[str, str]] = []
    selected_sigs: set[tuple[str, str, str]] = set()

    def add(item: Mapping[str, Any]) -> None:
        if len(selected) >= max(1, int(max_keys)):
            return
        key = _tuple_key(item)
        sig = (key["component"], key["reason"], key["datetime"])
        if sig in selected_sigs:
            return
        selected.append(key)
        selected_sigs.add(sig)

    if rows:
        add(rows[0])

    for mt in missing_types:
        best = max(
            (x for x in rows if _candidate_component_type(x, context) == mt),
            key=lambda x: (_local_evidence_strength(x, context), float(x.get("score", 0.0) or 0.0)),
            default=None,
        )
        if best is not None:
            add(best)

    for mr in missing_reasons:
        best = max(
            (x for x in rows if _reason_family(str(x.get("reason", "") or "")) == mr),
            key=lambda x: (_local_evidence_strength(x, context), float(x.get("score", 0.0) or 0.0)),
            default=None,
        )
        if best is not None:
            add(best)

    for mtc in missing_times[: max(0, int(max_keys) - len(selected))]:
        best = max(
            (x for x in rows if _candidate_time_cluster(str(x.get("datetime", "") or "")) == mtc),
            key=lambda x: (_local_evidence_strength(x, context), float(x.get("score", 0.0) or 0.0)),
            default=None,
        )
        if best is not None:
            add(best)

    # Always include the strongest local non-node challenger when top1 is a
    # possible infrastructure symptom and a local service/container candidate
    # exists.  This is the common case where deterministic rankers over-trust
    # broad node evidence.
    top_type = _candidate_component_type(rows[0], context) if rows else ""
    if need_comp and top_type == "node":
        best_local = max(
            (x for x in rows if _candidate_component_type(x, context) in {"container", "service", "pod"}),
            key=lambda x: (_local_evidence_strength(x, context), float(x.get("score", 0.0) or 0.0)),
            default=None,
        )
        if best_local is not None:
            add(best_local)

    for item in rows:
        if len(selected) >= max(1, int(max_keys)):
            break
        add(item)

    coverage = {
        "observed_component_types": sorted(observed_types),
        "candidate_component_types": sorted(candidate_types),
        "missing_component_types": missing_types,
        "observed_reason_families": sorted(observed_reason_families),
        "candidate_reason_families": sorted(candidate_reason_families),
        "missing_reason_families": missing_reasons,
        "observed_time_clusters": len(observed_time_clusters),
        "candidate_time_clusters": len(candidate_time_clusters),
        "missing_time_cluster_count": len(missing_times),
        "complete": not (missing_types or missing_reasons or missing_times),
    }
    return {"keys": selected, "coverage": coverage}


def _llm_only_challenger_expansion(
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    need_time: bool,
    need_comp: bool,
    need_reason: bool,
    max_extra: int = 16,
) -> List[Dict[str, Any]]:
    """Build label-free challenger tuples for the LLM prompt only.

    The deterministic ranker may be conservative and omit low-ranked reason
    families.  For LLM reranking, however, a challenger is useful only if it
    covers a genuinely different component/reason/time dimension.  This helper
    combines observed components with adapter taxonomy reasons using runtime
    metric support.  It does not read query labels or historical records, and
    the returned rows are not used for deterministic output.
    """
    rows = [dict(x) for x in list(candidates or []) if isinstance(x, Mapping)]
    if not rows:
        return []
    adapter = get_dataset_adapter(str((context or {}).get("dataset", "") or ""))
    prompt_limit = max(1, int((context or {}).get("llm_max_prompt_candidates", 6) or 6))
    visible = rows[: min(len(rows), prompt_limit)]
    visible_reason_families = {
        _reason_family(str(x.get("reason", "") or ""))
        for x in visible
        if str(x.get("reason", "") or "").strip()
    }
    visible_component_types = {
        _candidate_component_type(x, context)
        for x in visible
        if _candidate_component_type(x, context)
    }

    reasons: List[str] = []
    if need_reason:
        for r in list((context or {}).get("allowed_reasons") or []):
            rr = str(r or "").strip()
            if rr and rr not in reasons:
                reasons.append(rr)
        if adapter is not None:
            try:
                for r in list(adapter.basil_reason_taxonomy() or []):
                    rr = str(r or "").strip()
                    if rr and rr not in reasons:
                        reasons.append(rr)
            except Exception:
                pass

    component_rows: Dict[str, Dict[str, Any]] = {}
    for item in rows[: max(24, prompt_limit * 4)]:
        comp = str(item.get("component", "") or "").strip()
        if not comp:
            continue
        current = component_rows.get(comp)
        if current is None or _local_evidence_strength(item, context) > _local_evidence_strength(current, context):
            component_rows[comp] = dict(item)
    for comp in list((context or {}).get("allowed_components") or []):
        c = str(comp or "").strip()
        if c and c not in component_rows:
            component_rows[c] = {
                "component": c,
                "reason": "",
                "datetime": "",
                "score": float(dict((context or {}).get("component_scores") or {}).get(c, 0.0) or 0.0),
                "source": "llm_only_allowed_component",
                "supporting_metrics": list(dict((context or {}).get("top_metrics") or {}).get(c, []) or [])[:6],
            }

    time_by_component: Dict[str, str] = {}
    if need_time:
        for item in rows:
            comp = str(item.get("component", "") or "").strip()
            dt = str(item.get("datetime", "") or "").strip()
            if comp and dt and comp not in time_by_component:
                time_by_component[comp] = dt
        for item in list((context or {}).get("time_candidates") or []):
            if not isinstance(item, Mapping):
                continue
            comp = str(item.get("component", "") or "").strip()
            dt = str(item.get("datetime", "") or "").strip()
            if comp and dt and comp not in time_by_component:
                time_by_component[comp] = dt

    existing = {
        (
            str(x.get("component", "") or ""),
            str(x.get("reason", "") or ""),
            str(x.get("datetime", "") or ""),
        )
        for x in rows
    }
    extras: List[Dict[str, Any]] = []

    if need_reason and reasons:
        if not need_comp:
            reason_scores = dict((context or {}).get("reason_scores") or {})
            existing_reasons = {
                str(x.get("reason", "") or "").strip()
                for x in rows
                if str(x.get("reason", "") or "").strip()
            }
            existing_reason_families = {_reason_family(r) for r in existing_reasons if r}
            family_best: Dict[str, Dict[str, Any]] = {}
            family_variants: List[Dict[str, Any]] = []
            for reason in reasons:
                fam = _reason_family(reason)
                if not fam:
                    continue
                score = float(reason_scores.get(reason, 0.0) or 0.0)
                item = {
                    "component": "",
                    "reason": reason,
                    "datetime": str((context or {}).get("time_choice", "") or "") if need_time else "",
                    "score": max(0.01, score),
                    "source": "llm_only_reason_taxonomy_family_challenger",
                    "llm_only_challenger": True,
                    "llm_challenger_reason": "reason_family_taxonomy_coverage",
                }
                if fam not in existing_reason_families:
                    prev = family_best.get(fam)
                    if prev is None or float(prev.get("score", 0.0) or 0.0) < float(item.get("score", 0.0) or 0.0):
                        family_best[fam] = item
                elif reason not in existing_reasons:
                    item["source"] = "llm_only_reason_taxonomy_variant_challenger"
                    item["llm_challenger_reason"] = "reason_variant_coverage"
                    family_variants.append(item)
            extras.extend(family_best.values())
            family_variants.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            # Keep same-family variants too.  Many RCA labels differ inside a
            # family (read vs write I/O, packet loss vs corruption).  A family
            # level challenger is not enough for reason-only and reason-time
            # tasks, because evaluation is exact-label based.
            extras.extend(family_variants[: max(2, prompt_limit)])

        for comp, base in component_rows.items():
            metrics = [str(x) for x in list(base.get("supporting_metrics", []) or []) if str(x or "").strip()]
            if not metrics:
                metrics = [str(x) for x in list(dict((context or {}).get("top_metrics") or {}).get(comp, []) or []) if str(x or "").strip()]
            for reason in reasons:
                fam = _reason_family(reason)
                support = 0.0
                if adapter is not None:
                    try:
                        support = float(adapter._reason_metric_support_score(component=comp, reason=reason, metrics=metrics))  # noqa: SLF001
                    except Exception:
                        support = 0.0
                if support <= 0.0:
                    probe = dict(base)
                    probe["reason"] = reason
                    support = _reason_metric_match_score(probe, context)
                # Always keep one representative for a missing family when it
                # has any concrete metric support; also keep strong same-family
                # variants because exact RCA reason labels often differ within
                # one broad family.
                if fam in visible_reason_families and support < 0.35:
                    continue
                if support < 0.12:
                    continue
                dt = time_by_component.get(comp, str(base.get("datetime", "") or "")) if need_time else ""
                sig = (comp, reason, dt)
                if sig in existing:
                    continue
                item = dict(base)
                item.update(
                    {
                        "component": comp,
                        "reason": reason,
                        "datetime": dt,
                        "score": float(max(float(base.get("score", 0.0) or 0.0), 0.01) + 0.25 * support),
                        "source": f"{base.get('source', 'basil_tuple')}+llm_only_reason_family_challenger",
                        "supporting_metrics": metrics[:6],
                        "llm_only_challenger": True,
                        "llm_challenger_reason": "reason_family_coverage",
                    }
                )
                extras.append(item)

    if need_comp:
        for comp, base in component_rows.items():
            ctype = _candidate_component_type(base, context)
            if ctype and ctype not in visible_component_types:
                dt = time_by_component.get(comp, str(base.get("datetime", "") or "")) if need_time else str(base.get("datetime", "") or "")
                reason = str(base.get("reason", "") or "")
                sig = (comp, reason, dt)
                if sig in existing:
                    continue
                item = dict(base)
                item.update(
                    {
                        "datetime": dt,
                        "source": f"{base.get('source', 'basil_tuple')}+llm_only_component_type_challenger",
                        "llm_only_challenger": True,
                        "llm_challenger_reason": "component_type_coverage",
                    }
                )
                extras.append(item)

    if need_time:
        visible_clusters = {
            _candidate_time_cluster(str(x.get("datetime", "") or ""))
            for x in visible
            if str(x.get("datetime", "") or "").strip()
        }
        for item in list((context or {}).get("time_candidates") or []):
            if not isinstance(item, Mapping):
                continue
            dt = str(item.get("datetime", "") or "").strip()
            comp = str(item.get("component", "") or "").strip()
            if not dt or (need_comp and not comp):
                continue
            cluster = _candidate_time_cluster(dt)
            if cluster in visible_clusters:
                continue
            base = dict(component_rows.get(comp, {}))
            if not base and comp:
                base = {"component": comp, "supporting_metrics": list(dict((context or {}).get("top_metrics") or {}).get(comp, []) or [])[:6]}
            reason = str(base.get("reason", "") or "")
            sig = (comp, reason, dt)
            if sig in existing:
                continue
            base.update(
                {
                    "component": comp,
                    "reason": reason,
                    "datetime": dt,
                    "score": float(item.get("score", base.get("score", 0.0)) or 0.0),
                    "source": f"{item.get('source', 'indexed_time_candidate')}+llm_only_time_cluster_challenger",
                    "llm_only_challenger": True,
                    "llm_challenger_reason": "time_cluster_coverage",
                }
            )
            extras.append(base)

    extras.sort(
        key=lambda x: (
            _reason_metric_match_score(x, context),
            _local_evidence_strength(x, context),
            float(x.get("score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in extras:
        sig = (
            str(item.get("component", "") or ""),
            str(item.get("reason", "") or ""),
            str(item.get("datetime", "") or ""),
        )
        if sig in existing or sig in seen:
            continue
        out.append(dict(item))
        seen.add(sig)
        if len(out) >= max(0, int(max_extra)):
            break
    return out


def _source_flags(item: Mapping[str, Any], context: Mapping[str, Any]) -> Dict[str, bool]:
    comp = str(item.get("component", "") or "").strip()
    source_blob = " ".join(
        [
            str(item.get("source", "") or ""),
            " ".join(str(x) for x in list(item.get("supporting_metrics", []) or [])),
            " ".join(str(x) for x in list((context or {}).get("top_metrics", {}).get(comp, []) or [])),
        ]
    ).lower()
    metric = bool(comp and (item.get("supporting_metrics") or (context or {}).get("top_metrics", {}).get(comp)))
    trace = any(tok in source_blob for tok in ("trace", "span", "p95", "duration", "latency"))
    log = any(tok in source_blob for tok in ("log", "event", "exception", "error"))
    return {"metric": bool(metric), "trace": bool(trace), "log": bool(log)}


def _reason_supported_by_metrics(item: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    reason = str(item.get("reason", "") or "").strip()
    if not reason:
        return True
    comp = str(item.get("component", "") or "").strip()
    adapter = get_dataset_adapter(str((context or {}).get("dataset", "") or ""))
    metrics = [str(x).lower() for x in list(item.get("supporting_metrics", []) or [])]
    metrics.extend(str(x).lower() for x in list((context or {}).get("top_metrics", {}).get(comp, []) or []))
    if not metrics:
        return False
    if adapter is not None:
        try:
            keywords = [str(x).lower() for x in list(adapter.reason_metric_keywords(reason) or []) if str(x).strip()]
            if keywords:
                blob = " ".join(metrics)
                return any(k in blob for k in keywords)
        except Exception:
            pass
    return True


def _timeline_supported(item: Mapping[str, Any], context: Mapping[str, Any], *, need_time: bool) -> bool:
    if not bool(need_time):
        return True
    dt = str(item.get("datetime", "") or item.get("occurrence_datetime", "") or "").strip()
    if not dt:
        return False
    first = str(item.get("first_anomaly_time", "") or "").strip()
    source = str(item.get("source", "") or "").lower()
    if first and dt[:16] == first[:16]:
        return True
    if any(tok in source for tok in ("first", "onset", "impact", "peak", "trace", "metric")):
        return True
    comp = str(item.get("component", "") or "").strip()
    first_times = dict((context or {}).get("component_first_times", {}) or {})
    peak_times = dict((context or {}).get("component_peak_times", {}) or {})
    return dt[:16] in {str(first_times.get(comp, ""))[:16], str(peak_times.get(comp, ""))[:16]}


def _evidence_consistency_gate(
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    need_time: bool,
    need_comp: bool,
    need_reason: bool,
) -> Dict[str, Any]:
    """Accept top-1 without LLM when independent evidence sources agree.

    This is deliberately conservative: it only short-circuits when the selected
    tuple has the required fields, metric evidence, trace/propagation evidence,
    reason-metric compatibility, timeline compatibility, and no close competing
    candidate. Log evidence is required only when log/event candidates are
    actually present in the pool.
    """

    rows = [dict(x) for x in list(candidates or []) if isinstance(x, Mapping)]
    if not rows:
        return {"accept": False, "reasons": ["no_candidates_for_consistency_gate"]}
    top1 = rows[0]
    if need_comp and not str(top1.get("component", "") or "").strip():
        return {"accept": False, "reasons": ["missing_component"]}
    if need_reason and not str(top1.get("reason", "") or "").strip():
        return {"accept": False, "reasons": ["missing_reason"]}
    if need_time and not str(top1.get("datetime", "") or top1.get("occurrence_datetime", "") or "").strip():
        return {"accept": False, "reasons": ["missing_time"]}

    flags = _source_flags(top1, context)
    pool_has_log = any(_source_flags(item, context).get("log") for item in rows[:12])
    required_sources = ["metric", "trace"] + (["log"] if pool_has_log else [])
    missing_sources = [src for src in required_sources if not flags.get(src)]
    reasons: List[str] = []
    if missing_sources:
        reasons.append("missing_sources:" + ",".join(missing_sources))
    if not _reason_supported_by_metrics(top1, context):
        reasons.append("reason_metric_mismatch")
    if not _timeline_supported(top1, context, need_time=bool(need_time)):
        reasons.append("timeline_not_supported")

    try:
        top1_score = float(top1.get("score", 0.0) or 0.0)
    except Exception:
        top1_score = 0.0
    top2_score = 0.0
    if len(rows) >= 2:
        try:
            top2_score = float(rows[1].get("score", 0.0) or 0.0)
        except Exception:
            top2_score = 0.0
    score_gap = max(0.0, top1_score - top2_score)
    if len(rows) >= 2 and score_gap < 0.20:
        reasons.append("close_competing_candidate")

    try:
        symptomness = float(top1.get("symptomness", 0.0) or 0.0)
    except Exception:
        symptomness = 0.0
    if symptomness >= 0.65:
        reasons.append("top1_frequent_symptom")

    accept = not reasons and flags.get("metric") and flags.get("trace")
    return {
        "accept": bool(accept),
        "reasons": list(dict.fromkeys(reasons)),
        "candidate": dict(top1),
        "source_flags": dict(flags),
        "required_sources": list(required_sources),
        "top1_score": float(top1_score),
        "top2_score": float(top2_score),
        "score_gap": float(score_gap),
    }


def _safe_prediction_to_tuples_constrained(
    prediction: str,
    fallback_pool: List[Dict[str, Any]],
    k: int,
    *,
    task_id: int,
    dataset: str = "",
    adapter: Any = None,
    context: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Bounded LLM adoption over BASIL candidates.

    The LLM may reorder or choose candidates already proposed by BASIL. It may
    not introduce unseen components, reasons, or timestamps. Dataset adapters
    only canonicalize fields and validate semantic compatibility.
    """
    pool = [dict(x) for x in list(fallback_pool or []) if isinstance(x, Mapping)]
    limit = max(1, int(k))
    deterministic = pool[:limit]
    if not pool:
        return deterministic

    constrain = True
    if adapter is not None:
        try:
            constrain = bool(adapter.constrain_llm_to_index_candidates())
        except Exception:
            constrain = True
    if not constrain:
        return _safe_prediction_to_tuples(prediction, deterministic, limit)

    try:
        obj = json.loads(str(prediction or ""))
    except Exception:
        return deterministic
    if not isinstance(obj, Mapping):
        return deterministic

    need_time, need_comp, need_reason = _need_flags(int(task_id))

    def canon(comp: str, reason: str, dt: str) -> tuple[str, str, str]:
        c = str(comp or "").strip()
        r = str(reason or "").strip()
        d = str(dt or "").strip()
        if adapter is not None:
            try:
                c = str(adapter.parse_component_id(c) or c).strip()
            except Exception:
                pass
            try:
                r = str(adapter.parse_root_cause_label(r) or r).strip()
            except Exception:
                pass
        return c, r, d

    pool_keys: List[tuple[str, str, str, Dict[str, Any]]] = []
    for item in pool:
        c, r, d = canon(
            str(item.get("component") or ""),
            str(item.get("reason") or ""),
            str(item.get("datetime") or ""),
        )
        pool_keys.append((c, r, d, item))

    best_score = max([float(item.get("score", 0.0) or 0.0) for item in pool] + [0.0])
    score_guard = float((context or {}).get("llm_max_score_drop_vs_det_top1", 0.05) or 0.05)
    selected: List[Dict[str, Any]] = []
    used_idx: set[int] = set()

    for i in range(1, limit + 1):
        item = obj.get(str(i))
        if not isinstance(item, Mapping):
            continue
        pred_comp, pred_reason, pred_dt = canon(
            str(item.get("root cause component") or item.get("component") or ""),
            str(item.get("root cause reason") or item.get("reason") or ""),
            str(item.get("root cause occurrence datetime") or item.get("datetime") or ""),
        )

        best_idx = -1
        best_match_score = -1
        for j, (cand_comp, cand_reason, cand_dt, _cand) in enumerate(pool_keys):
            if j in used_idx:
                continue
            match_score = 0
            if need_comp:
                if pred_comp and cand_comp and pred_comp == cand_comp:
                    match_score += 4
                else:
                    continue
            if need_reason:
                if pred_reason and cand_reason and pred_reason == cand_reason:
                    match_score += 3
                else:
                    continue
            if need_time:
                if pred_dt and cand_dt and pred_dt == cand_dt:
                    match_score += 2
                elif not need_comp and not need_reason:
                    continue
            if match_score > best_match_score:
                best_match_score = match_score
                best_idx = j
        if best_idx < 0:
            continue

        matched = dict(pool_keys[best_idx][3])
        matched_score = float(matched.get("score", 0.0) or 0.0)
        reason_only_choice = bool(need_reason) and not bool(need_comp) and not bool(need_time)
        if (not reason_only_choice) and best_score > 0 and matched_score + max(score_guard, 0.0) < best_score:
            continue
        if adapter is not None:
            det_item = deterministic[0] if deterministic else {}
            try:
                ok, _why = adapter.validate_llm_candidate_adoption(
                    proposed_component=str(matched.get("component", "") or ""),
                    proposed_reason=str(matched.get("reason", "") or ""),
                    deterministic_component=str(det_item.get("component", "") or ""),
                    deterministic_reason=str(det_item.get("reason", "") or ""),
                    context=dict(context or {}),
                    need_comp=bool(need_comp),
                    need_reason=bool(need_reason),
                )
                if not ok:
                    continue
            except Exception:
                pass
        used_idx.add(best_idx)
        matched["source"] = "counterfactual_agent_candidate_rerank"
        selected.append(matched)
        if len(selected) >= limit:
            break

    if not selected:
        return deterministic
    selected_keys = {
        (str(x.get("component") or ""), str(x.get("reason") or ""), str(x.get("datetime") or ""))
        for x in selected
    }
    for item in pool:
        if len(selected) >= limit:
            break
        key = (str(item.get("component") or ""), str(item.get("reason") or ""), str(item.get("datetime") or ""))
        if key in selected_keys:
            continue
        selected.append(dict(item))
        selected_keys.add(key)
    return selected[:limit] or deterministic

def _collect_agent_stats(trajectory: Sequence[Mapping[str, Any]]) -> tuple[bool, int, List[str], Dict[str, int]]:
    llm_called = False
    input_tokens = 0
    reasons: List[str] = []
    stats = {
        "llm_verify_count": 0,
        "llm_changed_count": 0,
        "llm_veto_count": 0,
        "llm_adopted_count": 0,
    }
    for step in trajectory or []:
        if not isinstance(step, Mapping):
            continue
        action = str(step.get("action") or "")
        if action in {"llm_verify_topk", "llm_verify_topk_failed", "llm_verify_candidate", "llm_verify_candidate_failed"}:
            llm_called = True
        if action == "llm_verify_candidate":
            stats["llm_verify_count"] += 1
            output = step.get("output")
            changed = False
            if isinstance(output, Mapping):
                try:
                    changed = bool(output.get("change_from_top1", False)) or int(output.get("selected_id", 1) or 1) != 1
                except Exception:
                    changed = bool(output.get("change_from_top1", False))
            if changed:
                stats["llm_changed_count"] += 1
        if action == "llm_veto":
            stats["llm_veto_count"] += 1
        if action == "uncertainty_gate":
            output = step.get("output")
            if isinstance(output, Mapping):
                reasons.extend([str(x) for x in list(output.get("reasons") or [])])
        output = step.get("output")
        if isinstance(output, Mapping):
            input_tokens += int(output.get("approx_input_tokens", 0) or 0)
        time_selection = step.get("time_selection")
        if isinstance(time_selection, Mapping):
            input_tokens += int(time_selection.get("approx_input_tokens", 0) or 0)
    stats["llm_adopted_count"] = max(0, int(stats["llm_changed_count"]) - int(stats["llm_veto_count"]))
    return bool(llm_called), int(input_tokens), _unique(reasons), stats


def _candidate_evidence_profile(item: Mapping[str, Any], context: Mapping[str, Any]) -> Dict[str, Any]:
    comp = str(item.get("component", "") or "").strip()
    reason = str(item.get("reason", "") or "").strip()
    dt = str(item.get("datetime", "") or "").strip()
    evidence = item.get("evidence", {}) if isinstance(item.get("evidence", {}), Mapping) else {}
    source_blob = " ".join(
        [
            str(item.get("source", "") or ""),
            " ".join(str(x) for x in list(item.get("supporting_metrics", []) or [])),
            " ".join(str(x) for x in list(item.get("evidence", []) or []) if not isinstance(item.get("evidence", []), Mapping)),
            " ".join(str(x) for x in list(evidence.get("merged_sources", []) or [])),
            " ".join(str(x) for x in list((context or {}).get("top_metrics", {}).get(comp, []) or [])),
        ]
    ).lower()
    has_metric = bool(
        list(item.get("supporting_metrics", []) or [])
        or list((context or {}).get("top_metrics", {}).get(comp, []) or [])
        or any(tok in source_blob for tok in ("metric", "cpu", "memory", "disk", "network", "latency", "io", "breach", "peak"))
    )
    has_trace = any(tok in source_blob for tok in ("trace", "span", "topology", "causal", "p95", "duration"))
    has_log = any(tok in source_blob for tok in ("log", "event", "error", "exception"))
    has_time = bool(dt)
    evidence_strength = int(has_metric) + int(has_trace) + int(has_log)
    if has_time:
        evidence_strength += 1
    return {
        "component": comp,
        "component_family": _candidate_component_family(comp),
        "reason": reason,
        "time_cluster": _candidate_time_cluster(dt),
        "has_metric": bool(has_metric),
        "has_trace": bool(has_trace),
        "has_log": bool(has_log),
        "has_time": bool(has_time),
        "evidence_strength": int(evidence_strength),
        "sufficient": bool(evidence_strength >= 2 or (has_metric and has_time)),
    }


def _evidence_conflict_for_llm(
    top1: Mapping[str, Any],
    challengers: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    need_time: bool,
    need_comp: bool,
    need_reason: bool,
) -> Dict[str, Any]:
    top1_profile = _candidate_evidence_profile(top1, context)
    checked: List[Dict[str, Any]] = []
    if not bool(top1_profile.get("sufficient")):
        return {
            "has_conflict": False,
            "reason": "top1_lacks_sufficient_real_evidence",
            "top1_profile": top1_profile,
            "challenger_profile": {},
            "checked_challengers": checked,
        }

    for raw in challengers:
        if not isinstance(raw, Mapping):
            continue
        prof = _candidate_evidence_profile(raw, context)
        checked.append(prof)
        if not bool(prof.get("sufficient")):
            continue
        conflicts: List[str] = []
        if bool(need_comp) and str(prof.get("component_family", "")) != str(top1_profile.get("component_family", "")):
            conflicts.append("component_family")
        if bool(need_reason) and str(prof.get("reason", "")) != str(top1_profile.get("reason", "")):
            conflicts.append("reason")
        if bool(need_time) and str(prof.get("time_cluster", "")) != str(top1_profile.get("time_cluster", "")):
            conflicts.append("time_cluster")
        if bool(top1_profile.get("has_trace")) and bool(prof.get("has_metric")) and not bool(top1_profile.get("has_metric")):
            conflicts.append("trace_vs_metric")
        if bool(top1_profile.get("has_metric")) and bool(prof.get("has_trace")) and not bool(prof.get("has_metric")):
            conflicts.append("metric_vs_trace")
        if conflicts:
            return {
                "has_conflict": True,
                "reason": "evidence_supported_challenger_conflicts_with_top1",
                "conflict_dimensions": list(dict.fromkeys(conflicts)),
                "top1_profile": top1_profile,
                "challenger_profile": prof,
                "challenger_key": {
                    "component": str(raw.get("component", "") or ""),
                    "reason": str(raw.get("reason", "") or ""),
                    "datetime": str(raw.get("datetime", "") or ""),
                },
                "checked_challengers": checked[:5],
            }

    return {
        "has_conflict": False,
        "reason": "no_evidence_supported_conflicting_challenger",
        "top1_profile": top1_profile,
        "challenger_profile": {},
        "checked_challengers": checked[:5],
    }


def _uncertainty_gate_for_tuples(items: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> Dict[str, Any]:
    need_time = bool((context or {}).get("need_time", False))
    need_comp = bool((context or {}).get("need_comp", False))
    need_reason = bool((context or {}).get("need_reason", False))

    def signature(item: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(item.get("datetime", "") or "").strip() if need_time else "",
            str(item.get("component", "") or "").strip() if need_comp else "",
            str(item.get("reason", "") or "").strip() if need_reason else "",
        )

    unique_items: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        sig = signature(item)
        if sig in seen:
            continue
        seen.add(sig)
        unique_items.append(dict(item))

    top1 = dict(unique_items[0]) if unique_items else {}
    top2 = dict(unique_items[1]) if len(unique_items) > 1 else {}
    try:
        top1_score = float(top1.get("score", 0.0) or 0.0)
    except Exception:
        top1_score = 0.0
    try:
        top2_score = float(top2.get("score", 0.0) or 0.0)
    except Exception:
        top2_score = 0.0
    gap = max(0.0, top1_score - top2_score)

    gap_th = float(context.get("evidence_score_gap_threshold", 0.15) or 0.15)
    floor = float(context.get("evidence_top1_score_floor", 0.30) or 0.30)
    floor_single = float(context.get("evidence_top1_score_floor_single", 0.50) or 0.50)
    skip_gap_th = float(context.get("gate_skip_gap_threshold", 0.30) or 0.30)

    reasons: List[str] = []
    need_llm = False
    if not unique_items:
        need_llm = True
        reasons.append("no_candidates")
    elif len(unique_items) == 1:
        need_llm = top1_score < floor_single
        if need_llm:
            reasons.append("low_top1_score_single")
    else:
        if top1_score < floor:
            need_llm = True
            reasons.append("low_top1_score")
        if gap < gap_th:
            need_llm = True
            reasons.append("small_score_gap")

    if need_llm and len(unique_items) >= 2 and top1_score >= floor_single and gap >= skip_gap_th:
        need_llm = False
        reasons.append("skip_llm_due_to_strong_det_top1")

    diversity_window = unique_items[: min(len(unique_items), 12)]
    diversity: Dict[str, int] = {}
    if need_comp:
        diversity["component_family"] = len(
            {
                fam
                for fam in (_candidate_component_family(str(item.get("component", "") or "")) for item in diversity_window)
                if fam
            }
        )
    if need_reason:
        diversity["reason"] = len({str(item.get("reason", "") or "").strip() for item in diversity_window if str(item.get("reason", "") or "").strip()})
    if need_time:
        diversity["time_cluster"] = len(
            {
                cluster
                for cluster in (_candidate_time_cluster(str(item.get("datetime", "") or "")) for item in diversity_window)
                if cluster
            }
        )
    if need_llm and diversity:
        active_dims = [count for count in diversity.values() if count > 0]
        if active_dims and max(active_dims) <= 1:
            need_llm = False
            reasons.append("skip_llm_due_to_low_candidate_diversity")

    evidence_conflict = _evidence_conflict_for_llm(
        top1,
        unique_items[1: min(len(unique_items), 8)],
        context,
        need_time=bool(need_time),
        need_comp=bool(need_comp),
        need_reason=bool(need_reason),
    )
    has_evidence_conflict = bool(evidence_conflict.get("has_conflict", False))
    if need_llm and not has_evidence_conflict:
        need_llm = False
        reasons.append(f"skip_llm_no_evidence_conflict:{evidence_conflict.get('reason', '')}")
    elif (not need_llm) and has_evidence_conflict and len(unique_items) >= 2 and gap < max(gap_th, skip_gap_th):
        need_llm = True
        reasons.append("call_llm_due_to_evidence_conflict")
    if has_evidence_conflict:
        context["llm_challenger_candidate_keys"] = [dict(evidence_conflict.get("challenger_key", {}) or {})]

    return {
        "need_llm": bool(need_llm),
        "reasons": list(dict.fromkeys(reasons)),
        "top1_score": float(top1_score),
        "top2_score": float(top2_score),
        "score_gap": float(gap),
        "gap_threshold": float(gap_th),
        "top1_floor": float(floor),
        "top1_floor_single": float(floor_single),
        "skip_gap_threshold": float(skip_gap_th),
        "unique_required_candidate_count": int(len(unique_items)),
        "candidate_diversity": dict(diversity),
        "evidence_conflict": dict(evidence_conflict),
    }


class IndexedEvidenceAgentRunner:
    """
    Shared post-index decision flow:
    evidence query result -> deterministic candidate tuples -> counterfactual
    controller gate -> LLM only when uncertain -> final selected tuples.
    """

    def deterministic_select(self, *, result: Any, k: int) -> List[Dict[str, Any]]:
        tuples = list(getattr(result, "tuples", []) or [])
        if not tuples:
            tuples = self._tuples_from_hypotheses(result)
        selected: List[Dict[str, Any]] = []
        used_family_reason: set[tuple[str, str]] = set()
        deferred: List[Dict[str, Any]] = []
        for item in tuples:
            comp = str((item or {}).get("component") or "")
            reason = str((item or {}).get("reason") or "")
            family = str((item or {}).get("family") or (service_base(comp) if comp else ""))
            key = (family or comp, reason)
            if key in used_family_reason:
                deferred.append(dict(item))
                continue
            selected.append(dict(item))
            if comp or reason:
                used_family_reason.add(key)
            if len(selected) >= max(1, int(k)):
                break
        if len(selected) < max(1, int(k)):
            seen_exact = {
                (
                    str(item.get("component") or ""),
                    str(item.get("reason") or ""),
                    str(item.get("datetime") or ""),
                )
                for item in selected
            }
            for item in deferred:
                key = (
                    str(item.get("component") or ""),
                    str(item.get("reason") or ""),
                    str(item.get("datetime") or ""),
                )
                if key in seen_exact:
                    continue
                selected.append(dict(item))
                seen_exact.add(key)
                if len(selected) >= max(1, int(k)):
                    break
        if not selected and tuples:
            selected = [dict(tuples[0])]
        if not selected:
            times = list(getattr(result, "times", []) or [])
            if times:
                selected = [{"datetime": str(times[0]), "score": 1.0, "source": "indexed_time"}]
        return selected

    def _tuples_from_hypotheses(self, result: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for hyp in list(getattr(result, "hypotheses", []) or []):
            comp = str(getattr(hyp, "component", "") or "").strip()
            reason = str(getattr(hyp, "reason", "") or "").strip()
            evidence = dict(getattr(hyp, "evidence", {}) or {})
            dt = ""
            onset_s = getattr(hyp, "onset_s", None)
            if onset_s is not None:
                try:
                    dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(onset_s)))
                except Exception:
                    dt = ""
            if not dt:
                expanded = evidence.get("expanded_candidate")
                if isinstance(expanded, Mapping):
                    dt = str(expanded.get("occurrence_time", "") or "").strip()
            if not dt:
                dt = str(evidence.get("first_anomaly_time", "") or "").strip()
            out.append(
                {
                    "component": comp,
                    "reason": reason,
                    "datetime": dt,
                    "score": float(getattr(hyp, "score", 0.0) or 0.0),
                    "source": "basil_hypothesis",
                    "supporting_metrics": list(evidence.get("supporting_metrics", []) or []),
                    "component_type": str(evidence.get("component_type", "") or ""),
                    "first_anomaly_time": str(evidence.get("first_anomaly_time", "") or ""),
                    "symptomness": float(evidence.get("symptomness", 0.0) or 0.0),
                }
            )
        if out:
            return sorted(out, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

        candidates = list(getattr(result, "candidates", []) or [])
        scores = dict(getattr(result, "combined_scores", {}) or {})
        first_times = dict(getattr(result, "first_times", {}) or {})
        top_metrics = dict(getattr(result, "top_metrics", {}) or {})
        for comp in candidates:
            c = str(comp or "").strip()
            if not c:
                continue
            out.append(
                {
                    "component": c,
                    "reason": "",
                    "datetime": str(first_times.get(c, "") or "").strip(),
                    "score": float(scores.get(c, 0.0) or 0.0),
                    "source": "basil_component",
                    "supporting_metrics": list(top_metrics.get(c, []) or []),
                    "component_type": _candidate_component_family(c),
                    "first_anomaly_time": str(first_times.get(c, "") or "").strip(),
                    "symptomness": 0.0,
                }
            )
        return sorted(out, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

    def _reason_scores(self, result: Any) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        tuples = list(getattr(result, "tuples", []) or [])
        for item in tuples:
            reason = str((item or {}).get("reason") or "").strip()
            if not reason:
                continue
            scores[reason] = max(float(scores.get(reason, 0.0)), float((item or {}).get("score", 0.0) or 0.0))
        for reason in list(getattr(result, "global_reasons", []) or []):
            r = str(reason or "").strip()
            if r:
                scores.setdefault(r, 1.0)
        return scores

    def _time_candidates(self, result: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in list(getattr(result, "tuples", []) or []):
            dt = str((item or {}).get("datetime") or "").strip()
            if not dt:
                continue
            key = (dt, str((item or {}).get("component") or ""), str((item or {}).get("reason") or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "datetime": dt,
                    "source": "indexed_tuple_time",
                    "score": float((item or {}).get("score", 0.0) or 0.0),
                    "component": str((item or {}).get("component") or ""),
                    "reason": str((item or {}).get("reason") or ""),
                    "evidence": {"from_index_tuple": True},
                }
            )
        scores = (((getattr(result, "evidence", {}) or {}).get("time_evidence") or {}).get("scores") or {})
        for dt, score in scores.items():
            key = (str(dt), "", "")
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "datetime": str(dt),
                    "source": "indexed_time_evidence",
                    "score": float(score or 0.0),
                    "component": "",
                    "reason": "",
                    "evidence": {"from_time_evidence": True},
                }
            )
        return sorted(out, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

    def _build_context(
        self,
        *,
        result: Any,
        task_id: int,
        args: Any,
        instruction: str,
    ) -> Dict[str, Any]:
        need_time, need_comp, need_reason = _need_flags(int(task_id))
        adapter = get_dataset_adapter(str(getattr(result, "dataset", "") or ""))
        root_count = int(_required_root_count(result=result, instruction=str(instruction or "")))
        comp_ev = getattr(result, "component_evidence", None)
        comp_scores = dict(getattr(comp_ev, "combined_scores", {}) or {})
        if not comp_scores:
            comp_scores = dict(getattr(result, "combined_scores", {}) or {})
        component_candidates = list(getattr(comp_ev, "candidates", []) or [])
        if not component_candidates:
            component_candidates = list(getattr(result, "candidates", []) or [])
        all_tuples = [dict(x) for x in (list(getattr(result, "tuples", []) or []) or self._tuples_from_hypotheses(result)) if isinstance(x, Mapping)]
        reasons = _unique(list(getattr(result, "global_reasons", []) or []) + list((getattr(result, "reasons_by_component", {}) or {}).values()))
        if not reasons and adapter is None:
            reasons = ["unknown"]
        if not component_candidates:
            component_candidates = ["unknown"]
        top_metrics = dict(getattr(comp_ev, "top_metrics", {}) or {})
        if not top_metrics:
            top_metrics = dict(getattr(result, "top_metrics", {}) or {})
        causal_scores = dict(getattr(comp_ev, "causal_scores", {}) or {})
        if not causal_scores:
            causal_scores = dict(getattr(result, "causal_scores", {}) or {})
        causal_edges = []
        for edge in list(getattr(comp_ev, "causal_edges", []) or getattr(result, "causal_edges", []) or []):
            if isinstance(edge, Mapping):
                causal_edges.append(dict(edge))
            else:
                try:
                    causal_edges.append(edge.to_dict())
                except Exception:
                    causal_edges.append(
                        {
                            "src": str(getattr(edge, "src", "") or ""),
                            "dst": str(getattr(edge, "dst", "") or ""),
                            "edge_type": str(getattr(edge, "edge_type", "") or ""),
                            "score": float(getattr(edge, "score", 0.0) or 0.0),
                            "support_count": int(getattr(edge, "support_count", 0) or 0),
                        }
                    )
        component_level = ""
        if adapter is not None:
            try:
                # scoring_points contains evaluation labels.  Do not use it to
                # infer dataset-specific candidate hints during inference.
                component_level = ""
            except Exception:
                component_level = ""

        adapter_context_overrides: Dict[str, Any] = {}
        adapter_reason_scores: Dict[str, float] = {}
        if adapter is not None and bool(need_reason):
            try:
                reasons, adapter_reason_scores, adapter_context_overrides = adapter.augment_index_reasons(
                    reasons=list(reasons or []),
                    result=result,
                    instruction=str(instruction or ""),
                    task_id=int(task_id),
                    args=args,
                    need_reason=True,
                )
            except Exception:
                adapter_reason_scores = {}
                adapter_context_overrides = {}

        tuple_limit = 50 if root_count > 1 else 20
        tuple_candidates = all_tuples[: int(tuple_limit)]

        component_first_times = dict(getattr(comp_ev, "first_times", {}) or {})
        component_peak_times = dict(getattr(comp_ev, "peak_times", {}) or {})
        use_structured_time = bool(adapter is not None and adapter.enable_structured_time_candidates())
        if use_structured_time:
            if not component_first_times:
                component_first_times = dict(getattr(result, "first_times", {}) or {})
            if not component_peak_times:
                component_peak_times = dict(getattr(result, "peak_times", {}) or {})

        time_candidates = self._time_candidates(result)
        if use_structured_time:
            query_start_local = str(getattr(result, "start_local", "") or "")
            query_end_local = str(getattr(result, "end_local", "") or "")
            for item in time_candidates:
                if not isinstance(item, dict):
                    continue
                ev = item.get("evidence")
                if not isinstance(ev, dict):
                    ev = {}
                ev.setdefault("query_start_local", query_start_local)
                ev.setdefault("query_end_local", query_end_local)
                item["evidence"] = ev
            seen_time_keys = {
                (
                    str(item.get("datetime", "") or ""),
                    str(item.get("component", "") or ""),
                    str(item.get("source", "") or ""),
                )
                for item in time_candidates
                if isinstance(item, Mapping)
            }
            max_comp_score = max([abs(float(v)) for v in comp_scores.values()] + [1.0])
            for comp, dt in list(component_first_times.items()):
                dts = str(dt or "").strip()
                c = str(comp or "").strip()
                if not dts or not c:
                    continue
                src = f"{adapter.__class__.__name__.replace('Adapter', '').lower()}_basil_first_time" if adapter is not None else "basil_first_time"
                key = (dts, c, src)
                if key in seen_time_keys:
                    continue
                seen_time_keys.add(key)
                time_candidates.append(
                    {
                        "datetime": dts,
                        "source": src,
                        "score": max(0.05, float(comp_scores.get(c, 0.0) or 0.0) / max(max_comp_score, 1e-9)),
                        "component": c,
                        "reason": "",
                        "evidence": {"from_basil_first_times": True, "query_start_local": query_start_local, "query_end_local": query_end_local},
                    }
                )
            for comp, dt in list(component_peak_times.items()):
                dts = str(dt or "").strip()
                c = str(comp or "").strip()
                if not dts or not c:
                    continue
                src = f"{adapter.__class__.__name__.replace('Adapter', '').lower()}_basil_peak_time" if adapter is not None else "basil_peak_time"
                key = (dts, c, src)
                if key in seen_time_keys:
                    continue
                seen_time_keys.add(key)
                time_candidates.append(
                    {
                        "datetime": dts,
                        "source": src,
                        "score": 0.72 * max(0.05, float(comp_scores.get(c, 0.0) or 0.0) / max(max_comp_score, 1e-9)),
                        "component": c,
                        "reason": "",
                        "evidence": {"from_basil_peak_times": True, "query_start_local": query_start_local, "query_end_local": query_end_local},
                    }
                )
            time_candidates.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

        if adapter is not None and bool(need_time):
            try:
                time_candidates = adapter.augment_index_time_candidates(
                    time_candidates=list(time_candidates),
                    result=result,
                    instruction=str(instruction or ""),
                    task_id=int(task_id),
                    args=args,
                    need_time=True,
                    component_scores=comp_scores,
                    component_first_times=component_first_times,
                    component_peak_times=component_peak_times,
                    context_overrides=adapter_context_overrides,
                )
            except Exception:
                pass

        # DEBUG: print time_candidates to stderr for diagnosis
        import sys as _sys, os as _os
        if _os.environ.get("RCA_DEBUG_TIME"):
            _sys.stderr.write(f"\n[TIME_DEBUG] instruction={str(instruction or '')[:80]}\n")
            for _tc in (time_candidates or [])[:15]:
                _sys.stderr.write(f"  {_tc.get('datetime','?'):22s}  score={float(_tc.get('score',0)):.3f}  src={_tc.get('source','?')}  comp={_tc.get('component','')}\n")
            _sys.stderr.flush()

        k_req = int(_required_root_count(result=result, instruction=str(instruction or "")))
        context = {
            "instruction": str(instruction or ""),
            "dataset": str(getattr(result, "dataset", "") or ""),
            "task_id": int(task_id),
            "need_time": bool(need_time),
            "need_comp": bool(need_comp),
            "need_reason": bool(need_reason),
            "difficulty": _difficulty(int(task_id)),
            "required_root_count": int(_required_root_count(result, instruction)),
            "allowed_components": list(component_candidates),
            "allowed_reasons": list(reasons),
            "candidate_tuples": tuple_candidates,
            "component_scores": comp_scores,
            "reason_scores": adapter_reason_scores if adapter_reason_scores else self._reason_scores(result),
            "time_candidates": time_candidates,
            "time_choice": (list(getattr(result, "times", []) or [""])[0] if list(getattr(result, "times", []) or []) else ""),
            "component_first_times": component_first_times,
            "component_peak_times": component_peak_times,
            "top_metrics": top_metrics,
            "causal_scores": causal_scores,
            "causal_edges": causal_edges,
            "basil_supporting_metrics": top_metrics,
            "metric_scores": comp_scores,
            "matched_metrics": top_metrics,
            "component_level": component_level,
            "date_dir": str(getattr(result, "date_dir", "") or ""),
            **dict(adapter_context_overrides or {}),
            "query_start_local": str(getattr(result, "start_local", "") or ""),
            "query_end_local": str(getattr(result, "end_local", "") or ""),
            "enable_uncertainty_llm": bool(getattr(args, "uncertainty_llm", False)),
            "enable_time_llm": bool(getattr(args, "time_llm", False)),
            "deterministic_only": bool(getattr(args, "deterministic_only", False)),
            "llm_uncertain_gap_threshold": float(getattr(args, "evidence_score_gap_threshold", 0.15) or 0.15),
            "llm_confidence_floor": float(getattr(args, "evidence_top1_score_floor", 0.30) or 0.30),
            "llm_time_confidence_floor": float(getattr(args, "evidence_top1_score_floor", 0.30) or 0.30),
            "llm_verify_top_k": (
                max(8, int(_required_root_count(result, instruction)) * 4)
                if root_count > 1
                else max(3, int(_required_root_count(result, instruction)) * 3, int(getattr(args, "basil_candidate_reason_top_r", 3) or 3))
            ),
            "llm_max_prompt_candidates": (
                min(8, max(len(reasons), int(getattr(args, "basil_candidate_reason_top_r", 3) or 3)))
                if bool(need_reason) and not bool(need_comp)
                else min(8, max(5, int(_required_root_count(result, instruction)) * 4))
            ),
            "evidence_score_gap_threshold": float(getattr(args, "evidence_score_gap_threshold", 0.15) or 0.15),
            "evidence_top1_score_floor": float(getattr(args, "evidence_top1_score_floor", 0.30) or 0.30),
            "evidence_top1_score_floor_single": float(getattr(args, "evidence_top1_score_floor_single", 0.50) or 0.50),
            "gate_skip_gap_threshold": float(getattr(args, "gate_skip_gap_threshold", 0.30) or 0.30),
            "evidence_time_gap_threshold": 0.20,
            "basil_max_trace_rows": int(getattr(args, "basil_max_trace_rows", 0) or 0),
        }
        context["llm_change_min_confidence"] = float(getattr(args, "llm_change_min_confidence", 0.85) or 0.85)
        context["llm_max_score_drop_vs_det_top1"] = float(getattr(args, "llm_max_score_drop_vs_det_top1", 0.05) or 0.05)
        if adapter is not None:
            try:
                plan = adapter.core_plan(
                    task_id=int(task_id),
                    need_time=bool(need_time),
                    need_comp=bool(need_comp),
                    need_reason=bool(need_reason),
                    context=context,
                )
                context["dataset_skill"] = dict(adapter.skill_summary() or {})
                context["basil_core_modules"] = dict(plan.to_dict() or {})
            except Exception:
                context.setdefault("dataset_skill", {})
                context.setdefault("basil_core_modules", {})

        # Inject lightweight per-reason evidence from the BASIL index result.
        # In indexed mode this must not call dataset solve_reason() fallbacks:
        # those may re-scan raw telemetry and dominate latency.  Every dataset
        # adapter can expose reason scores/metric hints through
        # augment_index_reasons(); otherwise we fall back to component
        # top_metrics already stored in the BASIL evidence result.
        if bool(need_reason):
            try:
                reason_metrics: List[str] = []
                for key, value in dict(adapter_context_overrides or {}).items():
                    key_l = str(key).lower()
                    if "reason" not in key_l and "metric" not in key_l:
                        continue
                    if isinstance(value, Mapping):
                        for vv in value.values():
                            if isinstance(vv, (list, tuple, set)):
                                reason_metrics.extend(str(x) for x in vv if str(x).strip())
                            elif isinstance(vv, str) and vv.strip():
                                reason_metrics.append(str(vv))
                    elif isinstance(value, (list, tuple, set)):
                        reason_metrics.extend(str(x) for x in value if str(x).strip())
                    elif isinstance(value, str) and value.strip():
                        reason_metrics.append(str(value))
                if not reason_metrics:
                    for values in dict(top_metrics or {}).values():
                        reason_metrics.extend(str(x) for x in list(values or []) if str(x).strip())
                reason_metrics = list(dict.fromkeys(reason_metrics))
                if adapter_reason_scores:
                    per_reason: Dict[str, Any] = {}
                    for reason, score in sorted(adapter_reason_scores.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True):
                        keywords: List[str] = []
                        if adapter is not None:
                            try:
                                keywords = [str(x).lower() for x in list(adapter.reason_metric_keywords(str(reason)) or []) if str(x).strip()]
                            except Exception:
                                keywords = []
                        matched = [
                            m
                            for m in reason_metrics
                            if not keywords or any(k in str(m).lower() for k in keywords)
                        ][:5]
                        per_reason[str(reason)] = {
                            "score": round(float(score or 0.0), 3),
                            "top_metrics": matched,
                            "breach_count": len(matched),
                            "distinct_metrics": len(set(matched)),
                            "source": "basil_index_top_metrics",
                        }
                    context["reason_evidence"] = {
                        "per_reason_evidence": per_reason,
                        "source": "basil_index_lightweight_no_raw_scan",
                    }
            except Exception:
                pass
        elif adapter is not None:
            try:
                if bool(adapter.constrain_llm_to_index_candidates()):
                    context["llm_change_min_confidence"] = float(getattr(args, "llm_change_min_confidence", 0.85) or 0.85)
                    context["llm_max_score_drop_vs_det_top1"] = float(getattr(args, "llm_max_score_drop_vs_det_top1", 0.05) or 0.05)
            except Exception:
                pass
        return context

    def _adapter_rerank_selected(
        self,
        *,
        result: Any,
        task_id: int,
        context: Dict[str, Any],
        selected: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        adapter = get_dataset_adapter(str(getattr(result, "dataset", "") or ""))
        if adapter is None:
            return list(selected or [])
        need_time, need_comp, need_reason = _need_flags(int(task_id))
        if bool(need_reason) and not bool(need_comp):
            rows = [dict(x) for x in list(selected or []) if isinstance(x, Mapping)]
            if rows and all(not str(x.get("component", "") or "").strip() for x in rows) and any(
                str(x.get("reason", "") or "").strip() for x in rows
            ):
                # Reason-only / reason+time tasks evaluate the reason surface,
                # not latent component identity.  If the adapter or generic
                # reason scorer has already folded evidence into component-free
                # reason candidates, do not reintroduce component tuples here.
                try:
                    quick = getattr(adapter, "postprocess_reason_only_index_rows", None)
                    if callable(quick):
                        return list(quick(rows=rows, context=dict(context or {})) or rows)
                except Exception:
                    pass
                return rows

        # ── Supplement pool: metric-scan + closed-set (universal recall fix) ──
        core_modules = dict((context or {}).get("basil_core_modules", {}) or {})
        if (bool(need_comp) or bool(need_reason)) and bool(core_modules.get("typed_channel_recall", True)):
            try:
                date_dir = str(getattr(result, "date_dir", "") or "")
                supplement = adapter.supplement_candidate_pool(
                    list(selected or []),
                    dict(context or {}),
                    date_dir=date_dir,
                )
                if supplement:
                    selected = list(selected or []) + supplement
            except Exception:
                pass

        candidates = []
        for item in list(selected or []):
            if not isinstance(item, Mapping):
                continue
            try:
                from rca.counterfactor_agent.schemas import CandidateHypothesis

                cand = CandidateHypothesis(
                    component=str(item.get("component", "") or ""),
                    reason=str(item.get("reason", "") or ""),
                    datetime=str(item.get("datetime", "") or ""),
                    evidence=[str(x) for x in list(item.get("supporting_metrics", []) or []) if str(x).strip()],
                    score=float(item.get("score", 0.0) or 0.0),
                    source=str(item.get("source", "") or "basil_deterministic"),
                )
                if isinstance(item.get("rootness_features", {}), Mapping):
                    setattr(cand, "rootness_features", dict(item.get("rootness_features", {}) or {}))
                setattr(cand, "symptomness", float(item.get("symptomness", 0.0) or 0.0))
                candidates.append(cand)
            except Exception:
                return list(selected or [])
        if not candidates:
            return list(selected or [])
        try:
            reranked = adapter.rerank_candidate_hypotheses(
                candidates=candidates,
                context=dict(context or {}),
                component_level=str((context or {}).get("component_level", "") or ""),
                need_comp=bool(need_comp),
                need_reason=bool(need_reason),
                need_time=bool(need_time),
            )
        except Exception:
            reranked = None
        if not reranked:
            return list(selected or [])
        time_candidates = [x for x in list((context or {}).get("time_candidates") or []) if isinstance(x, Mapping)]

        avoid_reused_time_clusters = bool(need_time) and int((context or {}).get("required_root_count", 1) or 1) > 1

        def time_cluster(dt: str) -> str:
            text = str(dt or "").strip()
            m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}):(\d{2}):", text)
            if not m:
                return text
            try:
                minute = int(m.group(2))
            except Exception:
                minute = 0
            return f"{m.group(1)}:{(minute // 5) * 5:02d}"

        def pick_dt(component: str, reason: str, used_clusters: Optional[set[str]] = None) -> str:
            if not time_candidates:
                return ""
            filtered_time_candidates = list(time_candidates)
            if avoid_reused_time_clusters and used_clusters:
                tmp = [
                    item
                    for item in filtered_time_candidates
                    if time_cluster(str(item.get("datetime", "") or "")) not in used_clusters
                ]
                if tmp:
                    filtered_time_candidates = tmp
            try:
                override = adapter.select_time_candidate_override(
                    time_candidates=filtered_time_candidates,
                    component=str(component or ""),
                    reason=str(reason or ""),
                    need_comp=bool(need_comp),
                    need_reason=bool(need_reason),
                    need_time=bool(need_time),
                    fallback=str((context or {}).get("time_choice", "") or ""),
                )
                if override is not None and str(override[0] or "").strip():
                    return str(override[0])
            except Exception:
                pass
            for item in filtered_time_candidates:
                dt = str(item.get("datetime", "") or "").strip()
                if not dt:
                    continue
                c0 = str(item.get("component", "") or "").strip()
                if component and c0 and str(component) not in {c0, service_base(c0), service_base(component)}:
                    continue
                rr = str(item.get("reason", "") or "").strip()
                if reason and rr and rr != reason:
                    continue
                return dt
            return str((context or {}).get("time_choice", "") or "")

        out: List[Dict[str, Any]] = []
        used_time_clusters: set[str] = set()
        for cand in list(reranked or [])[: max(1, int(k))]:
            comp = str(getattr(cand, "component", "") or "")
            reason = str(getattr(cand, "reason", "") or "")
            dt = pick_dt(comp, reason, used_time_clusters) if need_time else ""
            if avoid_reused_time_clusters and dt:
                used_time_clusters.add(time_cluster(dt))
            evidence_items = [str(x) for x in list(getattr(cand, "evidence", []) or []) if str(x).strip()]
            original = next(
                (
                    dict(item)
                    for item in list(selected or [])
                    if isinstance(item, Mapping)
                    and str(item.get("component", "") or "") == comp
                    and str(item.get("reason", "") or "") == reason
                ),
                {},
            )
            comp_type = str(original.get("component_type", "") or "")
            if not comp_type and adapter is not None:
                try:
                    comp_type = str(adapter.component_semantic_type(comp) or "")
                except Exception:
                    comp_type = ""
            rootness_features = dict(getattr(cand, "rootness_features", {}) or {})
            if not rootness_features and isinstance(original.get("rootness_features", {}), Mapping):
                rootness_features = dict(original.get("rootness_features", {}) or {})
            topology_context = {}
            if isinstance(original.get("topology_context", {}), Mapping):
                topology_context = dict(original.get("topology_context", {}) or {})
            elif adapter is not None and comp:
                try:
                    topology_context = dict(adapter.topology_context_for_component(component=comp, context=context) or {})
                except Exception:
                    topology_context = {}
            out.append(
                {
                    "component": comp,
                    "reason": reason,
                    "datetime": dt,
                    "score": float(getattr(cand, "score", 0.0) or 0.0),
                    "source": str(getattr(cand, "source", "") or "adapter_rerank"),
                    "supporting_metrics": evidence_items or [str(x) for x in list(original.get("supporting_metrics", []) or [])],
                    "component_type": comp_type,
                    "first_anomaly_time": str(original.get("first_anomaly_time", "") or ""),
                    "symptomness": float(getattr(cand, "symptomness", original.get("symptomness", 0.0)) or 0.0),
                    "rootness_features": rootness_features,
                    "topology_context": topology_context,
                }
            )
        if bool(need_comp) and not bool(need_reason) and not bool(need_time):
            out = _root_likelihood_order(out)
        try:
            out = adapter.postprocess_output_tuples(
                out,
                context=dict(context or {}),
                need_comp=bool(need_comp),
                need_reason=bool(need_reason),
                need_time=bool(need_time),
            )
        except Exception:
            pass
        return out or list(selected or [])

    def run(
        self,
        *,
        dataset: str,
        instruction: str,
        result: Any,
        task_id: int,
        args: Any,
        no_llm_fallback: bool,
        logger: Any = None,
    ) -> IndexedEvidenceDecision:
        k = max(1, int(_required_root_count(result, instruction)))
        need_time, need_comp, need_reason = _need_flags(int(task_id))
        context = self._build_context(
            result=result,
            task_id=task_id,
            args=args,
            instruction=instruction,
        )
        adapter = get_dataset_adapter(str(getattr(result, "dataset", "") or ""))
        core_output = BasilCorePipeline(
            adapter=adapter,
            need_time=bool(need_time),
            need_comp=bool(need_comp),
            need_reason=bool(need_reason),
            task_id=int(task_id),
            k=int(k),
            args=args,
            result=result,
            context=context,
            deterministic_select=self.deterministic_select,
            root_order=_root_likelihood_order,
            adapter_rerank=self._adapter_rerank_selected,
            attach_topology=_attach_topology_context_to_rows,
            llm_extra=_llm_only_challenger_expansion,
            diagnostics=_candidate_pool_diagnostics,
            coverage=_coverage_aware_challenger_keys,
            intervention=_intervention_challenger_plan,
        ).build()
        deterministic_pool = core_output.deterministic_pool
        deterministic = core_output.deterministic
        llm_prompt_pool = core_output.llm_prompt_pool
        llm_only_extras = core_output.llm_only_extras
        pool_diagnostics = core_output.pool_diagnostics
        coverage_plan = core_output.coverage_plan
        intervention_plan = core_output.intervention_plan
        context.update(dict(core_output.context or {}))
        max_prompt_keys = int((context or {}).get("llm_max_prompt_candidates", 6) or 6)

        def _key_sig(key: Mapping[str, Any]) -> tuple[str, str, str]:
            return (
                str(key.get("component", "") or ""),
                str(key.get("reason", "") or ""),
                str(key.get("datetime", "") or ""),
            )

        challenger_keys: List[Dict[str, Any]] = []
        seen_key_sigs: set[tuple[str, str, str]] = set()
        for source_keys in (
            list(coverage_plan.get("keys") or []),
            [_tuple_key(x) for x in list(llm_only_extras or []) if isinstance(x, Mapping)],
            list(intervention_plan.get("keys") or []),
        ):
            for key in source_keys:
                if not isinstance(key, Mapping):
                    continue
                sig = _key_sig(key)
                if sig in seen_key_sigs:
                    continue
                challenger_keys.append(dict(key))
                seen_key_sigs.add(sig)
                if len(challenger_keys) >= max(1, int(max_prompt_keys)):
                    break
            if len(challenger_keys) >= max(1, int(max_prompt_keys)):
                break
        if challenger_keys:
            context["llm_challenger_candidate_keys"] = challenger_keys
        consistency_gate = _evidence_consistency_gate(
            deterministic_pool,
            context,
            need_time=bool(need_time),
            need_comp=bool(need_comp),
            need_reason=bool(need_reason),
        )
        pool_diagnostics["evidence_consistency_gate"] = dict(consistency_gate)
        if bool(consistency_gate.get("accept")):
            return IndexedEvidenceDecision(
                selected_tuples=deterministic,
                uncertainty_reasons=["evidence_consistency_accept"],
                candidate_pool_diagnostics=pool_diagnostics,
                trajectory=[
                    {
                        "step": 1,
                        "action": "evidence_consistency_gate",
                        "elapsed_seconds": 0.0,
                        "output": dict(consistency_gate),
                    }
                ],
            )
        if bool(no_llm_fallback) or bool(getattr(args, "deterministic_only", False)):
            return IndexedEvidenceDecision(selected_tuples=deterministic, candidate_pool_diagnostics=pool_diagnostics)
        if not bool((context or {}).get("enable_uncertainty_llm", False)):
            return IndexedEvidenceDecision(selected_tuples=deterministic, candidate_pool_diagnostics=pool_diagnostics)

        gate = _uncertainty_gate_for_tuples(deterministic_pool, context)
        gate["candidate_pool_diagnostics"] = dict(pool_diagnostics)
        coverage_info = dict(coverage_plan.get("coverage") or {})
        coverage_missing_dims: List[str] = []
        if bool(need_comp) and list(coverage_info.get("missing_component_types") or []):
            coverage_missing_dims.append("component")
        if bool(need_reason) and list(coverage_info.get("missing_reason_families") or []):
            coverage_missing_dims.append("reason")
        if bool(need_time) and int(coverage_info.get("missing_time_cluster_count", 0) or 0) > 0:
            coverage_missing_dims.append("time_cluster")
        if bool(intervention_plan.get("call_llm")):
            profiles = list(intervention_plan.get("profiles") or [])
            top_prof = dict(profiles[0]) if profiles and isinstance(profiles[0], Mapping) else {}
            challenger_prof = dict(profiles[1]) if len(profiles) > 1 and isinstance(profiles[1], Mapping) else {}
            dims = []
            if bool(need_comp) and str(top_prof.get("component", "")) != str(challenger_prof.get("component", "")):
                dims.append("component")
            if bool(need_reason) and str(top_prof.get("reason", "")) != str(challenger_prof.get("reason", "")):
                dims.append("reason")
            if bool(need_time) and _candidate_time_cluster(str(top_prof.get("datetime", ""))) != _candidate_time_cluster(str(challenger_prof.get("datetime", ""))):
                dims.append("time_cluster")
            gate["evidence_conflict"] = {
                "has_conflict": True,
                "reason": "actionable_intervention_challenger",
                "conflict_dimensions": dims,
                "top1_profile": top_prof,
                "challenger_profile": {
                    **challenger_prof,
                    "sufficient": True,
                    "evidence_strength": 3,
                    "component_family": _candidate_component_family(str(challenger_prof.get("component", "") or "")),
                    "time_cluster": _candidate_time_cluster(str(challenger_prof.get("datetime", "") or "")),
                },
                "challenger_key": dict((list(intervention_plan.get("keys") or []) + [{}])[1] or {}),
                "checked_challengers": profiles[1:6],
            }
        elif coverage_missing_dims and len(challenger_keys) >= 2:
            gate["evidence_conflict"] = {
                "has_conflict": True,
                "reason": "coverage_aware_challenger",
                "conflict_dimensions": list(coverage_missing_dims),
                "coverage": dict(coverage_info),
                "challenger_key": dict(challenger_keys[1] if len(challenger_keys) > 1 else {}),
            }
        if adapter is not None:
            try:
                gate = adapter.refine_index_llm_gate(
                    gate=dict(gate),
                    candidates=[dict(x) for x in list(deterministic_pool or []) if isinstance(x, Mapping)],
                    context=context,
                    task_id=int(task_id),
                )
            except Exception:
                pass
        if (
            not bool(gate.get("need_llm"))
            and (bool(intervention_plan.get("call_llm")) or bool(coverage_missing_dims))
            and len(list(context.get("llm_challenger_candidate_keys") or [])) >= 2
        ):
            gate["need_llm"] = True
            gate.setdefault("reasons", [])
            gate["reasons"] = list(
                dict.fromkeys(
                    list(gate.get("reasons") or [])
                    + [
                        "call_llm_due_to_intervention_challenger"
                        if bool(intervention_plan.get("call_llm"))
                        else "call_llm_due_to_coverage_challenger"
                    ]
                )
            )
        if not bool(gate.get("need_llm")):
            return IndexedEvidenceDecision(
                selected_tuples=deterministic,
                uncertainty_reasons=list(gate.get("reasons") or []),
                candidate_pool_diagnostics=pool_diagnostics,
                trajectory=[
                    {
                        "step": 1,
                        "action": "uncertainty_gate",
                        "elapsed_seconds": 0.0,
                        "output": dict(gate),
                    }
                ],
            )

        agent_prompt, basic_prompt = _load_counterfactual_prompts(dataset)
        agent = CounterfactualRCAAgent(agent_prompt=agent_prompt, basic_prompt=basic_prompt)
        run_logger = logger or logging.getLogger(__name__)
        t0 = time.perf_counter()
        try:
            prediction, trajectory, prompt = agent.run(
                instruction=instruction,
                logger=run_logger,
                dataset_name=dataset,
                max_step=int(getattr(args, "controller_max_step", 25) or 25),
                max_turn=int(getattr(args, "controller_max_turn", 5) or 5),
                task_focus=_task_focus(int(task_id)),
                context=context,
            )
        except Exception as exc:
            return IndexedEvidenceDecision(
                selected_tuples=deterministic,
                agent_called=True,
                candidate_pool_diagnostics=pool_diagnostics,
                elapsed_seconds=float(time.perf_counter() - t0),
                prediction=str(exc),
            )
        elapsed = float(time.perf_counter() - t0)
        llm_called, input_tokens, reasons, llm_stats = _collect_agent_stats(trajectory)
        pool_diagnostics.update(dict(llm_stats or {}))
        adapter = get_dataset_adapter(str(dataset or ""))
        selected = _safe_prediction_to_tuples_constrained(
            prediction,
            deterministic_pool,
            k,
            task_id=int(task_id),
            dataset=str(dataset or ""),
            adapter=adapter,
            context=context,
        )
        prompt_items = list(prompt or []) if isinstance(prompt, list) else ([prompt] if str(prompt or "").strip() else [])
        return IndexedEvidenceDecision(
            selected_tuples=selected,
            agent_called=True,
            llm_called=bool(llm_called),
            llm_used=bool(llm_called),
            uncertainty_reasons=reasons,
            candidate_pool_diagnostics=pool_diagnostics,
            input_tokens=int(input_tokens),
            elapsed_seconds=elapsed,
            prediction=str(prediction),
            trajectory=list(trajectory or []),
            prompt=prompt_items,
        )
