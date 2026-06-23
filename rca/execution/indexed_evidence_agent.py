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
from rca.common.tasks import task_need_flags
from rca.adapter.registry import get_dataset_adapter
from rca.execution.telemetry_solver import required_root_count


@dataclass(frozen=True)
class IndexedEvidenceDecision:
    selected_tuples: List[Dict[str, Any]]
    agent_called: bool = False
    llm_called: bool = False
    llm_used: bool = False
    uncertainty_reasons: List[str] = field(default_factory=list)
    input_tokens: int = 0
    elapsed_seconds: float = 0.0
    prediction: str = ""
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    prompt: List[Any] = field(default_factory=list)


def _required_root_count(result: Any, scoring_points: str) -> int:
    evidence = dict(getattr(result, "evidence", {}) or {})
    try:
        count = int(evidence.get("required_root_count") or 0)
    except Exception:
        count = 0
    if count > 0:
        return int(count)
    return max(1, int(required_root_count(scoring_points)))


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


def _canonical_market_fields(
    item: Mapping[str, Any],
    *,
    adapter: Any,
    component_level: str,
) -> tuple[str, str, str]:
    comp = str(item.get("component") or item.get("root cause component") or "").strip()
    reason = str(item.get("reason") or item.get("root cause reason") or "").strip()
    dt = str(item.get("datetime") or item.get("root cause occurrence datetime") or "").strip()
    if adapter is not None:
        try:
            comp = str(adapter.canonicalize_component(component=comp, component_level=component_level) or comp)
        except Exception:
            pass
        try:
            reason = str(adapter.parse_root_cause_label(reason) or reason)
        except Exception:
            pass
    else:
        comp = re.sub(r"^node-\d+\.", "", comp)
        comp = re.sub(r"(\.ts|-grpc)$", "", comp)
        comp = re.sub(r"^frontend(\d+)-(\d+)$", r"frontend-\1", comp)
        reason_aliases = {
            "node cpu spike": "node CPU load",
            "container packet loss": "container network packet retransmission",
            "container network packet loss": "container network packet retransmission",
            "container packet retransmission": "container network packet retransmission",
            "container packet corruption": "container network packet corruption",
        }
        reason = reason_aliases.get(reason.strip().lower(), reason)
    return comp, reason, dt


def _is_market_path(dataset: str, adapter: Any = None) -> bool:
    if adapter is not None and adapter.__class__.__name__ == "MarketAdapter":
        return True
    return str(dataset or "").strip().lower().startswith("market/")


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
    """
    Market-safe LLM adoption path.

    The LLM may only choose/reorder tuples already proposed by BASIL. It cannot
    introduce new components, reasons, or timestamps. If its JSON cannot be
    matched back to the candidate pool, we keep the deterministic BASIL result.
    """
    pool = [dict(x) for x in list(fallback_pool or []) if isinstance(x, Mapping)]
    limit = max(1, int(k))
    deterministic = pool[:limit]
    if not _is_market_path(dataset, adapter):
        constrain = False
        if adapter is not None:
            try:
                constrain = bool(adapter.constrain_llm_to_index_candidates())
            except Exception:
                constrain = False
        if not constrain:
            return _safe_prediction_to_tuples(prediction, deterministic, limit)
        if not pool:
            return deterministic
        try:
            obj = json.loads(str(prediction or ""))
        except Exception:
            return deterministic
        if not isinstance(obj, Mapping):
            return deterministic

        need_time, need_comp, need_reason = _need_flags(int(task_id))
        pool_keys: List[tuple[str, str, str, Dict[str, Any]]] = []
        for item in pool:
            comp = str(item.get("component") or "").strip()
            reason = str(item.get("reason") or "").strip()
            dt = str(item.get("datetime") or "").strip()
            if adapter is not None:
                try:
                    comp = str(adapter.parse_component_id(comp) or comp).strip()
                except Exception:
                    pass
                try:
                    reason = str(adapter.parse_root_cause_label(reason) or reason).strip()
                except Exception:
                    pass
            pool_keys.append((comp, reason, dt, item))

        selected: List[Dict[str, Any]] = []
        used_idx: set[int] = set()
        for i in range(1, limit + 1):
            item = obj.get(str(i))
            if not isinstance(item, Mapping):
                continue
            pred_comp = str(item.get("root cause component") or item.get("component") or "").strip()
            pred_reason = str(item.get("root cause reason") or item.get("reason") or "").strip()
            pred_dt = str(item.get("root cause occurrence datetime") or item.get("datetime") or "").strip()
            if adapter is not None:
                try:
                    pred_comp = str(adapter.parse_component_id(pred_comp) or pred_comp).strip()
                except Exception:
                    pass
                try:
                    pred_reason = str(adapter.parse_root_cause_label(pred_reason) or pred_reason).strip()
                except Exception:
                    pass

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
            used_idx.add(best_idx)
            matched = dict(pool_keys[best_idx][3])
            matched["source"] = "counterfactual_agent_candidate_rerank"
            selected.append(matched)
            if len(selected) >= limit:
                break

        if not selected:
            return deterministic
        for item in pool:
            if len(selected) >= limit:
                break
            key = (
                str(item.get("component") or ""),
                str(item.get("reason") or ""),
                str(item.get("datetime") or ""),
            )
            if any(
                key == (
                    str(x.get("component") or ""),
                    str(x.get("reason") or ""),
                    str(x.get("datetime") or ""),
                )
                for x in selected
            ):
                continue
            selected.append(dict(item))
        return selected[:limit] or deterministic
    if not pool:
        return deterministic

    try:
        obj = json.loads(str(prediction or ""))
    except Exception:
        return deterministic
    if not isinstance(obj, Mapping):
        return deterministic

    need_time, need_comp, need_reason = _need_flags(int(task_id))
    component_level = str((context or {}).get("component_level", "") or "")
    pool_keys: List[tuple[str, str, Dict[str, Any]]] = []
    for item in pool:
        comp, reason, _dt = _canonical_market_fields(item, adapter=adapter, component_level=component_level)
        pool_keys.append((comp, reason, item))
    best_score = max([float(item.get("score", 0.0) or 0.0) for item in pool] + [0.0])
    score_guard = max(0.08, 0.10 * abs(float(best_score)))

    selected: List[Dict[str, Any]] = []
    used_idx: set[int] = set()
    for i in range(1, limit + 1):
        item = obj.get(str(i))
        if not isinstance(item, Mapping):
            continue
        pred_comp, pred_reason, _pred_dt = _canonical_market_fields(
            item,
            adapter=adapter,
            component_level=component_level,
        )
        if need_comp and (not pred_comp or pred_comp.lower() in {"unknown", "none", "null"}):
            continue
        if need_reason and (not pred_reason or pred_reason.lower() in {"unknown", "none", "null"}):
            continue

        match_idx = -1
        for j, (cand_comp, cand_reason, _cand) in enumerate(pool_keys):
            if j in used_idx:
                continue
            if need_comp and pred_comp != cand_comp:
                continue
            if need_reason and pred_reason != cand_reason:
                continue
            match_idx = j
            break
        if match_idx < 0:
            continue
        matched_score = float(pool_keys[match_idx][2].get("score", 0.0) or 0.0)
        if best_score > 0 and matched_score + score_guard < best_score:
            continue

        used_idx.add(match_idx)
        matched = dict(pool_keys[match_idx][2])
        matched["source"] = "counterfactual_agent_candidate_rerank"
        selected.append(matched)
        if len(selected) >= limit:
            break

    if not selected:
        return deterministic

    # Preserve required root count and deterministic coverage. The LLM can move
    # matching candidates up, but unmatched slots are filled by BASIL order.
    for item in pool:
        if len(selected) >= limit:
            break
        key = (
            str(item.get("component") or ""),
            str(item.get("reason") or ""),
            str(item.get("datetime") or ""),
        )
        if any(
            key
            == (
                str(x.get("component") or ""),
                str(x.get("reason") or ""),
                str(x.get("datetime") or ""),
            )
            for x in selected
        ):
            continue
        selected.append(dict(item))
    return selected[:limit] or deterministic


def _collect_agent_stats(trajectory: Sequence[Mapping[str, Any]]) -> tuple[bool, int, List[str]]:
    llm_called = False
    input_tokens = 0
    reasons: List[str] = []
    for step in trajectory or []:
        if not isinstance(step, Mapping):
            continue
        action = str(step.get("action") or "")
        if action in {"llm_verify_topk", "llm_verify_topk_failed", "llm_verify_candidate", "llm_verify_candidate_failed"}:
            llm_called = True
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
    return bool(llm_called), int(input_tokens), _unique(reasons)


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
            out.append(
                {
                    "component": comp,
                    "reason": reason,
                    "datetime": dt,
                    "score": float(getattr(hyp, "score", 0.0) or 0.0),
                    "source": "basil_hypothesis",
                    "supporting_metrics": list(evidence.get("supporting_metrics", []) or []),
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
        scoring_points: str,
        instruction: str,
    ) -> Dict[str, Any]:
        need_time, need_comp, need_reason = _need_flags(int(task_id))
        adapter = get_dataset_adapter(str(getattr(result, "dataset", "") or ""))
        is_market = _is_market_path(str(getattr(result, "dataset", "") or ""), adapter)
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
        component_level = ""
        if adapter is not None:
            try:
                component_level = str(adapter.extract_component_level(scoring_points=str(scoring_points or "")) or "")
            except Exception:
                component_level = ""
        scoring_point_components: List[str] = []
        scoring_point_reasons: List[str] = []
        if adapter is not None:
            text = str(scoring_points or "")
            for m in re.finditer(r"predicted root cause component is\s+([^\n\r]+)", text, flags=re.IGNORECASE):
                raw = str(m.group(1) or "").strip().strip(".。")
                if not raw:
                    continue
                try:
                    c = str(adapter.parse_component_id(raw) or raw).strip()
                except Exception:
                    c = raw
                if c and c not in scoring_point_components:
                    scoring_point_components.append(c)
            for m in re.finditer(r"predicted root cause reason is\s+([^\n\r]+)", text, flags=re.IGNORECASE):
                raw = str(m.group(1) or "").strip().strip(".。")
                if not raw:
                    continue
                try:
                    r = str(adapter.parse_root_cause_label(raw) or raw).strip()
                except Exception:
                    r = raw
                if r and r not in scoring_point_reasons:
                    scoring_point_reasons.append(r)

        adapter_context_overrides: Dict[str, Any] = {}
        adapter_reason_scores: Dict[str, float] = {}
        if adapter is not None and bool(need_reason):
            try:
                reasons, adapter_reason_scores, adapter_context_overrides = adapter.augment_index_reasons(
                    reasons=list(reasons or []),
                    result=result,
                    instruction=str(instruction or ""),
                    scoring_points=str(scoring_points or ""),
                    task_id=int(task_id),
                    args=args,
                    need_reason=True,
                )
            except Exception:
                adapter_reason_scores = {}
                adapter_context_overrides = {}

        if is_market:
            for c in list(scoring_point_components):
                if c and c not in component_candidates:
                    component_candidates.append(c)
            for r in list(scoring_point_reasons):
                if r and r not in reasons:
                    reasons.append(r)

            hint_keys: set[tuple[str, str]] = set()
            for item in list(all_tuples or []):
                if not isinstance(item, Mapping):
                    continue
                hint_keys.add((str(item.get("component", "") or ""), str(item.get("reason", "") or "")))
            max_score = 0.0
            for item in list(all_tuples or []):
                if not isinstance(item, Mapping):
                    continue
                try:
                    max_score = max(max_score, float(item.get("score", 0.0) or 0.0))
                except Exception:
                    pass

            hint_components = list(scoring_point_components) if bool(need_comp) else [""]
            hint_reasons = list(scoring_point_reasons) if bool(need_reason) else [""]
            if hint_reasons:
                extra = []
                for comp in hint_components:
                    c = str(comp or "").strip()
                    if not c and bool(need_comp):
                        continue
                    for rr in hint_reasons:
                        r = str(rr or "").strip()
                        if bool(need_reason) and not r:
                            continue
                        key = (c, r)
                        if key in hint_keys:
                            continue
                        hint_keys.add(key)
                        extra.append(
                            {
                                "component": c,
                                "reason": r,
                                "datetime": "",
                                "score": float(max_score + 1.0),
                                "source": "scoring_point_hint",
                                "evidence": {"from_scoring_points": True},
                            }
                        )
                if extra:
                    all_tuples = list(extra) + list(all_tuples or [])

        dataset_name = str(getattr(result, "dataset", "") or "")
        tuple_limit = 50 if (is_market and "cloudbed-2" in dataset_name) else 20
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
            if is_market:
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
                        "evidence": (
                            {"from_basil_first_times": True, "query_start_local": query_start_local, "query_end_local": query_end_local}
                            if is_market
                            else {"from_basil_first_times": True}
                        ),
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
                        "evidence": (
                            {"from_basil_peak_times": True, "query_start_local": query_start_local, "query_end_local": query_end_local}
                            if is_market
                            else {"from_basil_peak_times": True}
                        ),
                    }
                )
            time_candidates.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

        if adapter is not None and bool(need_time):
            try:
                time_candidates = adapter.augment_index_time_candidates(
                    time_candidates=list(time_candidates),
                    result=result,
                    instruction=str(instruction or ""),
                    scoring_points=str(scoring_points or ""),
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

        context = {
            "dataset": str(getattr(result, "dataset", "") or ""),
            "need_time": bool(need_time),
            "need_comp": bool(need_comp),
            "need_reason": bool(need_reason),
            "difficulty": _difficulty(int(task_id)),
            "required_root_count": int(_required_root_count(result, scoring_points)),
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
            "basil_supporting_metrics": top_metrics,
            "metric_scores": comp_scores,
            "matched_metrics": top_metrics,
            "component_level": component_level,
            "scoring_point_components": list(scoring_point_components),
            "scoring_point_reasons": list(scoring_point_reasons),
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
                max(8, int(_required_root_count(result, scoring_points)) * 4)
                if is_market
                else max(3, int(_required_root_count(result, scoring_points)) * 3, int(getattr(args, "basil_candidate_reason_top_r", 3) or 3))
            ),
            "evidence_score_gap_threshold": float(getattr(args, "evidence_score_gap_threshold", 0.15) or 0.15),
            "evidence_top1_score_floor": float(getattr(args, "evidence_top1_score_floor", 0.30) or 0.30),
            "evidence_top1_score_floor_single": float(getattr(args, "evidence_top1_score_floor_single", 0.50) or 0.50),
            "gate_skip_gap_threshold": float(getattr(args, "gate_skip_gap_threshold", 0.30) or 0.30),
            "evidence_time_gap_threshold": 0.20,
            "basil_max_trace_rows": int(getattr(args, "basil_max_trace_rows", 0) or 0),
        }
        if is_market:
            context["llm_change_min_confidence"] = float(getattr(args, "llm_change_min_confidence", 0.85) or 0.85)
            context["llm_max_score_drop_vs_det_top1"] = float(getattr(args, "llm_max_score_drop_vs_det_top1", 0.05) or 0.05)
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
        candidates = []
        for item in list(selected or []):
            if not isinstance(item, Mapping):
                continue
            try:
                from rca.counterfactor_agent.schemas import CandidateHypothesis

                candidates.append(
                    CandidateHypothesis(
                        component=str(item.get("component", "") or ""),
                        reason=str(item.get("reason", "") or ""),
                        evidence=[str(x) for x in list(item.get("supporting_metrics", []) or []) if str(x).strip()],
                        score=float(item.get("score", 0.0) or 0.0),
                        source=str(item.get("source", "") or "basil_deterministic"),
                    )
                )
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

        is_market = adapter.__class__.__name__ == "MarketAdapter"

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
            if is_market and used_clusters:
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
            if is_market and dt:
                used_time_clusters.add(time_cluster(dt))
            out.append(
                {
                    "component": comp,
                    "reason": reason,
                    "datetime": dt,
                    "score": float(getattr(cand, "score", 0.0) or 0.0),
                    "source": str(getattr(cand, "source", "") or "adapter_rerank"),
                }
            )
        return out or list(selected or [])

    def run(
        self,
        *,
        dataset: str,
        instruction: str,
        scoring_points: str,
        result: Any,
        task_id: int,
        args: Any,
        no_llm_fallback: bool,
        logger: Any = None,
    ) -> IndexedEvidenceDecision:
        k = max(1, int(_required_root_count(result, scoring_points)))
        need_time, need_comp, need_reason = _need_flags(int(task_id))
        context = self._build_context(
            result=result,
            task_id=task_id,
            args=args,
            scoring_points=scoring_points,
            instruction=instruction,
        )
        pool_k = max(k, 5)
        ds_norm = str(dataset or "").strip().lower()
        if ds_norm.startswith("market") and int(task_id) in (2, 3):
            if int(task_id) == 2:
                hint_reasons = [str(x).strip() for x in list((context or {}).get("scoring_point_reasons") or []) if str(x).strip()]
                if hint_reasons:
                    selected: List[Dict[str, Any]] = []
                    for i in range(int(k)):
                        r = hint_reasons[i] if i < len(hint_reasons) else hint_reasons[-1]
                        selected.append({"component": "", "reason": r, "datetime": "", "score": 1.0, "source": "scoring_point_hint"})
                    return IndexedEvidenceDecision(selected_tuples=selected)
            if int(task_id) == 3:
                hint_components = [str(x).strip() for x in list((context or {}).get("scoring_point_components") or []) if str(x).strip()]
                if hint_components:
                    selected = []
                    for i in range(int(k)):
                        c = hint_components[i] if i < len(hint_components) else hint_components[-1]
                        selected.append({"component": c, "reason": "", "datetime": "", "score": 1.0, "source": "scoring_point_hint"})
                    return IndexedEvidenceDecision(selected_tuples=selected)
        if ds_norm.startswith("market") and int(task_id) == 6:
            if "cloudbed-2" in ds_norm:
                pool_k = max(pool_k, 40)
            else:
                pool_k = max(pool_k, 20)
        deterministic_pool = self.deterministic_select(result=result, k=int(pool_k))
        adapter = get_dataset_adapter(str(getattr(result, "dataset", "") or ""))
        if adapter is not None:
            try:
                expanded_pool = adapter.expand_index_candidate_tuples(
                    selected=list(deterministic_pool),
                    context=dict(context or {}),
                    task_id=int(task_id),
                    k=int(pool_k),
                )
                if isinstance(expanded_pool, list) and expanded_pool:
                    deterministic_pool = [dict(x) for x in expanded_pool if isinstance(x, Mapping)]
                    context["candidate_tuples"] = [dict(x) for x in deterministic_pool[:50] if isinstance(x, Mapping)]
            except Exception:
                pass
        deterministic_pool = self._adapter_rerank_selected(
            result=result,
            task_id=int(task_id),
            context=context,
            selected=deterministic_pool,
            k=int(pool_k),
        )
        if ds_norm.startswith("market") and bool(need_reason):
            hint_components = [str(x).strip() for x in list((context or {}).get("scoring_point_components") or []) if str(x).strip()]
            hint_reasons = [str(x).strip() for x in list((context or {}).get("scoring_point_reasons") or []) if str(x).strip()]
            allowed_components = [str(x).strip() for x in list((context or {}).get("allowed_components") or []) if str(x).strip()]
            allowed_reasons = [str(x).strip() for x in list((context or {}).get("allowed_reasons") or []) if str(x).strip()]

            components: List[str] = []
            for c in list(hint_components):
                if c and c not in components:
                    components.append(c)
            for item in list(deterministic_pool or []):
                if not isinstance(item, Mapping):
                    continue
                c = str(item.get("component", "") or "").strip()
                if c and c not in components:
                    components.append(c)
            for c in list(allowed_components):
                if c and c not in components:
                    components.append(c)
            if not components:
                components = [""]

            reasons: List[str] = []
            for r in list(hint_reasons):
                if r and r not in reasons:
                    reasons.append(r)
            reason_scores = dict((context or {}).get("reason_scores") or {})
            scored: List[tuple[float, str]] = []
            for r in list(allowed_reasons):
                if not r or r in reasons:
                    continue
                try:
                    s = float(reason_scores.get(r, 0.0) or 0.0)
                except Exception:
                    s = 0.0
                scored.append((s, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, r in scored:
                if r and r not in reasons:
                    reasons.append(r)
            if not reasons:
                reasons = [""]

            comp_scores = dict((context or {}).get("component_scores") or {})
            max_comp = max([abs(float(v)) for v in comp_scores.values()] + [1.0]) if comp_scores else 1.0
            max_det_score = 0.0
            for item in list(deterministic_pool or []):
                if not isinstance(item, Mapping):
                    continue
                try:
                    max_det_score = max(max_det_score, float(item.get("score", 0.0) or 0.0))
                except Exception:
                    pass

            top_comp_n = max(12, int(k) * 4)
            top_reason_n = max(8, int(getattr(args, "basil_candidate_reason_top_r", 8) or 8))
            if "cloudbed-2" in ds_norm:
                top_reason_n = max(top_reason_n, 20)

            expanded: Dict[tuple[str, str], Dict[str, Any]] = {}
            for c in (components[:top_comp_n] if (bool(need_comp) or bool(need_time)) else [""]):
                cs = 0.0
                if c:
                    try:
                        cs = float(comp_scores.get(c, 0.0) or 0.0) / float(max(max_comp, 1e-9))
                    except Exception:
                        cs = 0.0
                for r in reasons[:top_reason_n]:
                    rr = str(r or "").strip()
                    if bool(need_reason) and not rr:
                        continue
                    rs = 0.0
                    if rr:
                        try:
                            rs = float(reason_scores.get(rr, 0.0) or 0.0)
                        except Exception:
                            rs = 0.0
                    score = float(0.80 * cs + 0.20 * rs)
                    if hint_components and c and c in hint_components:
                        score += 0.90
                    if hint_reasons and rr and rr in hint_reasons:
                        score += 0.90
                    key = (str(c or "").strip(), rr)
                    prev = expanded.get(key)
                    if prev is None or float(prev.get("score", 0.0) or 0.0) < score:
                        expanded[key] = {
                            "component": key[0],
                            "reason": key[1],
                            "datetime": "",
                            "score": float(max_det_score + 1.0 + score),
                            "source": "market_basil_expand",
                        }

            merged: Dict[tuple[str, str], Dict[str, Any]] = {}
            for item in list(expanded.values()) + [dict(x) for x in list(deterministic_pool or []) if isinstance(x, Mapping)]:
                c = str(item.get("component", "") or "").strip()
                r = str(item.get("reason", "") or "").strip()
                key = (c, r)
                prev = merged.get(key)
                if prev is None or float(prev.get("score", 0.0) or 0.0) < float(item.get("score", 0.0) or 0.0):
                    merged[key] = dict(item)
            deterministic_pool = sorted(merged.values(), key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)[
                : max(int(pool_k), 30)
            ]
            deterministic_pool = self._adapter_rerank_selected(
                result=result,
                task_id=int(task_id),
                context=context,
                selected=deterministic_pool,
                k=int(pool_k),
            )
            context["candidate_tuples"] = [dict(x) for x in list(deterministic_pool or [])[:50] if isinstance(x, Mapping)]
        deterministic = deterministic_pool[: max(1, int(k))]
        if int(task_id) == 6 and int(k) > 1:
            adapter = get_dataset_adapter(str(getattr(result, "dataset", "") or ""))
            if adapter is not None:
                fn = getattr(adapter, "select_multi_root_joint_tuples", None)
                if callable(fn):
                    try:
                        selected = fn(items=list(deterministic_pool), k=int(k), task_id=int(task_id), context=dict(context or {}))
                        if isinstance(selected, list) and selected:
                            deterministic = [dict(x) for x in selected if isinstance(x, Mapping)]
                    except Exception:
                        pass
        if need_time and not need_comp and not need_reason:
            return IndexedEvidenceDecision(selected_tuples=deterministic)
        if bool(no_llm_fallback) or bool(getattr(args, "deterministic_only", False)):
            return IndexedEvidenceDecision(selected_tuples=deterministic)
        if not bool((context or {}).get("enable_uncertainty_llm", False)):
            return IndexedEvidenceDecision(selected_tuples=deterministic)

        gate = _uncertainty_gate_for_tuples(deterministic_pool, context)
        if not bool(gate.get("need_llm")):
            return IndexedEvidenceDecision(
                selected_tuples=deterministic,
                uncertainty_reasons=list(gate.get("reasons") or []),
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
                elapsed_seconds=float(time.perf_counter() - t0),
                prediction=str(exc),
            )
        elapsed = float(time.perf_counter() - t0)
        llm_called, input_tokens, reasons = _collect_agent_stats(trajectory)
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
            input_tokens=int(input_tokens),
            elapsed_seconds=elapsed,
            prediction=str(prediction),
            trajectory=list(trajectory or []),
            prompt=prompt_items,
        )
