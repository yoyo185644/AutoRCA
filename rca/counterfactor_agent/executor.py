import json
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from rca.api_router import get_chat_completion
from rca.adapter.registry import get_dataset_adapter
from rca.adapter.utils.time_candidates import TimeCandidate
from rca.execution.evidence_builder import build_candidate_evidence, build_time_evidence, build_reason_evidence
from .schemas import CandidateHypothesis, CounterfactualCheck, DeterministicEvidence
from .utils import ensure_list, normalize_score, safe_json_loads


def deduplicate_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        s = str(item)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _top_metric_events(rows: List[Dict[str, Any]], *, limit: int = 5) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for row in list(rows or []):
        if not isinstance(row, Mapping):
            continue
        ratio = _safe_float(row.get("ratio", 0.0), 0.0)
        events.append(
            {
                "minute": str(row.get("minute", "") or ""),
                "metric": str(row.get("metric", "") or ""),
                "value": row.get("value", ""),
                "baseline": row.get("baseline", ""),
                "ratio": round(float(ratio), 3),
                "first_breach": bool(row.get("is_first_breach", False)),
            }
        )
    events.sort(key=lambda x: (bool(x.get("first_breach")), _safe_float(x.get("ratio", 0.0))), reverse=True)
    return events[: int(limit)]


def _top_trace_events(rows: List[Dict[str, Any]], *, limit: int = 3) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for row in list(rows or []):
        if not isinstance(row, Mapping):
            continue
        events.append(
            {
                "minute": str(row.get("minute", "") or ""),
                "ratio": round(_safe_float(row.get("ratio", 0.0), 0.0), 3),
                "p95": row.get("p95", ""),
                "baseline": row.get("baseline", ""),
            }
        )
    events.sort(key=lambda x: _safe_float(x.get("ratio", 0.0)), reverse=True)
    return events[: int(limit)]


def _reason_rules(dataset_name: str) -> List[str]:
    adapter = get_dataset_adapter(str(dataset_name or ""))
    if adapter is not None:
        try:
            rules = [str(x) for x in list(adapter.llm_reason_rules() or []) if str(x).strip()]
            if rules:
                return rules
        except Exception:
            pass
    return [
        "Prefer candidates whose reason label is semantically consistent with their anomalous metrics.",
        "Prefer upstream/root components over downstream symptoms when causal evidence is available.",
    ]


def _compact_candidate_evidence(ev: Dict[str, Any], *, candidate_id: int) -> Dict[str, Any]:
    causal = dict(ev.get("causal_evidence", {}) or {})
    return {
        "id": int(candidate_id),
        "component": str(ev.get("component", "") or ""),
        "reason": str(ev.get("reason", "") or ""),
        "datetime": str(ev.get("datetime", "") or ""),
        # Keep the index score internal. The LLM should judge causal evidence,
        # not simply follow the deterministic rank.
        "evidence_for": [str(x) for x in list(ev.get("evidence_for") or [])[:2]],
        "evidence_against": [str(x) for x in list(ev.get("evidence_against") or [])[:2]],
        "key_metric_events": _top_metric_events(list(ev.get("metric_timeline") or []), limit=3),
        "key_trace_events": _top_trace_events(list(ev.get("trace_timeline") or []), limit=1),
        "causal_position": {
            "first_anomaly_among_candidates": bool(causal.get("first_anomaly_among_candidates", False)),
            "upstream_of": [str(x) for x in list(causal.get("upstream_of") or [])[:2]],
            "impacted_by": [str(x) for x in list(causal.get("impacted_by") or [])[:2]],
        },
    }


def _intervention_view(compact_ev: Dict[str, Any]) -> Dict[str, Any]:
    evidence = []
    evidence.extend([str(x) for x in list(compact_ev.get("evidence_for") or [])])
    evidence.extend([str(x) for x in list(compact_ev.get("candidate_summary_evidence") or [])])
    blob = " ".join(evidence).lower()
    ctype = ""
    owned = 0.0
    sibling = 0.0
    causal = 0.0
    symptom = 0.0
    metric_count = 0
    for item in evidence:
        text = str(item)
        if text.startswith("component_type="):
            ctype = text.split("=", 1)[1].strip().lower()
        if text.startswith("topology_context="):
            try:
                topo = json.loads(text.split("=", 1)[1])
                if isinstance(topo, dict):
                    if bool(topo.get("shared_infra", False)):
                        symptom = max(symptom, 0.45)
                    fanout = float(topo.get("fanout", 0) or 0)
                    if fanout >= 2:
                        symptom = max(symptom, min(0.75, 0.35 + 0.08 * fanout))
                    if not ctype and str(topo.get("role", "") or "").strip():
                        ctype = str(topo.get("role", "") or "").strip().lower()
            except Exception:
                pass
        if text.startswith("supporting_metrics="):
            metric_count = max(metric_count, len([x for x in text.split("=", 1)[1].split(",") if x.strip()]))
        if text.startswith("rootness_features="):
            for part in text.split("=", 1)[1].split(","):
                if "=" not in part:
                    continue
                key, val = part.split("=", 1)
                try:
                    fval = float(val)
                except Exception:
                    continue
                key = key.strip()
                if key in {"instance_owned_score", "ownership"}:
                    owned = max(owned, fval)
                elif key == "sibling_contrast":
                    sibling = max(sibling, fval)
                elif key in {"causal_source_score", "outgoing_propagation"}:
                    causal = max(causal, fval)
                elif key == "incoming_symptomness":
                    symptom = max(symptom, fval)
    if ctype == "node":
        symptom = min(1.0, symptom + 0.30)
    reason = str(compact_ev.get("reason", "") or "").lower()
    reason_metric = 0.0
    if reason:
        if "cpu" in reason and "cpu" in blob:
            reason_metric = 1.0
        elif "memory" in reason and any(tok in blob for tok in ("mem", "memory", "rss")):
            reason_metric = 1.0
        elif any(tok in reason for tok in ("network", "packet", "latency", "loss")) and any(tok in blob for tok in ("network", "packet", "tcp", "retrans", "latency")):
            if "loss" in reason and "loss" in blob:
                reason_metric = 1.0
            elif any(tok in reason for tok in ("retrans", "corruption", "corrupt")) and any(tok in blob for tok in ("retrans", "corrupt")):
                reason_metric = 1.0
            elif "latency" in reason and "latency" in blob:
                reason_metric = 1.0
            else:
                reason_metric = 0.65
        elif any(tok in reason for tok in ("read", "write", "disk", "i/o", "io", "space")) and any(tok in blob for tok in ("read", "write", "disk", "io", "fs_")):
            if "read" in reason and any(tok in blob for tok in ("read", "r_s", "rkb", "r_await")):
                reason_metric = 1.0
            elif "write" in reason and any(tok in blob for tok in ("write", "w_s", "wkb", "w_await")):
                reason_metric = 1.0
            elif "space" in reason and any(tok in blob for tok in ("space", "usage", "used", "fs_")):
                reason_metric = 1.0
            else:
                reason_metric = 0.55
    local = min(1.0, owned + 0.35 * sibling + 0.20 * min(1.0, metric_count / 6.0))
    intervention_score = max(0.0, min(1.0, 0.45 * local + 0.22 * reason_metric + 0.18 * causal - 0.25 * symptom + 0.15))
    return {
        "component_type": ctype,
        "local_evidence": round(float(local), 3),
        "reason_metric_match": round(float(reason_metric), 3),
        "causal_source": round(float(causal), 3),
        "symptom_risk": round(float(symptom), 3),
        "intervention_score": round(float(intervention_score), 3),
    }


def _field_evidence_card(compact_ev: Dict[str, Any]) -> Dict[str, Any]:
    """Build a small field-level card so the LLM judges evidence, not rank."""

    view = dict(compact_ev.get("intervention_view", {}) or {})
    summary = [str(x) for x in list(compact_ev.get("candidate_summary_evidence") or []) if str(x or "").strip()]
    evidence_for = [str(x) for x in list(compact_ev.get("evidence_for") or []) if str(x or "").strip()]
    evidence_blob = " | ".join(summary + evidence_for)

    coverage_reason = ""
    for item in summary:
        if item.startswith("llm_challenger_reason="):
            coverage_reason = item.split("=", 1)[1].strip()
            break
    is_llm_only = any(item == "llm_only_challenger=true" for item in summary)

    component = str(compact_ev.get("component", "") or "")
    reason = str(compact_ev.get("reason", "") or "")
    datetime_s = str(compact_ev.get("datetime", "") or "")
    topo_note = ""
    for item in summary:
        if item.startswith("topology_context="):
            try:
                topo = json.loads(item.split("=", 1)[1])
                if isinstance(topo, dict):
                    topo_note = str(topo.get("note", "") or "")
            except Exception:
                topo_note = ""
            break

    return {
        "component_claim": {
            "value": component,
            "component_type": str(view.get("component_type", "") or ""),
            "local_evidence": float(view.get("local_evidence", 0.0) or 0.0),
            "causal_source": float(view.get("causal_source", 0.0) or 0.0),
            "symptom_risk": float(view.get("symptom_risk", 0.0) or 0.0),
            "topology_note": topo_note[:180],
        },
        "reason_claim": {
            "value": reason,
            "reason_metric_match": float(view.get("reason_metric_match", 0.0) or 0.0),
            "support_hint": "reason-compatible metrics present"
            if float(view.get("reason_metric_match", 0.0) or 0.0) > 0.0
            else "reason support not visible",
        },
        "time_claim": {
            "value": datetime_s,
            "has_time": bool(datetime_s),
        },
        "coverage_claim": {
            "llm_only_challenger": bool(is_llm_only),
            "why_included": coverage_reason,
        },
        "evidence_excerpt": evidence_blob[:420],
    }


def _explainable_evidence_snippet(compact_ev: Mapping[str, Any]) -> Dict[str, Any]:
    summary = [str(x) for x in list(compact_ev.get("candidate_summary_evidence") or []) if str(x or "").strip()]
    evidence_for = [str(x) for x in list(compact_ev.get("evidence_for") or []) if str(x or "").strip()]
    metrics: List[str] = []
    rootness = ""
    topology: Dict[str, Any] = {}
    challenger_reason = ""
    for item in summary:
        if item.startswith("supporting_metrics="):
            metrics = [x.strip() for x in item.split("=", 1)[1].split(",") if x.strip()][:8]
        elif item.startswith("rootness_features="):
            rootness = item.split("=", 1)[1].strip()[:240]
        elif item.startswith("topology_context="):
            try:
                parsed = json.loads(item.split("=", 1)[1])
                if isinstance(parsed, dict):
                    topology = {
                        "role": str(parsed.get("role", "") or ""),
                        "shared_infra": bool(parsed.get("shared_infra", False)),
                        "fanout": parsed.get("fanout", 0),
                        "fanin": parsed.get("fanin", 0),
                        "upstream": [str(x) for x in list(parsed.get("upstream", []) or [])[:3]],
                        "downstream": [str(x) for x in list(parsed.get("downstream", []) or [])[:3]],
                        "note": str(parsed.get("note", "") or "")[:180],
                    }
            except Exception:
                topology = {}
        elif item.startswith("llm_challenger_reason="):
            challenger_reason = item.split("=", 1)[1].strip()

    return {
        "component": str(compact_ev.get("component", "") or ""),
        "reason": str(compact_ev.get("reason", "") or ""),
        "datetime": str(compact_ev.get("datetime", "") or ""),
        "metrics": metrics,
        "rootness_features": rootness,
        "topology": topology,
        "why_in_prompt": challenger_reason,
        "supporting_text": (evidence_for + summary)[:5],
    }


def _pairwise_counterfactual_tests(candidate_payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare current top-1 against each challenger in evidence space."""

    if not candidate_payload:
        return []
    top1 = dict(candidate_payload[0] or {})
    top_view = dict(top1.get("intervention_view", {}) or {})
    tests: List[Dict[str, Any]] = []

    def f(view: Mapping[str, Any], key: str) -> float:
        try:
            return float(view.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    top_local = f(top_view, "local_evidence")
    top_reason = f(top_view, "reason_metric_match")
    top_causal = f(top_view, "causal_source")
    top_symptom = f(top_view, "symptom_risk")
    top_intervention = f(top_view, "intervention_score")

    for cand in candidate_payload[1:]:
        view = dict(cand.get("intervention_view", {}) or {})
        local_delta = f(view, "local_evidence") - top_local
        reason_delta = f(view, "reason_metric_match") - top_reason
        causal_delta = f(view, "causal_source") - top_causal
        symptom_delta = f(view, "symptom_risk") - top_symptom
        intervention_delta = f(view, "intervention_score") - top_intervention

        challenger_advantages: List[str] = []
        top1_advantages: List[str] = []
        if local_delta >= 0.15:
            challenger_advantages.append("stronger component-local evidence")
        elif local_delta <= -0.15:
            top1_advantages.append("stronger component-local evidence")
        if reason_delta >= 0.50:
            challenger_advantages.append("reason better supported by metrics")
        elif reason_delta <= -0.50:
            top1_advantages.append("reason better supported by metrics")
        if causal_delta >= 0.15:
            challenger_advantages.append("more plausible causal source")
        elif causal_delta <= -0.15:
            top1_advantages.append("more plausible causal source")
        if symptom_delta <= -0.20:
            challenger_advantages.append("lower downstream/global symptom risk")
        elif symptom_delta >= 0.20:
            top1_advantages.append("lower downstream/global symptom risk")

        suggested = "keep_top1"
        if len(challenger_advantages) >= 2 and intervention_delta >= -0.05:
            suggested = "challenge_top1"
        elif intervention_delta >= 0.12 and challenger_advantages:
            suggested = "challenge_top1"

        tests.append(
            {
                "challenger_id": int(cand.get("id", 0) or 0),
                "challenger_component": str(cand.get("component", "") or ""),
                "challenger_reason": str(cand.get("reason", "") or ""),
                "challenger_datetime": str(cand.get("datetime", "") or ""),
                "deltas_vs_top1": {
                    "local_evidence": round(float(local_delta), 3),
                    "reason_metric_match": round(float(reason_delta), 3),
                    "causal_source": round(float(causal_delta), 3),
                    "symptom_risk": round(float(symptom_delta), 3),
                    "intervention_score": round(float(intervention_delta), 3),
                },
                "challenger_advantages": challenger_advantages,
                "top1_advantages": top1_advantages,
                "suggested_test_result": suggested,
                "falsifies_top1": bool(suggested == "challenge_top1"),
                "evidence_comparison": {
                    "top1": _explainable_evidence_snippet(top1),
                    "challenger": _explainable_evidence_snippet(cand),
                },
            }
        )
    tests.sort(
        key=lambda x: (
            bool(x.get("falsifies_top1", False)),
            float(dict(x.get("deltas_vs_top1", {}) or {}).get("intervention_score", 0.0) or 0.0),
            len(list(x.get("challenger_advantages") or [])),
        ),
        reverse=True,
    )
    # Keep the prompt bounded.  The LLM only needs the strongest pairwise
    # interventions, not every low-evidence challenger.
    return tests[:4]


def _compare_two_candidates(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Generic pairwise comparison card between any two candidate payloads.

    Same heuristics as `_pairwise_counterfactual_tests` (which is hardwired to
    "top1 vs every challenger"), but usable for an arbitrary pair -- this is
    what the multi-turn agent loop's `compare` action calls on demand."""

    view_a = dict(a.get("intervention_view", {}) or {})
    view_b = dict(b.get("intervention_view", {}) or {})

    def f(view: Mapping[str, Any], key: str) -> float:
        try:
            return float(view.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    deltas = {
        "local_evidence": round(f(view_b, "local_evidence") - f(view_a, "local_evidence"), 3),
        "reason_metric_match": round(f(view_b, "reason_metric_match") - f(view_a, "reason_metric_match"), 3),
        "causal_source": round(f(view_b, "causal_source") - f(view_a, "causal_source"), 3),
        "symptom_risk": round(f(view_b, "symptom_risk") - f(view_a, "symptom_risk"), 3),
        "intervention_score": round(f(view_b, "intervention_score") - f(view_a, "intervention_score"), 3),
    }
    b_advantages: List[str] = []
    a_advantages: List[str] = []
    if deltas["local_evidence"] >= 0.15:
        b_advantages.append("stronger component-local evidence")
    elif deltas["local_evidence"] <= -0.15:
        a_advantages.append("stronger component-local evidence")
    if deltas["reason_metric_match"] >= 0.50:
        b_advantages.append("reason better supported by metrics")
    elif deltas["reason_metric_match"] <= -0.50:
        a_advantages.append("reason better supported by metrics")
    if deltas["causal_source"] >= 0.15:
        b_advantages.append("more plausible causal source")
    elif deltas["causal_source"] <= -0.15:
        a_advantages.append("more plausible causal source")
    if deltas["symptom_risk"] <= -0.20:
        b_advantages.append("lower downstream/global symptom risk")
    elif deltas["symptom_risk"] >= 0.20:
        a_advantages.append("lower downstream/global symptom risk")

    suggested = "keep_a"
    if len(b_advantages) >= 2 and deltas["intervention_score"] >= -0.05:
        suggested = "prefer_b"
    elif deltas["intervention_score"] >= 0.12 and b_advantages:
        suggested = "prefer_b"

    return {
        "a_id": int(a.get("id", 0) or 0),
        "b_id": int(b.get("id", 0) or 0),
        "deltas_b_minus_a": deltas,
        "b_advantages": b_advantages,
        "a_advantages": a_advantages,
        "suggested": suggested,
        "evidence_comparison": {
            "a": _explainable_evidence_snippet(a),
            "b": _explainable_evidence_snippet(b),
        },
    }


class CounterfactualExecutor:
    """
    Executor 负责具体动作执行：
    1. 提取候选根因
    2. 为候选生成反事实问题
    3. 对候选进行反事实验证（LLM + deterministic evidence）
    4. 汇总最终结论
    """

    def __init__(self, agent_prompt: str, basic_prompt: str, logger=None):
        self.agent_prompt = agent_prompt
        self.basic_prompt = basic_prompt
        self.logger = logger
        self.last_llm_call: Dict[str, Any] = {}

    @staticmethod
    def estimate_input_tokens(messages: List[Dict[str, str]]) -> int:
        text = "\n".join(str(m.get("role", "")) + "\n" + str(m.get("content", "")) for m in messages)
        return max(1, int(len(text) / 4))

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(user_prompt)},
        ]
        input_tokens = self.estimate_input_tokens(messages)
        self.last_llm_call = {
            "approx_input_tokens": int(input_tokens),
            "elapsed_seconds": 0.0,
            "attempted": True,
        }
        if self.logger:
            self.logger.info(f"[Executor] Sending prompt to LLM... approx_input_tokens={input_tokens}")

        t0 = time.perf_counter()
        try:
            resp = get_chat_completion(messages, temperature=0.0)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.last_llm_call = {
                "approx_input_tokens": int(input_tokens),
                "elapsed_seconds": float(elapsed),
                "attempted": True,
                "error": str(e),
            }
            if self.logger:
                self.logger.info(f"[Executor] LLM request failed. elapsed_seconds={elapsed:.6f}, error={e}")
            raise
        elapsed = time.perf_counter() - t0
        content = str(resp)
        self.last_llm_call = {
            "approx_input_tokens": int(input_tokens),
            "elapsed_seconds": float(elapsed),
            "attempted": True,
        }

        if self.logger:
            self.logger.info(f"[Executor] LLM response received. elapsed_seconds={elapsed:.6f}")
        return content

    @staticmethod
    def _dedupe_candidate_pool(cands: List[CandidateHypothesis]) -> List[CandidateHypothesis]:
        """Collapse exact (component, reason, datetime) duplicates, keeping the
        highest-scored copy at its first-seen position.

        Candidate expansion can legitimately emit the same tuple more than once
        from different expansion paths (e.g. dimension recombination), which
        just wastes menu slots and dilutes the agent-loop's attention across
        rows that carry zero additional information."""
        best: Dict[tuple, CandidateHypothesis] = {}
        order: List[tuple] = []
        for cand in cands:
            key = (str(cand.component), str(cand.reason), str(getattr(cand, "datetime", "") or ""))
            if key not in best:
                best[key] = cand
                order.append(key)
            elif float(cand.score) > float(best[key].score):
                best[key] = cand
        return [best[k] for k in order]

    def _build_candidate_payload_entry(
        self,
        *,
        cand: CandidateHypothesis,
        candidate_id: int,
        next_score: float,
        instruction: str,
        dataset_name: str,
        context: Dict[str, Any],
        all_components: List[str],
    ) -> Dict[str, Any]:
        """Build one candidate's compact evidence card (component/reason/datetime
        summary + metric/trace/topology digest). Shared by the single-shot
        bounded verifier and the multi-turn agent loop's on-demand `inspect`
        action, so evidence is computed identically in both paths."""
        enable_trace = bool(context.get("enable_trace_evidence", True))
        if not bool(context.get("llm_build_raw_evidence", False)):
            ev = {
                "component": str(cand.component),
                "reason": str(cand.reason),
                "datetime": str(getattr(cand, "datetime", "") or ""),
                "score": float(cand.score),
                "score_gap_vs_next": max(0.0, float(cand.score) - float(next_score)),
                "metric_timeline": [],
                "trace_timeline": [],
                "causal_evidence": {},
                "evidence_for": [str(x) for x in list(cand.evidence or [])[:6]],
                "evidence_against": [],
            }
        else:
            try:
                ev = build_candidate_evidence(
                    dataset=dataset_name,
                    instruction=instruction,
                    component=str(cand.component),
                    reason=str(cand.reason),
                    score=float(cand.score),
                    score_gap_vs_next=max(0.0, float(cand.score) - float(next_score)),
                    all_candidate_components=all_components,
                    topology_max_rows=context.get("basil_max_trace_rows"),
                    enable_trace=bool(enable_trace),
                ).to_dict()
            except Exception:
                ev = {
                    "component": str(cand.component),
                    "reason": str(cand.reason),
                    "datetime": str(getattr(cand, "datetime", "") or ""),
                    "score": float(cand.score),
                    "score_gap_vs_next": max(0.0, float(cand.score) - float(next_score)),
                    "metric_timeline": [],
                    "trace_timeline": [],
                    "causal_evidence": {},
                    "evidence_for": [str(x) for x in list(cand.evidence or [])[:6]],
                    "evidence_against": ["Evidence builder failed; fallback to indexed candidate summary."],
                }
        compact_ev = _compact_candidate_evidence(ev, candidate_id=candidate_id)
        compact_ev["datetime"] = str(getattr(cand, "datetime", "") or compact_ev.get("datetime", "") or "")
        if list(getattr(cand, "evidence", []) or []):
            compact_ev["candidate_summary_evidence"] = [str(x) for x in list(getattr(cand, "evidence", []) or [])[:8]]
        compact_ev["intervention_view"] = _intervention_view(compact_ev)
        compact_ev["field_evidence"] = _field_evidence_card(compact_ev)
        return compact_ev

    def select_best_candidate_with_llm(
        self,
        instruction: str,
        candidates: List[CandidateHypothesis],
        dataset_name: str = "",
        task_focus: str = "",
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        context = context or {}
        if bool(context.get("force_llm_verify", False)):
            # Experimental "give the agent every candidate" mode: no top-8 cap.
            max_prompt_candidates = max(1, int(context.get("llm_max_prompt_candidates", len(candidates or [])) or len(candidates or [])))
        else:
            max_prompt_candidates = min(8, max(1, int(context.get("llm_max_prompt_candidates", top_k) or top_k)))
        top = list(candidates or [])[: max_prompt_candidates]
        if not top:
            raise ValueError("No candidates for LLM verification.")

        candidate_payload: List[Dict[str, Any]] = []
        all_components = [str(c.component) for c in list(candidates or []) if str(getattr(c, "component", "") or "").strip()]
        for i, cand in enumerate(top, start=1):
            next_score = float(top[i].score) if i < len(top) else 0.0
            candidate_payload.append(
                self._build_candidate_payload_entry(
                    cand=cand,
                    candidate_id=i,
                    next_score=next_score,
                    instruction=instruction,
                    dataset_name=dataset_name,
                    context=context,
                    all_components=all_components,
                )
            )

        uncertainty = dict(context.get("uncertainty_gate", {}) or {})
        evidence_conflict = dict(uncertainty.get("evidence_conflict", {}) or {})
        min_conf = _safe_float(context.get("llm_change_min_confidence", 0.85), 0.85)
        max_drop = _safe_float(context.get("llm_max_score_drop_vs_det_top1", 0.05), 0.05)

        pairwise_tests = _pairwise_counterfactual_tests(candidate_payload)

        system_prompt = """
You are a counterfactual RCA evidence judge. Do bounded reranking only.
Do not invent components, reasons, timestamps, or evidence.

Your decision must be a two-stage intervention test, not a generic ranking.
Candidate #1 is only the deterministic current answer; it is not a truth label.

Stage A: falsify candidate #1.
Return top1_falsified=true only if a challenger provides a more specific root
intervention that explains the required field(s) better than candidate #1.

Stage B: select the minimal intervention.
If top1 is falsified, select the challenger whose removal would best explain
the local anomaly, reason-compatible metrics, propagation, and time evidence.
If top1 is not falsified, selected_id must be 1.

Reject candidates supported mainly by global node symptoms, weak early onset,
unsupported reason labels, or duplicated symptom evidence.

Important decision policy:
- If pairwise_tests marks a challenger as falsifies_top1 and the concrete
  evidence supports it, set top1_falsified=true and select that challenger.
- If candidate #1 and a challenger each win different fields, use field_decisions
  to adopt only the fields with clear evidence gain.
- Do not keep candidate #1 merely because it is candidate #1.
- Do not select a challenger merely because it covers a missing label family.

Some challengers are intentionally included to cover missing evidence dimensions
(component type, reason family, or time cluster). Treat them as serious
alternatives only if their concrete evidence supports that dimension; do not
discard them merely because they are lower ranked.

Use the intervention_view fields as a compact guide:
- local_evidence: component-local owned metrics and sibling contrast.
- reason_metric_match: whether the reason is supported by component metrics.
- causal_source: whether the component can explain downstream symptoms.
- symptom_risk: high values mean likely downstream/global symptom.
- intervention_score: evidence-only estimate; use it to compare challengers.

Use topology_context when present:
- A shared infrastructure/node candidate with high fanout is often a symptom
  surface. Do not choose it over a service/container on the affected path unless
  the node has clearly stronger local evidence and reason-compatible metrics.
- A service/container with local metrics and topology connection to symptoms is
  a stronger intervention target than a broad node symptom.

When pairwise_tests contains evidence_comparison, use the concrete metrics and
topology notes as the primary evidence. If numeric summaries conflict with the
metric/topology text, explain the conflict and prefer the candidate whose
component, reason, and time are directly supported by concrete evidence.

Output JSON only. In field_decisions, judge only fields required by the task.
Use "evidence_gain" to state how much stronger the selected candidate is for
that field compared with candidate #1: 0 means no gain, 1 means decisive gain.
{"top1_falsified":false,"selected_id":1,"confidence":0.0,"change_from_top1":false,
"top1_intervention_score":0.0,"selected_intervention_score":0.0,
"falsification_reason":"why top1 is or is not falsified",
"rationale":"one sentence","decisive_evidence":"one concrete evidence item",
"field_decisions":{
  "component":{"adopt":false,"evidence_gain":0.0,"reason":"short evidence comparison"},
  "reason":{"adopt":false,"evidence_gain":0.0,"reason":"short evidence comparison"},
  "datetime":{"adopt":false,"evidence_gain":0.0,"reason":"short evidence comparison"}
},
"rejected_candidates":[{"id":2,"reason":"short reason"}]}
"""

        concise_conflict = {
            "reasons": ensure_list(uncertainty.get("reasons"))[:3],
            "score_gap": uncertainty.get("score_gap"),
            "conflict_dimensions": ensure_list(evidence_conflict.get("conflict_dimensions"))[:4],
            "top1_profile": evidence_conflict.get("top1_profile", {}),
            "challenger_profile": evidence_conflict.get("challenger_profile", {}),
        }
        incident_summary = {
            "dataset": str(dataset_name or ""),
            "task_focus": str(task_focus or "")[:240],
            "query": str(instruction or "")[:420],
            "adoption": {
                "min_confidence_to_replace_top1": round(float(min_conf), 3),
                "max_score_drop_to_replace_top1": round(float(max_drop), 4),
            },
            "conflict": concise_conflict,
            "coverage_agent": dict((context or {}).get("coverage_aware_agent") or {}),
            "intervention_agent": dict((context or {}).get("intervention_agent") or {}),
            "reason_rules": _reason_rules(dataset_name)[:2],
            "deterministic_top": candidate_payload[0] if candidate_payload else {},
            "challengers": candidate_payload[1:],
            "pairwise_tests": pairwise_tests,
        }
        user_prompt = json.dumps(incident_summary, ensure_ascii=False, separators=(",", ":"))

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(
            raw,
            {
                "selected_id": 1,
                "confidence": float(top[0].score),
                "rationale": "Fallback to top-1 because the LLM output is invalid.",
                "supporting_evidence": [],
            },
        )
        try:
            selected_id = int(data.get("selected_id", 1))
        except Exception:
            selected_id = 1
        if selected_id < 1 or selected_id > len(top):
            selected_id = 1
        selected = top[selected_id - 1]
        details = {
            "selected_id": int(selected_id),
            "selected_candidate": selected.to_dict(),
            "confidence": normalize_score(data.get("confidence", selected.score)),
            "rationale": str(data.get("rationale", "")).strip(),
            "conflict_resolution": str(data.get("conflict_resolution", "") or "").strip(),
            "change_from_top1": bool(selected_id != 1 or data.get("change_from_top1", False)),
            # An objective fact (a different candidate id was actually
            # selected) must not be overridden by the model's own,
            # separately-filled top1_falsified flag saying otherwise -- that
            # flag can only ever add True, never take away a real difference.
            # Seen on a real Bank run: the model picked a different component
            # with strong, well-reasoned evidence but still wrote
            # top1_falsified=false, which used to silently block every field
            # from being adopted regardless of how good the evidence was.
            "top1_falsified": bool(selected_id != 1 or data.get("top1_falsified", False)),
            "falsification_reason": str(data.get("falsification_reason", "") or "").strip(),
            "top1_intervention_score": normalize_score(data.get("top1_intervention_score", 0.0)),
            "selected_intervention_score": normalize_score(data.get("selected_intervention_score", 0.0)),
            "decisive_evidence": str(data.get("decisive_evidence", "") or "").strip(),
            "field_decisions": dict(data.get("field_decisions", {}) or {}) if isinstance(data.get("field_decisions", {}), Mapping) else {},
            "rejected_candidates": ensure_list(data.get("rejected_candidates")),
            "supporting_evidence": ensure_list(data.get("supporting_evidence")) or ensure_list(data.get("decisive_evidence")),
            "raw_response": raw,
            "approx_input_tokens": int(self.last_llm_call.get("approx_input_tokens", 0) or 0),
            "elapsed_seconds": float(self.last_llm_call.get("elapsed_seconds", 0.0) or 0.0),
            "prompt_candidate_count": int(len(candidate_payload)),
        }
        return {"candidate": selected, "details": details}

    def select_best_candidate_with_agent_loop(
        self,
        instruction: str,
        candidates: List[CandidateHypothesis],
        dataset_name: str = "",
        task_focus: str = "",
        context: Optional[Dict[str, Any]] = None,
        max_turns: int = 5,
    ) -> Dict[str, Any]:
        """Multi-turn (ReAct-style) verifier over the FULL candidate pool.

        Unlike `select_best_candidate_with_llm` (single-shot, capped at 8
        candidates, full evidence pre-computed for every candidate up front),
        this hands the LLM a cheap id/component/reason/datetime/score menu for
        ALL candidates and lets it decide, turn by turn, which candidates are
        worth pulling full evidence for (`inspect`), which pair is worth a
        direct comparison (`compare`), and when it has enough to commit
        (`decide`). Evidence is computed lazily, on demand, instead of for
        every candidate regardless of relevance.

        Bounded by `max_turns` real LLM calls; if the budget runs out without
        a `decide` action, this falls back to the deterministic top-1 exactly
        like the bounded verifier's own fallback path.
        """
        context = context or {}
        pool = self._dedupe_candidate_pool(list(candidates or []))
        if not pool:
            raise ValueError("No candidates for agent-loop verification.")

        all_components = [str(c.component) for c in pool if str(getattr(c, "component", "") or "").strip()]

        def _cheap_hint(cand: CandidateHypothesis) -> Dict[str, float]:
            # Free triage signal: parsed straight out of the evidence strings
            # already attached to the candidate (rootness_features=,
            # supporting_metrics=, topology_context=), no telemetry lookup.
            # Without this, the menu is bare index scores and the LLM has no
            # way to tell "Redis02 + high memory usage" apart from five other
            # Redis02 + <other reason> rows with similar scores.
            compact_ev = {
                "reason": str(cand.reason),
                "candidate_summary_evidence": [str(x) for x in list(getattr(cand, "evidence", []) or [])[:8]],
                "evidence_for": [],
            }
            view = _intervention_view(compact_ev)
            return {
                "reason_metric_match": float(view.get("reason_metric_match", 0.0) or 0.0),
                "intervention_score": float(view.get("intervention_score", 0.0) or 0.0),
                "symptom_risk": float(view.get("symptom_risk", 0.0) or 0.0),
            }

        def _menu_entry(cand: CandidateHypothesis, idx: int) -> Dict[str, Any]:
            hint = _cheap_hint(cand)
            return {
                "id": int(idx),
                "component": str(cand.component),
                "reason": str(cand.reason),
                "datetime": str(getattr(cand, "datetime", "") or ""),
                "score": round(float(cand.score), 4),
                "source": str(getattr(cand, "source", "") or ""),
                "reason_metric_match": round(hint["reason_metric_match"], 3),
                "intervention_score": round(hint["intervention_score"], 3),
                "symptom_risk": round(hint["symptom_risk"], 3),
            }

        menu = [_menu_entry(c, i) for i, c in enumerate(pool, start=1)]
        inspected: Dict[int, Dict[str, Any]] = {}
        comparisons: List[Dict[str, Any]] = []
        actions_log: List[Dict[str, Any]] = []
        total_input_tokens = 0
        total_elapsed = 0.0

        def _payload_for(idx: int) -> Optional[Dict[str, Any]]:
            if idx in inspected:
                return inspected[idx]
            if idx < 1 or idx > len(pool):
                return None
            cand = pool[idx - 1]
            next_score = float(pool[idx].score) if idx < len(pool) else 0.0
            entry = self._build_candidate_payload_entry(
                cand=cand,
                candidate_id=idx,
                next_score=next_score,
                instruction=instruction,
                dataset_name=dataset_name,
                context=context,
                all_components=all_components,
            )
            inspected[idx] = entry
            return entry

        system_prompt = """
You are a counterfactual RCA evidence judge working in a bounded multi-turn loop.
You are given a MENU of ALL candidate root causes (id, component, reason,
datetime, deterministic score only -- no detailed evidence yet). Candidate #1
is only the deterministic current answer; it is not a truth label.

Each turn you must return exactly one JSON action, nothing else:
{"action":"inspect","id":<int>}            -- pull detailed evidence for one candidate
{"action":"compare","ids":[<int>,<int>]}    -- pairwise evidence comparison of two candidates
{"action":"decide", ...}                    -- commit to a final answer (schema below)

Each menu row also has three free, already-computed hints (no evidence fetch
needed to see these): reason_metric_match (does the reason label match the
kind of metrics typically seen for it), intervention_score (rough evidence-only
plausibility as a root cause), symptom_risk (high means this candidate looks
like a downstream/global symptom rather than a root cause). These are cheap
heuristics, not proof -- use them to decide what to `inspect`, not as the
final answer. A candidate with high intervention_score / low symptom_risk
but a lower raw score is often worth inspecting even if it is not near the
top of the menu -- do not only look at the top few rows by score.

Guidance:
- Triage from the menu first using score AND the three hints together; do not
  rely on raw score alone -- several rows can share a component with very
  similar scores but only one reason variant is actually evidence-supported.
- Only `inspect` candidates that could plausibly change the answer; do not
  inspect every candidate on the menu.
- Use `compare` once you have inspected two specific candidates you are
  genuinely torn between.
- You have a limited number of turns. Decide as soon as you have enough
  evidence; do not spend turns you do not need.
- Do not invent components, reasons, timestamps, or evidence beyond what the
  menu and the evidence you have inspected show.

When ready, return "decide" with:
{"action":"decide","top1_falsified":false,"selected_id":1,"confidence":0.0,
"change_from_top1":false,"top1_intervention_score":0.0,"selected_intervention_score":0.0,
"falsification_reason":"why top1 is or is not falsified",
"rationale":"one sentence","decisive_evidence":"one concrete evidence item",
"field_decisions":{
  "component":{"adopt":false,"evidence_gain":0.0,"reason":"short evidence comparison"},
  "reason":{"adopt":false,"evidence_gain":0.0,"reason":"short evidence comparison"},
  "datetime":{"adopt":false,"evidence_gain":0.0,"reason":"short evidence comparison"}
}}

Output JSON only, exactly one action per turn.
"""

        final_data: Dict[str, Any] = {}
        turns_used = 0
        for turn in range(1, max(1, int(max_turns)) + 1):
            turns_used = turn
            turns_remaining = max(0, int(max_turns) - turn)
            user_payload = {
                "dataset": str(dataset_name or ""),
                "task_focus": str(task_focus or "")[:240],
                "query": str(instruction or "")[:420],
                "turn": int(turn),
                "turns_remaining": int(turns_remaining),
                "candidate_menu": menu,
                "inspected_candidates": [inspected[i] for i in sorted(inspected.keys())],
                "comparisons_done": comparisons,
            }
            user_prompt = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
            raw = self._chat(system_prompt, user_prompt)
            total_input_tokens += int(self.last_llm_call.get("approx_input_tokens", 0) or 0)
            total_elapsed += float(self.last_llm_call.get("elapsed_seconds", 0.0) or 0.0)
            data = safe_json_loads(raw, {"action": "decide", "selected_id": 1, "confidence": float(pool[0].score)})
            if not isinstance(data, Mapping):
                data = {"action": "invalid"}
            action = str(data.get("action", "") or "").strip().lower()
            actions_log.append({"turn": int(turn), "action": action or "invalid"})

            if action == "inspect":
                try:
                    idx = int(data.get("id", 0) or 0)
                except Exception:
                    idx = 0
                _payload_for(idx)
                continue
            if action == "compare":
                ids = data.get("ids") or []
                try:
                    id_a, id_b = int(ids[0]), int(ids[1])
                except Exception:
                    continue
                pa, pb = _payload_for(id_a), _payload_for(id_b)
                if pa is not None and pb is not None:
                    comparisons.append(_compare_two_candidates(pa, pb))
                continue
            if action == "decide":
                final_data = dict(data)
                break
            # Unknown / malformed action: treat as a wasted turn and continue.
            continue

        if not final_data:
            # Ran out of turns without a decision: safe fallback to deterministic top-1.
            final_data = {
                "selected_id": 1,
                "confidence": float(pool[0].score),
                "rationale": "Agent loop exhausted max_turns without deciding; fell back to deterministic top-1.",
                "top1_falsified": False,
                "field_decisions": {},
            }
            actions_log.append({"turn": int(turns_used), "action": "fallback_max_turns_exhausted"})

        try:
            selected_id = int(final_data.get("selected_id", 1))
        except Exception:
            selected_id = 1
        if selected_id < 1 or selected_id > len(pool):
            selected_id = 1
        selected = pool[selected_id - 1]

        details = {
            "selected_id": int(selected_id),
            "selected_candidate": selected.to_dict(),
            "confidence": normalize_score(final_data.get("confidence", selected.score)),
            "rationale": str(final_data.get("rationale", "")).strip(),
            "change_from_top1": bool(selected_id != 1 or final_data.get("change_from_top1", False)),
            "top1_falsified": bool(selected_id != 1 or final_data.get("top1_falsified", False)),
            "falsification_reason": str(final_data.get("falsification_reason", "") or "").strip(),
            "top1_intervention_score": normalize_score(final_data.get("top1_intervention_score", 0.0)),
            "selected_intervention_score": normalize_score(final_data.get("selected_intervention_score", 0.0)),
            "decisive_evidence": str(final_data.get("decisive_evidence", "") or "").strip(),
            "field_decisions": dict(final_data.get("field_decisions", {}) or {}) if isinstance(final_data.get("field_decisions", {}), Mapping) else {},
            "rejected_candidates": ensure_list(final_data.get("rejected_candidates")),
            "supporting_evidence": ensure_list(final_data.get("supporting_evidence")) or ensure_list(final_data.get("decisive_evidence")),
            "approx_input_tokens": int(total_input_tokens),
            "elapsed_seconds": float(total_elapsed),
            "prompt_candidate_count": int(len(pool)),
            "agent_loop_turns_used": int(turns_used),
            "agent_loop_max_turns": int(max_turns),
            "agent_loop_inspected_count": int(len(inspected)),
            "agent_loop_comparisons_count": int(len(comparisons)),
            "agent_loop_actions": actions_log,
        }
        return {"candidate": selected, "details": details}

    def verify_with_tool_agent(
        self,
        instruction: str,
        candidates: List[CandidateHypothesis],
        dataset_name: str = "",
        task_focus: str = "",
        context: Optional[Dict[str, Any]] = None,
        max_turns: int = 6,
    ) -> Dict[str, Any]:
        """General tool-calling RCA verifier: candidate pool = recall, agent = verification.

        The candidate list handed to the LLM here is deliberately score-free --
        just candidate_id/component/reason/datetime. No score, rank, source,
        or pre-computed evidence hint is shown. The point of the candidate
        list is only to maximize the chance the right tuple is present
        somewhere (recall); it must not anchor the LLM's judgment the way a
        sorted, pre-scored menu does. Verification is done entirely through
        real tool calls (`query_evidence`, `query_time`, `compare`) against
        `build_candidate_evidence`/`build_time_evidence`, so the LLM is
        reasoning from actual metric/trace data it fetched itself, not from a
        rank it was shown. It can also query components/reasons that were
        never in the candidate list at all, which is what a fixed-menu
        verifier can never recover from (e.g. row6 in the Bank agent-loop
        experiment, where the ground-truth component never appeared anywhere
        in the 39-candidate pool).

        Same backstop as the other two verifier paths: the proposed answer
        still goes through the controller's field-level evidence-gated
        adoption and adapter.validate_llm_candidate_adoption() veto before it
        can actually replace the deterministic answer.
        """
        context = context or {}
        pool = self._dedupe_candidate_pool(list(candidates or []))
        if not pool:
            raise ValueError("No candidates for tool-agent verification.")

        all_components = [str(c.component) for c in pool if str(getattr(c, "component", "") or "").strip()]
        enable_trace = bool(context.get("enable_trace_evidence", True))

        def _menu_entry(cand: CandidateHypothesis, idx: int) -> Dict[str, Any]:
            # Deliberately no score/rank/source/evidence here -- see docstring.
            return {
                "candidate_id": f"c{idx}",
                "component": str(cand.component),
                "reason": str(cand.reason),
                "datetime": str(getattr(cand, "datetime", "") or ""),
            }

        menu = [_menu_entry(c, i) for i, c in enumerate(pool, start=1)]
        id_to_cand = {f"c{i}": c for i, c in enumerate(pool, start=1)}
        top1 = pool[0]
        query_log: List[Dict[str, Any]] = []
        actions_log: List[Dict[str, Any]] = []
        total_input_tokens = 0
        total_elapsed = 0.0
        compare_count = 0
        reason_checks = 0
        # Stateful pruning: `remaining_ids` is the live working set. Pruning a
        # candidate here actually removes it from every later list_candidates
        # view -- it is not just a note attached to the final decide action,
        # so "repeat until few candidates remain" is a real narrowing loop,
        # not something the agent has to self-report after the fact.
        remaining_ids: List[str] = [f"c{i}" for i in range(1, len(pool) + 1)]
        pruned_log: List[Dict[str, Any]] = []

        # General (not query-specific) safety net for the failure pattern
        # seen on real Bank runs: the agent tests one or two reason
        # hypotheses, gets inconclusive ("weak") support, drifts into
        # comparing an unrelated pair, and then decides on a THIRD
        # (component, reason) it never queried at all -- typically silently
        # reverting to the deterministic top-1 without ever having verified
        # it either. This log records every (component, reason) pair this
        # conversation actually ran query_reason_evidence on (directly, or as
        # either side of compare_reason_candidates), so decide-time can tell
        # a verified answer from an untested guess -- for any dataset/query,
        # not just this one.
        reason_evidence_log: Dict[tuple[str, str], Dict[str, Any]] = {}
        _SUPPORT_RANK = {"strong": 2, "weak": 1, "not_supported": 0}

        def _best_logged_reason_candidate() -> tuple[Optional[tuple[str, str]], Dict[str, Any]]:
            best_key: Optional[tuple[str, str]] = None
            best_ev: Dict[str, Any] = {}
            best_score = (-1, -1.0)
            for key, ev in reason_evidence_log.items():
                score = (_SUPPORT_RANK.get(str(ev.get("support", "")), -1), float(ev.get("reason_deviation", 0.0) or 0.0))
                if score > best_score:
                    best_score = score
                    best_key = key
                    best_ev = ev
            return best_key, best_ev

        def _tool_query_evidence(component: str, reason: str, include_topology: bool = False) -> Dict[str, Any]:
            try:
                # Topology/trace extraction parses the full per-day trace_span.csv,
                # which for datasets like Bank can be multi-GB and take tens of
                # seconds on a first (uncached) call -- far more expensive than the
                # metric-only lookup. Default it off; the agent has to explicitly
                # ask for it via include_topology=true and pay that cost knowingly.
                use_trace = bool(enable_trace) and bool(include_topology)
                ev = build_candidate_evidence(
                    dataset=str(dataset_name or ""),
                    instruction=str(instruction or ""),
                    component=str(component or ""),
                    reason=str(reason or ""),
                    score=0.0,
                    score_gap_vs_next=0.0,
                    all_candidate_components=all_components,
                    topology_max_rows=context.get("basil_max_trace_rows"),
                    enable_trace=bool(use_trace),
                ).to_dict()
                return {
                    "component": str(component or ""),
                    "reason": str(reason or ""),
                    "metric_timeline": list(ev.get("metric_timeline") or [])[:8],
                    "trace_timeline": list(ev.get("trace_timeline") or [])[:5],
                    "causal_evidence": dict(ev.get("causal_evidence") or {}),
                    "evidence_for": list(ev.get("evidence_for") or []),
                    "evidence_against": list(ev.get("evidence_against") or []),
                    "topology_queried": bool(use_trace),
                }
            except Exception as exc:
                return {"error": f"query_evidence failed: {exc}"}

        def _tool_query_time(component: str, reason: str, dt: str, include_topology: bool = False) -> Dict[str, Any]:
            try:
                use_trace = bool(enable_trace) and bool(include_topology)
                tev = build_time_evidence(
                    dataset=str(dataset_name or ""),
                    instruction=str(instruction or ""),
                    time_candidate=TimeCandidate(
                        datetime=str(dt or ""),
                        source="tool_agent_query",
                        score=0.0,
                        component=str(component or ""),
                        reason=str(reason or ""),
                        evidence={},
                    ),
                    component=str(component or ""),
                    reason=str(reason or ""),
                    enable_trace=bool(use_trace),
                ).to_dict()
                return {
                    "datetime": str(tev.get("datetime", "") or ""),
                    "metric_at_this_time": list(tev.get("metric_at_this_time") or [])[:8],
                    "trace_at_this_time": dict(tev.get("trace_at_this_time") or {}),
                    "metric_1min_before": list(tev.get("metric_1min_before") or [])[:5],
                    "metric_1min_after": list(tev.get("metric_1min_after") or [])[:5],
                    "evidence_for": list(tev.get("evidence_for") or []),
                    "evidence_against": list(tev.get("evidence_against") or []),
                }
            except Exception as exc:
                return {"error": f"query_time failed: {exc}"}

        def _resolve_item(item: Mapping[str, Any]) -> tuple[str, str, str]:
            """A compare side can be given as {"candidate_id": "c17"} or as an
            explicit {"component":..., "reason":..., "datetime":...} -- the
            latter lets the agent compare something it found via query_evidence
            that was never on the recall menu at all."""
            cid = str(item.get("candidate_id", "") or "").strip()
            if cid and cid in id_to_cand:
                c = id_to_cand[cid]
                return str(c.component), str(c.reason), str(getattr(c, "datetime", "") or "")
            return (
                str(item.get("component", "") or ""),
                str(item.get("reason", "") or ""),
                str(item.get("datetime", "") or ""),
            )

        def _tool_compare(item_a: Mapping[str, Any], item_b: Mapping[str, Any], include_topology: bool = False) -> Dict[str, Any]:
            # No heuristic verdict is computed here on purpose -- this just
            # fetches real evidence for both sides side by side. The LLM does
            # the actual comparison reasoning itself in its next response,
            # instead of trusting a system-computed score delta.
            comp_a, reason_a, dt_a = _resolve_item(item_a)
            comp_b, reason_b, dt_b = _resolve_item(item_b)
            ev_a = _tool_query_evidence(comp_a, reason_a, include_topology=include_topology)
            ev_b = _tool_query_evidence(comp_b, reason_b, include_topology=include_topology)
            out: Dict[str, Any] = {
                "a": {"component": comp_a, "reason": reason_a, "datetime": dt_a, "evidence": ev_a},
                "b": {"component": comp_b, "reason": reason_b, "datetime": dt_b, "evidence": ev_b},
            }
            if dt_a:
                out["a"]["time_evidence"] = _tool_query_time(comp_a, reason_a, dt_a, include_topology=include_topology)
            if dt_b:
                out["b"]["time_evidence"] = _tool_query_time(comp_b, reason_b, dt_b, include_topology=include_topology)
            return out

        all_candidate_pairs = [(str(c.component), str(c.reason)) for c in pool]

        def _tool_query_reason_evidence(component: str, reason: str) -> Dict[str, Any]:
            # Deviation/rank/contrast based reason test -- this is the real
            # way to test a reason hypothesis. It replaces the old "does a
            # reason keyword appear in this component's top-8 metrics" check
            # (still visible inside query_evidence's evidence_for/against
            # lines for backward compatibility, but that check is weak: on
            # real Bank data, "high memory usage" and "high disk I/O read
            # usage" both came back "matches" for the SAME component with
            # near-identical hit counts, giving no way to tell them apart).
            try:
                ev = build_reason_evidence(
                    dataset=str(dataset_name or ""),
                    instruction=str(instruction or ""),
                    component=str(component or ""),
                    reason=str(reason or ""),
                    all_candidates=all_candidate_pairs,
                ).to_dict()
                return ev
            except Exception as exc:
                return {"error": f"query_reason_evidence failed: {exc}"}

        def _tool_compare_reason_candidates(item_a: Mapping[str, Any], item_b: Mapping[str, Any]) -> Dict[str, Any]:
            comp_a, reason_a, _dt_a = _resolve_item(item_a)
            comp_b, reason_b, _dt_b = _resolve_item(item_b)
            ev_a = _tool_query_reason_evidence(comp_a, reason_a)
            ev_b = _tool_query_reason_evidence(comp_b, reason_b)
            score_a = float(ev_a.get("reason_deviation", 0.0) or 0.0) if "error" not in ev_a else 0.0
            score_b = float(ev_b.get("reason_deviation", 0.0) or 0.0) if "error" not in ev_b else 0.0
            return {
                "a": {"component": comp_a, "reason": reason_a, "evidence": ev_a},
                "b": {"component": comp_b, "reason": reason_b, "evidence": ev_b},
                "reason_deviation_gap_a_minus_b": round(score_a - score_b, 4),
            }

        system_prompt = """
You are a counterfactual RCA agent with real tool access to telemetry evidence.

The candidate list below is deliberately score-free: no rank, no confidence,
no pre-computed evidence. It exists only so you know which (component, reason,
datetime) tuples the system's own pipeline considers plausible -- it is a
recall aid, not a verdict, and it may be wrong, incomplete, or missing the
right answer entirely. Your job is verification, not "pick the one that was
listed first": form your own view of which candidates are worth checking by
comparing component/reason/time textually, then confirm or refute them with
real evidence, not with any ordering the system gave you.

Each turn, return exactly one JSON action, nothing else:
{"action":"list_candidates"}
   -- see the recall candidate list (candidate_id/component/reason/datetime only)
{"action":"query_evidence","component":"<name>","reason":"<label>","include_topology":false}
   -- fetch real metric evidence for this component (pass reason="" to just
      inspect the component). If you pass a reason label, the
      evidence_for/evidence_against lines about it are only a coarse
      keyword-presence check (does some anomalous metric name contain a
      word associated with this reason) -- it is NOT reliable for choosing
      between two plausible reasons on the same component, because several
      different reasons routinely share the same keyword-matching metrics
      during one real incident. Use query_reason_evidence (below) instead
      whenever you actually need to test or compare a reason hypothesis.
      You may query ANY component, not just ones on the candidate list.
      include_topology=true also fetches trace/dependency evidence but is
      much slower (can take tens of seconds on some datasets) -- only set it
      true when you specifically need upstream/downstream dependency
      evidence.
{"action":"query_time","component":"<name>","reason":"<label>","datetime":"YYYY-MM-DD HH:MM:SS","include_topology":false}
   -- fetch metric evidence at one specific timestamp for this component.
{"action":"query_reason_evidence","component":"<name>","reason":"<label>"}
   -- the real way to test a reason hypothesis. Returns how strongly and
      distinctly this component's actual metric deviation in the time window
      supports this specific reason: reason_deviation (deviation magnitude of
      this reason's metric group), reason_rank_on_component (1 = this is the
      single best-fitting reason on this component out of all known reasons
      for this dataset), gap_to_second_reason (how far ahead of the next-best
      reason on this component), cross_candidate_percentile (how this
      candidate's own deviation compares to every other candidate's own
      claimed reason), matched_metrics (which metrics actually drove the
      score), and competing_reasons (what else this component's data could
      instead support). `support` summarizes these as "strong"/"weak"/
      "not_supported" but the raw numbers are there for you to judge
      yourself -- a reason with rank 1 but a small gap, or a low
      cross_candidate_percentile, is not solid evidence even if support
      says "weak" rather than "not_supported".
{"action":"compare_reason_candidates","a":{"candidate_id":"c3"},"b":{"candidate_id":"c17"}}
   -- run query_reason_evidence for both sides in one turn (each side may
      also be given as {"component":"...","reason":"..."} for something not
      on the candidate list) and report the deviation gap between them
      directly, so you can weigh two competing (component, reason)
      hypotheses against each other without two separate turns.
{"action":"compare","a":{"candidate_id":"c3"},"b":{"candidate_id":"c17"},"include_topology":false}
   -- fetch real evidence for two candidates side by side in one turn, so you
      can weigh them against each other. Either side may instead be given as
      {"component":"...","reason":"...","datetime":"..."} to compare something
      not on the candidate list. This does not compute a verdict for you --
      you must judge which side the evidence actually supports yourself, in
      your next action.
{"action":"prune","ids":["c4","c9"],"reason":"short reason"}
   -- permanently remove these candidate_ids from the working set based on
      evidence you already queried (weak/missing/contradicted). This is a
      real removal: list_candidates will not show them again. Use this
      repeatedly to narrow a large candidate list down to a handful before
      you commit to a final answer. Do not prune a candidate you have not
      actually queried evidence for.
{"action":"decide", ...}
   -- commit to a final answer (schema below)

Suggested workflow (repeat steps 2-4 across several turns as needed, choosing
whichever dimension -- time, component, or reason -- looks most useful to
narrow on next; there is no fixed order). Whichever order you pick, do not
decide on a final reason for a candidate you have never run
query_reason_evidence on -- component fit and time fit do not substitute for
actually testing the reason dimension.
1. list_candidates, and skim it for component/reason/time plausibility only
   (no scores to lean on -- judge by content, not by list position).
2. query_evidence / query_time on whichever candidates (or components not
   even on the list) look most worth checking next for component/time fit.
   query_reason_evidence / compare_reason_candidates on whichever
   (component, reason) pairs look most worth checking next for reason fit --
   this is the only action that actually tests a reason hypothesis with a
   discriminative signal (rank/gap/percentile), not just keyword presence.
3. prune candidates whose evidence turned out weak, missing, or contradicted
   -- do this as you go, not only at the very end.
4. Once list_candidates shows only a few candidates left (or you have checked
   enough that you are confident none of the rest matter), do a final close
   compare among what remains.
5. decide once you have real evidence behind your answer, not before.

Do not invent metric values, timestamps, or evidence beyond what the tool
responses actually show you. You have a limited number of turns -- do not
query components you have no concrete reason to suspect.

Hard rule: you may only decide on a (component, reason) pair that is either
(a) identical to the deterministic top-1 you started with, or (b) a pair you
have actually run query_reason_evidence (or compare_reason_candidates) on
during this conversation. Deciding on some third pair you never checked --
for example because your first hypothesis only got "weak" support and you
drifted into comparing two other components instead of following through --
will be rejected and cost you a turn. If you got inconclusive support for
your first idea, either dig further on it, explicitly test the deterministic
top-1 itself, or test whatever new candidate you actually want to propose --
do not default to something you never verified.

When ready, return "decide" with:
{"action":"decide","component":"<name>","reason":"<label>","datetime":"YYYY-MM-DD HH:MM:SS or empty",
"confidence":0.0,"change_from_top1":false,"top1_falsified":false,
"falsification_reason":"why the deterministic top-1 is or is not right",
"rationale":"one sentence","decisive_evidence":"one concrete evidence item",
"rejected_candidates":[{"component":"...","reason":"...","reason_rejected":"short reason"}],
"field_decisions":{
  "component":{"adopt":false,"evidence_gain":0.0,"reason":"short evidence comparison"},
  "reason":{"adopt":false,"evidence_gain":0.0,"reason":"short evidence comparison, must cite a query_reason_evidence result (rank/gap/percentile), not a guess from raw metrics"},
  "datetime":{"adopt":false,"evidence_gain":0.0,"reason":"short evidence comparison"}
}}

Output JSON only, exactly one action per turn.
"""

        final_data: Dict[str, Any] = {}
        turns_used = 0
        for turn in range(1, max(1, int(max_turns)) + 1):
            turns_used = turn
            turns_remaining = max(0, int(max_turns) - turn)
            user_payload = {
                "dataset": str(dataset_name or ""),
                "task_focus": str(task_focus or "")[:240],
                "query": str(instruction or "")[:420],
                "turn": int(turn),
                "turns_remaining": int(turns_remaining),
                "candidate_count": int(len(menu)),
                "remaining_candidate_count": int(len(remaining_ids)),
                "queries_done": query_log,
            }
            user_prompt = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
            raw = self._chat(system_prompt, user_prompt)
            total_input_tokens += int(self.last_llm_call.get("approx_input_tokens", 0) or 0)
            total_elapsed += float(self.last_llm_call.get("elapsed_seconds", 0.0) or 0.0)
            data = safe_json_loads(
                raw,
                {
                    "action": "decide",
                    "component": str(top1.component),
                    "reason": str(top1.reason),
                    "datetime": str(getattr(top1, "datetime", "") or ""),
                    "confidence": float(top1.score),
                },
            )
            if not isinstance(data, Mapping):
                # safe_json_loads' fallback regex can match a bracketed list
                # in the raw text and return that instead of a dict (seen on
                # real Bank runs: "'list' object has no attribute 'get'"
                # crashed the whole turn loop). Treat a non-dict parse the
                # same as an invalid/malformed action rather than letting it
                # blow up the loop.
                data = {"action": "invalid"}
            action = str(data.get("action", "") or "").strip().lower()

            if action == "list_candidates":
                live_menu = [m for m in menu if m["candidate_id"] in remaining_ids]
                query_log.append({"turn": int(turn), "tool": "list_candidates", "result_preview": live_menu})
                actions_log.append({"turn": int(turn), "action": "list_candidates", "remaining_count": int(len(remaining_ids))})
                continue
            if action == "query_evidence":
                comp = str(data.get("component", "") or "")
                reason = str(data.get("reason", "") or "")
                include_topo = bool(data.get("include_topology", False))
                result = _tool_query_evidence(comp, reason, include_topology=include_topo)
                query_log.append({"turn": int(turn), "tool": "query_evidence", "component": comp, "reason": reason, "include_topology": include_topo, "result": result})
                actions_log.append({"turn": int(turn), "action": "query_evidence", "component": comp, "reason": reason})
                continue
            if action == "query_time":
                comp = str(data.get("component", "") or "")
                reason = str(data.get("reason", "") or "")
                dt = str(data.get("datetime", "") or "")
                include_topo = bool(data.get("include_topology", False))
                result = _tool_query_time(comp, reason, dt, include_topology=include_topo)
                query_log.append({"turn": int(turn), "tool": "query_time", "component": comp, "reason": reason, "datetime": dt, "include_topology": include_topo, "result": result})
                actions_log.append({"turn": int(turn), "action": "query_time", "component": comp, "reason": reason, "datetime": dt})
                continue
            if action == "query_reason_evidence":
                comp = str(data.get("component", "") or "")
                reason = str(data.get("reason", "") or "")
                reason_checks += 1
                result = _tool_query_reason_evidence(comp, reason)
                if "error" not in result:
                    reason_evidence_log[(comp, reason)] = result
                query_log.append({"turn": int(turn), "tool": "query_reason_evidence", "component": comp, "reason": reason, "result": result})
                actions_log.append({"turn": int(turn), "action": "query_reason_evidence", "component": comp, "reason": reason, "support": result.get("support")})
                continue
            if action == "compare_reason_candidates":
                item_a = data.get("a", {}) if isinstance(data.get("a", {}), Mapping) else {}
                item_b = data.get("b", {}) if isinstance(data.get("b", {}), Mapping) else {}
                reason_checks += 2
                compare_count += 1
                result = _tool_compare_reason_candidates(item_a, item_b)
                for side in ("a", "b"):
                    side_info = result.get(side) if isinstance(result.get(side), Mapping) else {}
                    side_ev = side_info.get("evidence") if isinstance(side_info.get("evidence"), Mapping) else {}
                    if side_ev and "error" not in side_ev:
                        reason_evidence_log[(str(side_info.get("component", "")), str(side_info.get("reason", "")))] = side_ev
                query_log.append({"turn": int(turn), "tool": "compare_reason_candidates", "a": dict(item_a), "b": dict(item_b), "result": result})
                actions_log.append({"turn": int(turn), "action": "compare_reason_candidates", "a": dict(item_a), "b": dict(item_b)})
                continue
            if action == "compare":
                item_a = data.get("a", {}) if isinstance(data.get("a", {}), Mapping) else {}
                item_b = data.get("b", {}) if isinstance(data.get("b", {}), Mapping) else {}
                include_topo = bool(data.get("include_topology", False))
                result = _tool_compare(item_a, item_b, include_topology=include_topo)
                compare_count += 1
                query_log.append({"turn": int(turn), "tool": "compare", "a": dict(item_a), "b": dict(item_b), "result": result})
                actions_log.append({"turn": int(turn), "action": "compare", "a": dict(item_a), "b": dict(item_b)})
                continue
            if action == "prune":
                ids_to_prune = [str(x) for x in list(data.get("ids") or []) if str(x).strip()]
                prune_reason = str(data.get("reason", "") or "").strip()
                actually_pruned = [i for i in ids_to_prune if i in remaining_ids]
                for i in actually_pruned:
                    remaining_ids.remove(i)
                pruned_log.append({"turn": int(turn), "ids": actually_pruned, "reason": prune_reason})
                query_log.append({"turn": int(turn), "tool": "prune", "ids": actually_pruned, "reason": prune_reason, "remaining_count": int(len(remaining_ids))})
                actions_log.append({"turn": int(turn), "action": "prune", "ids": actually_pruned, "remaining_count": int(len(remaining_ids))})
                continue
            if action == "decide":
                decide_component = str(data.get("component", "") or top1.component)
                decide_reason = str(data.get("reason", "") or top1.reason)
                is_top1_pair = (decide_component == str(top1.component) and decide_reason == str(top1.reason))
                is_tested_pair = (decide_component, decide_reason) in reason_evidence_log
                if (not is_top1_pair) and (not is_tested_pair) and turns_remaining > 0:
                    # General guardrail (not specific to any one dataset/query):
                    # deciding on a (component, reason) that was never actually
                    # run through query_reason_evidence -- and isn't just a
                    # no-op agreement with the deterministic top-1 -- silently
                    # reproduces whatever the deterministic pipeline already
                    # said without the agent ever verifying it, which defeats
                    # the entire point of tool-based verification. Reject the
                    # action and consume the turn instead of accepting it,
                    # while turns remain to actually go test it.
                    actions_log.append(
                        {
                            "turn": int(turn),
                            "action": "decide_rejected_untested_reason",
                            "component": decide_component,
                            "reason": decide_reason,
                        }
                    )
                    query_log.append(
                        {
                            "turn": int(turn),
                            "tool": "decide_rejected",
                            "component": decide_component,
                            "reason": decide_reason,
                            "note": "rejected: not the deterministic top-1 and never checked with query_reason_evidence",
                        }
                    )
                    continue
                if (not is_top1_pair) and (not is_tested_pair):
                    # Out of turns to go verify it. Do not silently trust an
                    # untested guess over something this run actually checked
                    # with real evidence -- prefer the best logged
                    # query_reason_evidence result if it cleared at least a
                    # "weak" bar, whatever dataset/query this is.
                    best_key, best_ev = _best_logged_reason_candidate()
                    if best_key is not None and str(best_ev.get("support", "")) in ("strong", "weak"):
                        data = dict(data)
                        data["component"], data["reason"] = best_key[0], best_key[1]
                        data["rationale"] = (
                            str(data.get("rationale", "") or "")
                            + f" [auto-substituted: original decide ({decide_component}, {decide_reason}) was never "
                            f"verified with query_reason_evidence; replaced with the best-tested result ({best_key[0]}, "
                            f"{best_key[1]}, support={best_ev.get('support')})."
                        ).strip()
                        actions_log.append(
                            {
                                "turn": int(turn),
                                "action": "decide_auto_substituted",
                                "component": best_key[0],
                                "reason": best_key[1],
                                "support": best_ev.get("support"),
                            }
                        )
                final_data = dict(data)
                actions_log.append({"turn": int(turn), "action": "decide"})
                break
            # Unknown / malformed action: treat as a wasted turn and continue.
            actions_log.append({"turn": int(turn), "action": action or "invalid"})
            continue

        if not final_data:
            best_key, best_ev = _best_logged_reason_candidate()
            if best_key is not None and str(best_ev.get("support", "")) in ("strong", "weak"):
                final_data = {
                    "component": best_key[0],
                    "reason": best_key[1],
                    "datetime": str(getattr(top1, "datetime", "") or ""),
                    "confidence": float(best_ev.get("reason_deviation", 0.0) or 0.0),
                    "rationale": (
                        "Tool agent exhausted max_turns without deciding; auto-selected the best "
                        f"query_reason_evidence result ({best_key[0]}, {best_key[1]}, support={best_ev.get('support')}) "
                        "instead of blindly falling back to the deterministic top-1."
                    ),
                    "top1_falsified": bool(best_key != (str(top1.component), str(top1.reason))),
                    "field_decisions": {},
                }
                actions_log.append({"turn": int(turns_used), "action": "fallback_max_turns_exhausted_used_best_tested"})
            else:
                final_data = {
                    "component": str(top1.component),
                    "reason": str(top1.reason),
                    "datetime": str(getattr(top1, "datetime", "") or ""),
                    "confidence": float(top1.score),
                    "rationale": "Tool agent exhausted max_turns without deciding; fell back to deterministic top-1.",
                    "top1_falsified": False,
                    "field_decisions": {},
                }
                actions_log.append({"turn": int(turns_used), "action": "fallback_max_turns_exhausted"})

        proposed_component = str(final_data.get("component", "") or top1.component)
        proposed_reason = str(final_data.get("reason", "") or top1.reason)
        proposed_datetime = str(final_data.get("datetime", "") or getattr(top1, "datetime", "") or "")

        # controller._field_level_adopt_candidate gates every proposed field
        # change on "auto evidence" it extracts by parsing this candidate's
        # own `evidence` string list (component_type=/supporting_metrics=/
        # rootness_features= tokens) -- if evidence=[] here, that parsing
        # sees nothing, prop_features all come back 0.0, and
        # component_evidence_gain/reason_evidence_gain end up negative no
        # matter how strong the real evidence the agent gathered was. That
        # silently vetoed every tool-agent adoption regardless of the actual
        # verification quality (confirmed on real Bank runs: llm_changed_count
        # =1, llm_veto_count=1, llm_adopted_count=0 even after the deviation-
        # based reason evidence correctly favored the true answer). Populate
        # evidence from whatever this run actually established for the
        # decided (component, reason): the original pool candidate's evidence
        # if it was already a recall candidate, plus the real matched metrics
        # from any query_reason_evidence call this turn loop made for it.
        selected_evidence: List[str] = []
        for c in pool:
            if str(c.component) == proposed_component and str(c.reason) == proposed_reason:
                selected_evidence = list(getattr(c, "evidence", []) or [])
                break
        for q in query_log:
            if (
                q.get("tool") == "query_reason_evidence"
                and str(q.get("component", "")) == proposed_component
                and str(q.get("reason", "")) == proposed_reason
            ):
                result = q.get("result") or {}
                matched = [
                    str(m.get("metric", ""))
                    for m in list(result.get("matched_metrics") or [])
                    if isinstance(m, Mapping) and str(m.get("metric", "")).strip()
                ]
                if matched:
                    selected_evidence.append("supporting_metrics=" + ",".join(matched[:6]))
                selected_evidence.append(f"reason_support={result.get('support', '')}")
                break

        selected = CandidateHypothesis(
            component=proposed_component,
            reason=proposed_reason,
            datetime=proposed_datetime,
            evidence=selected_evidence,
            score=normalize_score(final_data.get("confidence", top1.score)),
            source="tool_agent_decide",
        )

        changed_default = (
            proposed_component != str(top1.component)
            or proposed_reason != str(top1.reason)
            or proposed_datetime != str(getattr(top1, "datetime", "") or "")
        )
        details = {
            "selected_id": -1,
            "selected_candidate": selected.to_dict(),
            "confidence": normalize_score(final_data.get("confidence", selected.score)),
            "rationale": str(final_data.get("rationale", "")).strip(),
            "change_from_top1": bool(changed_default or final_data.get("change_from_top1", False)),
            # See the same fix in select_best_candidate_with_llm above: a
            # real field difference must win over the model's own
            # (possibly inconsistent) top1_falsified flag, not the other way
            # around. This is exactly what blocked row1 on a real Bank run --
            # the model proposed a different, well-evidenced component
            # (component_evidence_gain=1.09) but wrote top1_falsified=false,
            # and controller._allow_field's hard "if not top1_falsified:
            # return False" gate rejected the change outright before it ever
            # got to weighing the evidence.
            "top1_falsified": bool(changed_default or final_data.get("top1_falsified", False)),
            "falsification_reason": str(final_data.get("falsification_reason", "") or "").strip(),
            "top1_intervention_score": normalize_score(final_data.get("top1_intervention_score", 0.0)),
            "selected_intervention_score": normalize_score(final_data.get("selected_intervention_score", 0.0)),
            "decisive_evidence": str(final_data.get("decisive_evidence", "") or "").strip(),
            "field_decisions": dict(final_data.get("field_decisions", {}) or {}) if isinstance(final_data.get("field_decisions", {}), Mapping) else {},
            "rejected_candidates": ensure_list(final_data.get("rejected_candidates")) + pruned_log,
            "supporting_evidence": ensure_list(final_data.get("supporting_evidence")) or ensure_list(final_data.get("decisive_evidence")),
            "approx_input_tokens": int(total_input_tokens),
            "elapsed_seconds": float(total_elapsed),
            "prompt_candidate_count": int(len(pool)),
            "remaining_candidate_count": int(len(remaining_ids)),
            "agent_loop_turns_used": int(turns_used),
            "agent_loop_max_turns": int(max_turns),
            "agent_loop_inspected_count": int(len([q for q in query_log if q.get("tool") in ("query_evidence", "query_time", "query_reason_evidence")])),
            "agent_loop_comparisons_count": int(compare_count),
            "agent_loop_reason_checks_count": int(reason_checks),
            "agent_loop_actions": actions_log,
        }
        return {"candidate": selected, "details": details}

    def generate_multi_root_with_llm(
        self,
        *,
        instruction: str,
        evidence_groups: List[Dict[str, Any]],
        target_k: int,
        dataset_name: str = "",
        task_focus: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        k = int(target_k) if int(target_k) >= 1 else 1
        groups = list(evidence_groups or [])
        component_level = str(context.get("component_level", "") or "").strip().lower()

        system_prompt = f"""
You are an RCA agent for multi-failure queries.

The query explicitly contains {k} root causes / failures.
The system has selected a candidate root-cause set using coverage-first joint selection.
Each EvidenceGroup represents one possible root cause.

Component naming note:
- The expected component level for this dataset is: "{component_level or 'unknown'}".
- Ensure your output component matches that level (pod vs service vs node) when possible.

Your job:
- Verify the evidence in each EvidenceGroup.
- Output exactly {k} root causes.
- Avoid duplicate components unless the evidence clearly indicates repeated independent failures.
- For cascading cases, prefer upstream root causes unless the query asks for all failure sources.

Output strictly in JSON with keys "1"..."{k}". Each item must include:
- "root cause occurrence datetime"
- "root cause component"
- "root cause reason"
- "supporting evidence" (list of short strings)
"""

        user_prompt = f"""
Dataset: {dataset_name}
Task focus: {task_focus}
RCA query:
{instruction}

QueryMeta:
{json.dumps(context.get("query_meta", {}), ensure_ascii=False)}

Joint selection details:
{json.dumps(context.get("multi_root_selection_details", {}), ensure_ascii=False)}

EvidenceGroups (selected set, structured):
{json.dumps(groups, ensure_ascii=False, indent=2)}
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(raw, {})

        out: Dict[str, Any] = {}
        for i in range(1, k + 1):
            key = str(i)
            item = data.get(key) if isinstance(data, dict) else None
            if not isinstance(item, dict):
                item = {}
            comp = str(item.get("root cause component", "") or "").strip()
            reason = str(item.get("root cause reason", "") or "").strip()
            dt = str(item.get("root cause occurrence datetime", "") or "").strip()
            sup = ensure_list(item.get("supporting evidence", []))[:6]

            fallback_comp = ""
            fallback_reason = ""
            fallback_dt = ""
            if 0 <= (i - 1) < len(groups):
                g = groups[i - 1] if isinstance(groups[i - 1], dict) else {}
                fallback_comp = str(g.get("candidate_component", "") or "").strip()
                fallback_reason = str(g.get("candidate_reason", "") or "").strip()
                fallback_dt = str(g.get("candidate_time", "") or "").strip()

            out[key] = {
                "root cause component": comp or fallback_comp,
                "root cause reason": reason or fallback_reason,
            }
            if dt or fallback_dt:
                out[key]["root cause occurrence datetime"] = dt or fallback_dt
            if sup:
                out[key]["supporting evidence"] = sup

        details = {
            "target_k": int(k),
            "group_count": int(len(groups)),
            "raw_response": str(raw),
            "approx_input_tokens": int(self.last_llm_call.get("approx_input_tokens", 0) or 0),
            "elapsed_seconds": float(self.last_llm_call.get("elapsed_seconds", 0.0) or 0.0),
        }
        return {"prediction_json": json.dumps(out, ensure_ascii=False), "details": details}

    def select_time_with_llm(
        self,
        instruction: str,
        component: str,
        reason: str,
        time_candidates: List[Dict[str, Any]],
        dataset_name: str = "",
        task_focus: str = "",
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        context = context or {}
        comp = str(component or "")
        from rca.execution.evidence_builder import build_time_evidence
        from rca.adapter.utils.time_candidates import TimeCandidate
        enable_trace = bool(context.get("enable_trace_evidence", True))

        def _base_score(item: Dict[str, Any]) -> float:
            try:
                return float(item.get("score", 0.0) or 0.0)
            except Exception:
                return 0.0

        by_dt: Dict[str, Dict[str, Any]] = {}
        for rank, cand in enumerate(time_candidates or [], start=1):
            if not isinstance(cand, dict):
                continue
            dt = str(cand.get("datetime", "") or "").strip()
            if not dt:
                continue
            item = dict(cand)
            item["_original_rank"] = rank
            old = by_dt.get(dt)
            if old is None or _base_score(item) > _base_score(old):
                merged_sources = []
                merged_candidates = []
                if old is not None:
                    merged_sources.extend(list(old.get("_merged_sources", []) or [old.get("source", "")]))
                    merged_candidates.extend(list(old.get("_merged_candidates", []) or [old]))
                merged_sources.append(str(item.get("source", "")))
                merged_candidates.append(dict(item))
                item["_merged_sources"] = [s for s in merged_sources if s]
                item["_merged_candidates"] = merged_candidates
                by_dt[dt] = item
            else:
                old.setdefault("_merged_sources", [])
                src = str(item.get("source", ""))
                if src and src not in old["_merged_sources"]:
                    old["_merged_sources"].append(src)
                old.setdefault("_merged_candidates", [])
                old["_merged_candidates"].append(dict(item))

        deduped = list(by_dt.values())
        deterministic_order = sorted(
            deduped,
            key=lambda x: (_base_score(x), -int(x.get("_original_rank", 10**9))),
            reverse=True,
        )
        deterministic_best = dict(deterministic_order[0]) if deterministic_order else {}

        def _rerank_score(item: Dict[str, Any]) -> float:
            score = _base_score(item)
            details = {"base_score": score}
            c = str(item.get("component", "") or "")
            r = str(item.get("reason", "") or "")
            if comp and c:
                if c == comp:
                    score += 0.32
                    details["component_match_bonus"] = 0.32
                else:
                    score -= 0.10
                    details["component_mismatch_penalty"] = -0.10
            if reason and r:
                if r == reason:
                    score += 0.28
                    details["reason_match_bonus"] = 0.28
                else:
                    score -= 0.08
                    details["reason_mismatch_penalty"] = -0.08
            item["_rerank_score"] = float(score)
            item["_rerank_details"] = details
            return float(score)

        reranked = sorted(deduped, key=lambda x: (_rerank_score(x), _base_score(x), -int(x.get("_original_rank", 10**9))), reverse=True)
        top = reranked[: min(5, max(1, int(top_k)))]
        if not top:
            raise ValueError("No time candidates for LLM verification.")

        time_payload = []
        for i, cand in enumerate(top, start=1):
            evidence = dict(cand.get("evidence", {}) or {})
            if cand.get("_merged_sources"):
                evidence["merged_sources"] = list(cand.get("_merged_sources", []) or [])
            minute_evidence = {}
            try:
                dt = str(cand.get("datetime", "") or "")
                src = str(cand.get("source", "") or "")
                score = float(cand.get("score", 0.0) or 0.0)
                minute_evidence = build_time_evidence(
                    dataset=dataset_name,
                    instruction=instruction,
                    time_candidate=TimeCandidate(
                        datetime=dt,
                        source=src,
                        score=score,
                        component=str(cand.get("component", "") or ""),
                        reason=str(cand.get("reason", "") or ""),
                        evidence=dict(cand.get("evidence", {}) or {}),
                    ),
                    component=comp,
                    reason=str(reason or ""),
                    enable_trace=bool(enable_trace),
                ).to_dict()
            except Exception:
                minute_evidence = {}
            time_payload.append(
                {
                    "id": i,
                    "datetime": str(cand.get("datetime", "")),
                    "source": str(cand.get("source", "")),
                    "score": float(cand.get("score", 0.0) or 0.0),
                    "rerank_score": float(cand.get("_rerank_score", cand.get("score", 0.0)) or 0.0),
                    "rerank_details": dict(cand.get("_rerank_details", {}) or {}),
                    "component": str(cand.get("component", "") or ""),
                    "reason": str(cand.get("reason", "") or ""),
                    "evidence": evidence,
                    "minute_evidence": minute_evidence,
                }
            )

        first_times = context.get("component_first_times", {}) if isinstance(context.get("component_first_times"), dict) else {}
        peak_times = context.get("component_peak_times", {}) if isinstance(context.get("component_peak_times"), dict) else {}
        top_metrics = context.get("top_metrics", {}) if isinstance(context.get("top_metrics"), dict) else {}

        system_prompt = """
You are a concise RCA time verifier.

Choose exactly one root-cause occurrence datetime from the provided time candidates.
Do not invent a new datetime. Your task is bounded reranking, not time generation.

Decision rules:
1. Candidate #1 is deterministic BASIL top-1. Keep it unless another candidate has clearer component-specific onset evidence.
2. Prefer component-specific metric FIRST_BREACH aligned with the fixed reason.
3. Prefer stable root-cause onset over global trace symptom or query-window boundary.
4. If trace is already severely impacted at a candidate time, that time may be too late.
5. If all candidates have weak evidence, keep #1 with low confidence.

You may only rerank the provided candidates.
If the deterministic top-1 candidate has trace_first_breach/trace_onset/trace_stable_onset with score >= 1.0,
treat it as a protected high-confidence trace onset. Override it only when another candidate has stronger
and earlier component-metric onset evidence for the fixed component.

Output strictly in JSON:
{
  "selected_id": 1,
  "confidence": 0.0,
  "change_from_top1": false,
  "rationale": "short reason",
  "decisive_evidence": "specific metric/trace evidence"
}
"""

        user_prompt = f"""
Dataset: {dataset_name}
Task focus: {task_focus}
RCA query:
{instruction}

Fixed root-cause component: {comp}
Fixed root-cause reason: {reason}

Component time evidence:
- first_time: {first_times.get(comp, "")}
- peak_time: {peak_times.get(comp, "")}
- top_metrics: {json.dumps(list(top_metrics.get(comp, []) or [])[:8], ensure_ascii=False)}

Time candidates:
{json.dumps(time_payload, ensure_ascii=False, indent=2)}

Select the best root-cause occurrence datetime from the candidate list only.
Prefer component-specific first breach over global trace onset when the evidence is strong.
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(
            raw,
            {
                "selected_id": 1,
                "confidence": float(top[0].get("score", 0.0) or 0.0),
                "rationale": "Fallback to top time candidate because the LLM output is invalid.",
            },
        )
        try:
            selected_id = int(data.get("selected_id", 1))
        except Exception:
            selected_id = 1
        if selected_id < 1 or selected_id > len(top):
            selected_id = 1
        proposed = dict(top[selected_id - 1])
        rerank_best = dict(top[0])
        if not deterministic_best:
            deterministic_best = dict(rerank_best)

        def _score(item: Dict[str, Any]) -> float:
            try:
                return float(item.get("score", 0.0) or 0.0)
            except Exception:
                return 0.0

        def _source_parts(item: Dict[str, Any]) -> set:
            parts = set()
            source = str(item.get("source", "") or "")
            parts.update([s for s in source.split("+") if s])
            for src in item.get("_merged_sources", []) or []:
                parts.update([s for s in str(src).split("+") if s])
            evidence = item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
            for src in evidence.get("merged_sources", []) or []:
                parts.update([s for s in str(src).split("+") if s])
            return parts

        def _evidence_allows_low_score(item: Dict[str, Any]) -> bool:
            source_parts = _source_parts(item)
            c = str(item.get("component", "") or "")
            r = str(item.get("reason", "") or "")
            strong_sources = {
                "reason_metric_onset",
                "component_metric_first",
                "trace_stable_onset",
                "trace_onset",
            }
            if not (source_parts & strong_sources):
                return False
            if c and c == comp:
                return True
            if r and r == str(reason or ""):
                return True
            return bool(source_parts & {"reason_metric_onset", "component_metric_first"})

        def _is_protected_trace_onset(item: Dict[str, Any]) -> bool:
            return _score(item) >= 1.0 and bool(_source_parts(item) & {"trace_stable_onset", "trace_onset"})

        def _strong_override_for_protected(item: Dict[str, Any], drop: float) -> bool:
            source_parts = _source_parts(item)
            c = str(item.get("component", "") or "")
            r = str(item.get("reason", "") or "")
            if source_parts & {"trace_stable_onset", "trace_onset"}:
                return True
            if "reason_metric_onset" in source_parts and r and r == str(reason or ""):
                return True
            if "component_metric_first" in source_parts and c and c == comp and drop <= 0.05:
                return True
            return False

        max_drop = float(context.get("time_llm_max_score_drop", 0.20) or 0.20)
        score_drop = max(0.0, _score(deterministic_best) - _score(proposed))
        differs_from_deterministic = str(proposed.get("datetime", "")) != str(deterministic_best.get("datetime", ""))
        protected_deterministic_top1 = _is_protected_trace_onset(deterministic_best)
        accepted = True
        veto_reason = ""
        if differs_from_deterministic and score_drop > max_drop and not _evidence_allows_low_score(proposed):
            accepted = False
            veto_reason = (
                "LLM selected a lower-scored time candidate without explicit "
                "component/reason/source evidence; reverted to deterministic top-1."
            )
        if (
            accepted
            and differs_from_deterministic
            and protected_deterministic_top1
            and not _strong_override_for_protected(proposed, score_drop)
        ):
            accepted = False
            veto_reason = (
                "Deterministic top-1 is a protected high-confidence trace onset; "
                "LLM override lacked strong reason/trace evidence."
            )
        selected = proposed if accepted else deterministic_best
        try:
            def _dt_s(x: str) -> Optional[int]:
                try:
                    return int(datetime.fromisoformat(str(x)).timestamp())
                except Exception:
                    return None

            selected_dt_s = _dt_s(selected.get("datetime", ""))
            trace_items = []
            for it in deduped:
                if not isinstance(it, dict):
                    continue
                parts = _source_parts(it)
                if parts & {"trace_first_breach", "trace_onset", "trace_stable_onset"} and _score(it) >= 0.90:
                    t_s = _dt_s(it.get("datetime", ""))
                    if t_s is not None:
                        trace_items.append((t_s, dict(it)))
            if selected_dt_s is not None and trace_items:
                trace_items.sort(key=lambda x: x[0])
                earliest_trace_s, earliest_trace = trace_items[0]
                delta = int(selected_dt_s) - int(earliest_trace_s)
                if 120 <= delta <= 300 and not (_source_parts(selected) & {"trace_first_breach", "trace_onset", "trace_stable_onset"}):
                    selected = dict(earliest_trace)
                    accepted = True
                    veto_reason = "Prefer earlier high-confidence trace onset within 2-5 minutes."
        except Exception:
            pass
        accepted_selected_id = -1
        for i, item in enumerate(top, start=1):
            if str(item.get("datetime", "")) == str(selected.get("datetime", "")):
                accepted_selected_id = i
                break
        details = {
            "selected_id": int(selected_id),
            "accepted_selected_id": int(accepted_selected_id),
            "selected_time_candidate": selected,
            "deterministic_top1_candidate": deterministic_best,
            "reranked_time_candidates": time_payload,
            "datetime": str(selected.get("datetime", "")),
            "confidence": normalize_score(data.get("confidence", selected.get("score", 0.0))),
            "rationale": str(data.get("rationale", "")).strip(),
            "change_from_top1": bool(data.get("change_from_top1", selected_id != 1)),
            "decisive_evidence": str(data.get("decisive_evidence", "") or "").strip(),
            "rerank_policy": "llm_may_rerank_top_candidates_with_protected_trace_onset_veto",
            "score_drop_from_deterministic_top1": float(score_drop),
            "protected_deterministic_top1": bool(protected_deterministic_top1),
            "max_score_drop_without_evidence": float(max_drop),
            "accepted": bool(accepted),
            "veto_reason": veto_reason,
            "raw_response": raw,
            "approx_input_tokens": int(self.last_llm_call.get("approx_input_tokens", 0) or 0),
            "elapsed_seconds": float(self.last_llm_call.get("elapsed_seconds", 0.0) or 0.0),
        }
        return {"datetime": details["datetime"], "details": details}

    def propose_candidates(
        self,
        instruction: str,
        dataset_name: str = "",
        task_focus: str = "",
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateHypothesis]:
        context = context or {}
        allowed_components = context.get("allowed_components")
        allowed_reasons = context.get("allowed_reasons")
        component_scores = context.get("component_scores", {}) if isinstance(context.get("component_scores"), dict) else {}
        reason_scores = context.get("reason_scores", {}) if isinstance(context.get("reason_scores"), dict) else {}
        det_components: List[str] = []
        det_reasons: List[str] = []
        if isinstance(allowed_components, list):
            det_components = [str(x) for x in allowed_components if str(x or "").strip()]
        if isinstance(allowed_reasons, list):
            det_reasons = [str(x) for x in allowed_reasons if str(x or "").strip()]

        if not det_components and component_scores:
            try:
                det_components = [
                    str(k)
                    for k, _v in sorted(component_scores.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:12]
                    if str(k or "").strip()
                ]
            except Exception:
                det_components = []
        if not det_reasons and reason_scores:
            try:
                det_reasons = [
                    str(k)
                    for k, _v in sorted(reason_scores.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:12]
                    if str(k or "").strip()
                ]
            except Exception:
                det_reasons = []

        if det_components or det_reasons:
            max_c = max([abs(float(v)) for v in component_scores.values()] + [0.0])
            max_r = max([abs(float(v)) for v in reason_scores.values()] + [0.0])
            comp_list = det_components[:12] if det_components else ["unknown"]
            reason_list = det_reasons[:12] if det_reasons else ["unknown"]
            cand: List[CandidateHypothesis] = []
            for c in comp_list:
                cs = float(component_scores.get(str(c), 0.0) or 0.0)
                cn = cs / max(max_c, 1e-9) if max_c > 0 else 0.0
                for r in reason_list:
                    rs = float(reason_scores.get(str(r), 0.0) or 0.0)
                    rn = rs / max(max_r, 1e-9) if max_r > 0 else 0.0
                    if not det_components:
                        joint = rn
                    elif not det_reasons:
                        joint = cn
                    else:
                        joint = 0.60 * cn + 0.40 * rn
                    cand.append(
                        CandidateHypothesis(
                            component=str(c),
                            reason=str(r),
                            evidence=[],
                            score=normalize_score(joint),
                            source="deterministic_proposal",
                        )
                    )
            cand.sort(key=lambda x: x.score, reverse=True)
            return cand[: int(top_k)]

        system_prompt = f"""{self.agent_prompt}

You are an RCA executor specialized in counterfactual reasoning.
Your task is to read the RCA query and propose plausible root-cause hypotheses.

Output strictly in JSON:
{{
  "candidates": [
    {{
      "component": "...",
      "reason": "...",
      "evidence": ["...", "..."],
      "score": 0.0
    }}
  ]
}}

Requirements:
- score must be in [0,1]
- return at most {top_k} candidates
- prefer precise component and reason names
- evidence should be short, factual, and grounded in the query
"""

        user_prompt = f"""
Dataset: {dataset_name}
Task focus: {task_focus}
RCA query:
{instruction}
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(raw, {"candidates": []})

        candidates = []
        for item in data.get("candidates", [])[:top_k]:
            component = str(item.get("component", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not component or not reason:
                continue

            candidates.append(
                CandidateHypothesis(
                    component=component,
                    reason=reason,
                    evidence=ensure_list(item.get("evidence")),
                    score=normalize_score(item.get("score", 0.0)),
                    source="llm_proposal",
                )
            )

        return candidates

    def build_counterfactual_question(
        self,
        instruction: str,
        candidate: CandidateHypothesis,
        dataset_name: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        context = context or {}
        if bool(context.get("deterministic_only", False)):
            return {
                "counterfactual_question": (
                    f"If {candidate.component} had remained normal and {candidate.reason} had not occurred, "
                    "would the failure still happen?"
                ),
                "expected_effect": "The key anomaly signals related to this candidate should disappear or be significantly weakened.",
            }
        system_prompt = f"""{self.basic_prompt}

You are generating a counterfactual validation question for RCA.

Output strictly in JSON:
{{
  "counterfactual_question": "...",
  "expected_effect": "..."
}}

Requirements:
- The question must follow the form:
  'If component X had remained normal / if reason Y had not occurred, would the observed failure still happen?'
- expected_effect should describe what anomaly/path should disappear or weaken if the candidate is true
"""

        user_prompt = f"""
Dataset: {dataset_name}
RCA query:
{instruction}

Candidate:
- component: {candidate.component}
- reason: {candidate.reason}
- evidence: {candidate.evidence}
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(
            raw,
            {
                "counterfactual_question": (
                    f"If {candidate.component} had remained normal and "
                    f"{candidate.reason} had not occurred, would the failure still happen?"
                ),
                "expected_effect": (
                    "The downstream anomaly should disappear or be significantly weakened."
                ),
            },
        )
        return data

    def _compute_deterministic_evidence(
        self,
        candidate: CandidateHypothesis,
        context: Optional[Dict[str, Any]] = None,
    ) -> DeterministicEvidence:
        """
        根据外部上下文计算确定性证据。
        context 里可接:
        - metric_scores
        - peak_alignment
        - dependency_scores
        - trace_scores
        - historical_stats
        - graph_scores
        - matched_metrics
        - matched_traces
        - dependency_paths
        - historical_cases
        - graph_explanations
        """
        context = context or {}

        metric_scores = context.get("metric_scores", {})
        peak_alignment = context.get("peak_alignment", {})
        dependency_scores = context.get("dependency_scores", {})
        trace_scores = context.get("trace_scores", {})
        historical_stats = context.get("historical_stats", {})
        graph_scores = context.get("graph_scores", {})

        matched_metrics_map = context.get("matched_metrics", {})
        matched_traces_map = context.get("matched_traces", {})
        dependency_paths_map = context.get("dependency_paths", {})
        historical_cases_map = context.get("historical_cases", {})
        graph_explanations_map = context.get("graph_explanations", {})

        key_component_reason = f"{candidate.component}::{candidate.reason}"
        key_component = candidate.component

        metric_anomaly_score = normalize_score(
            metric_scores.get(key_component_reason, metric_scores.get(key_component, 0.0))
        )
        peak_timestamp_alignment = normalize_score(
            peak_alignment.get(key_component_reason, peak_alignment.get(key_component, 0.0))
        )

        dep_item = dependency_scores.get(key_component_reason, dependency_scores.get(key_component, {}))
        if not isinstance(dep_item, dict):
            dep_item = {}

        upstream_dependency_score = normalize_score(dep_item.get("upstream", 0.0))
        downstream_dependency_score = normalize_score(dep_item.get("downstream", 0.0))

        trace_duration_anomaly = normalize_score(
            trace_scores.get(key_component_reason, trace_scores.get(key_component, 0.0))
        )
        historical_frequency = normalize_score(
            historical_stats.get(key_component_reason, historical_stats.get(key_component, 0.0))
        )
        causal_graph_consistency = normalize_score(
            graph_scores.get(key_component_reason, graph_scores.get(key_component, 0.0))
        )

        matched_metrics = deduplicate_keep_order(
            ensure_list(matched_metrics_map.get(key_component_reason))
            + ensure_list(matched_metrics_map.get(key_component))
        )
        matched_traces = deduplicate_keep_order(
            ensure_list(matched_traces_map.get(key_component_reason))
            + ensure_list(matched_traces_map.get(key_component))
        )
        dependency_paths = deduplicate_keep_order(
            ensure_list(dependency_paths_map.get(key_component_reason))
            + ensure_list(dependency_paths_map.get(key_component))
        )
        historical_cases = deduplicate_keep_order(
            ensure_list(historical_cases_map.get(key_component_reason))
            + ensure_list(historical_cases_map.get(key_component))
        )
        graph_explanations = deduplicate_keep_order(
            ensure_list(graph_explanations_map.get(key_component_reason))
            + ensure_list(graph_explanations_map.get(key_component))
        )

        return DeterministicEvidence(
            metric_anomaly_score=metric_anomaly_score,
            peak_timestamp_alignment=peak_timestamp_alignment,
            upstream_dependency_score=upstream_dependency_score,
            downstream_dependency_score=downstream_dependency_score,
            trace_duration_anomaly=trace_duration_anomaly,
            historical_frequency=historical_frequency,
            causal_graph_consistency=causal_graph_consistency,
            matched_metrics=matched_metrics,
            matched_traces=matched_traces,
            dependency_paths=dependency_paths,
            historical_cases=historical_cases,
            graph_explanations=graph_explanations,
        )

    def _aggregate_deterministic_score(self, ev: DeterministicEvidence) -> float:
        """
        一个较稳妥的默认加权方式，后面可以按数据集再调。
        """
        score = (
            0.22 * ev.metric_anomaly_score
            + 0.18 * ev.peak_timestamp_alignment
            + 0.12 * ev.upstream_dependency_score
            + 0.08 * ev.downstream_dependency_score
            + 0.15 * ev.trace_duration_anomaly
            + 0.10 * ev.historical_frequency
            + 0.15 * ev.causal_graph_consistency
        )
        return normalize_score(score)

    def validate_candidate(
        self,
        instruction: str,
        candidate: CandidateHypothesis,
        counterfactual_question: str,
        expected_effect: str,
        dataset_name: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> CounterfactualCheck:
        deterministic_evidence = self._compute_deterministic_evidence(
            candidate=candidate,
            context=context,
        )
        deterministic_score = self._aggregate_deterministic_score(deterministic_evidence)
        context = context or {}
        if bool(context.get("deterministic_only", False)):
            conf = normalize_score(candidate.score, deterministic_score)
            judgement = "support" if conf >= 0.75 else ("weaken" if conf <= 0.25 else "uncertain")
            return CounterfactualCheck(
                candidate=candidate,
                counterfactual_question=counterfactual_question,
                expected_effect=expected_effect,
                judgement=judgement,
                confidence=conf,
                rationale="Deterministic-only mode: decision derived from deterministic evidence.",
                supporting_evidence=[],
                deterministic_evidence=deterministic_evidence,
            )

        system_prompt = f"""{self.agent_prompt}

You are performing counterfactual validation for RCA.

You must combine:
1. counterfactual reasoning
2. deterministic evidence

Output strictly in JSON:
{{
  "judgement": "support",
  "confidence": 0.0,
  "rationale": "...",
  "supporting_evidence": ["...", "..."]
}}

Rules:
- judgement must be one of: support, weaken, uncertain
- support: if removing this cause would likely eliminate or significantly reduce the observed anomaly
- weaken: if the anomaly likely remains even after removing this cause
- uncertain: insufficient evidence
- confidence must be in [0,1]
- deterministic evidence should be treated as strong grounding signals
"""

        user_prompt = f"""
Dataset: {dataset_name}
RCA query:
{instruction}

Candidate:
- component: {candidate.component}
- reason: {candidate.reason}
- evidence: {candidate.evidence}
- prior score: {candidate.score}

Counterfactual question:
{counterfactual_question}

Expected effect:
{expected_effect}

Deterministic evidence:
- metric_anomaly_score: {deterministic_evidence.metric_anomaly_score}
- peak_timestamp_alignment: {deterministic_evidence.peak_timestamp_alignment}
- upstream_dependency_score: {deterministic_evidence.upstream_dependency_score}
- downstream_dependency_score: {deterministic_evidence.downstream_dependency_score}
- trace_duration_anomaly: {deterministic_evidence.trace_duration_anomaly}
- historical_frequency: {deterministic_evidence.historical_frequency}
- causal_graph_consistency: {deterministic_evidence.causal_graph_consistency}
- aggregated_deterministic_score: {deterministic_score}

Matched details:
- matched_metrics: {deterministic_evidence.matched_metrics}
- matched_traces: {deterministic_evidence.matched_traces}
- dependency_paths: {deterministic_evidence.dependency_paths}
- historical_cases: {deterministic_evidence.historical_cases}
- graph_explanations: {deterministic_evidence.graph_explanations}

Please decide whether this candidate is supported, weakened, or uncertain under the counterfactual assumption.
"""

        raw = self._chat(system_prompt, user_prompt)
        data = safe_json_loads(
            raw,
            {
                "judgement": "uncertain",
                "confidence": deterministic_score,
                "rationale": "Fallback to deterministic evidence because the LLM output is invalid.",
                "supporting_evidence": [],
            },
        )

        llm_conf = normalize_score(data.get("confidence", deterministic_score), deterministic_score)
        final_conf = normalize_score(0.6 * llm_conf + 0.4 * deterministic_score)

        judgement = str(data.get("judgement", "uncertain")).strip().lower()
        if judgement not in {"support", "weaken", "uncertain"}:
            judgement = "uncertain"

        return CounterfactualCheck(
            candidate=candidate,
            counterfactual_question=counterfactual_question,
            expected_effect=expected_effect,
            judgement=judgement,
            confidence=final_conf,
            rationale=str(data.get("rationale", "")).strip(),
            supporting_evidence=ensure_list(data.get("supporting_evidence")),
            deterministic_evidence=deterministic_evidence,
        )

    def refine_candidates(
        self,
        checks: List[CounterfactualCheck],
        top_k: int = 3,
    ) -> List[CandidateHypothesis]:
        scored = []
        for chk in checks:
            base = chk.candidate.score
            cf = chk.confidence

            det = 0.0
            if chk.deterministic_evidence is not None:
                det = self._aggregate_deterministic_score(chk.deterministic_evidence)

            if chk.judgement == "support":
                final_score = 0.25 * base + 0.45 * cf + 0.30 * det
            elif chk.judgement == "weaken":
                final_score = 0.35 * base + 0.10 * cf + 0.15 * det
            else:
                final_score = 0.30 * base + 0.30 * cf + 0.25 * det

            merged_evidence = list(chk.candidate.evidence) + list(chk.supporting_evidence)
            if chk.deterministic_evidence is not None:
                merged_evidence += chk.deterministic_evidence.matched_metrics
                merged_evidence += chk.deterministic_evidence.matched_traces
                merged_evidence += chk.deterministic_evidence.dependency_paths
                merged_evidence += chk.deterministic_evidence.graph_explanations

            scored.append(
                CandidateHypothesis(
                    component=chk.candidate.component,
                    reason=chk.candidate.reason,
                    evidence=deduplicate_keep_order(merged_evidence),
                    score=normalize_score(final_score),
                    source="counterfactual_refine",
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def finalize(
        self,
        checks: List[CounterfactualCheck],
        refined_candidates: List[CandidateHypothesis],
    ) -> Dict[str, Any]:
        """
        最终决策规则：
        1. 优先选择 judgement = support 且 confidence 最高的
        2. 若没有 support，则从 refined_candidates 里选分数最高的
        """
        support_checks = [c for c in checks if c.judgement == "support"]
        if support_checks:
            support_checks.sort(key=lambda x: x.confidence, reverse=True)
            best = support_checks[0]
            return {
                "component": best.candidate.component,
                "reason": best.candidate.reason,
                "confidence": best.confidence,
                "reasoning": best.rationale,
            }

        if refined_candidates:
            best = refined_candidates[0]
            return {
                "component": best.component,
                "reason": best.reason,
                "confidence": best.score,
                "reasoning": "Selected as the highest-ranked hypothesis after counterfactual refinement.",
            }

        return {
            "component": "unknown",
            "reason": "unknown",
            "confidence": 0.0,
            "reasoning": "No valid candidate was produced.",
        }
