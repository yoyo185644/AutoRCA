import json
import time
from typing import Any, Dict, List, Mapping, Optional

from rca.adapter.registry import get_dataset_adapter

from .executor import CounterfactualExecutor
from .schemas import CandidateHypothesis


class CounterfactualController:
    def __init__(self, *, executor: CounterfactualExecutor, logger=None) -> None:
        self.executor = executor
        self.logger = logger

    @staticmethod
    def _field_level_adopt_candidate(
        *,
        deterministic: CandidateHypothesis,
        proposed: CandidateHypothesis,
        llm_details: Mapping[str, Any],
        context: Mapping[str, Any],
        adapter_ok: bool,
        min_conf: float,
        max_drop: float,
        det_score: float,
        prop_score: float,
        need_comp: bool,
        need_reason: bool,
        need_time: bool,
    ) -> tuple[CandidateHypothesis, Dict[str, Any]]:
        conf = float(llm_details.get("confidence", 0.0) or 0.0)
        score_drop = max(0.0, float(det_score) - float(prop_score))
        evidence_conflict = dict((context or {}).get("uncertainty_gate", {}).get("evidence_conflict", {}) or {})
        challenger_profile = dict(evidence_conflict.get("challenger_profile", {}) or {})
        conflict_dims = {str(x) for x in list(evidence_conflict.get("conflict_dimensions") or []) if str(x).strip()}
        challenger_sufficient = bool(challenger_profile.get("sufficient", False))
        challenger_strength = int(challenger_profile.get("evidence_strength", 0) or 0)

        def _features(cand: CandidateHypothesis) -> Dict[str, float | str]:
            evidence = [str(x) for x in list(getattr(cand, "evidence", []) or []) if str(x or "").strip()]
            blob = " ".join(evidence).lower()
            ctype = ""
            rootness: Dict[str, float] = {}
            metric_count = 0
            incoming = 0.0
            for item in evidence:
                if item.startswith("component_type="):
                    ctype = item.split("=", 1)[1].strip().lower()
                if item.startswith("topology_context="):
                    try:
                        topo = json.loads(item.split("=", 1)[1])
                        if isinstance(topo, dict):
                            if bool(topo.get("shared_infra", False)):
                                incoming = max(incoming, 0.45)
                            fanout = float(topo.get("fanout", 0) or 0)
                            if fanout >= 2:
                                incoming = max(incoming, min(0.75, 0.35 + 0.08 * fanout))
                            if not ctype and str(topo.get("role", "") or "").strip():
                                ctype = str(topo.get("role", "") or "").strip().lower()
                    except Exception:
                        pass
                if item.startswith("supporting_metrics="):
                    vals = [x for x in item.split("=", 1)[1].split(",") if x.strip()]
                    metric_count = max(metric_count, len(vals))
                if item.startswith("rootness_features="):
                    for part in item.split("=", 1)[1].split(","):
                        if "=" not in part:
                            continue
                        k, v = part.split("=", 1)
                        try:
                            rootness[k.strip()] = float(v)
                        except Exception:
                            pass
            owned = max(rootness.get("instance_owned_score", 0.0), rootness.get("ownership", 0.0))
            sibling = rootness.get("sibling_contrast", 0.0)
            causal = max(rootness.get("causal_source_score", 0.0), rootness.get("outgoing_propagation", 0.0))
            incoming = max(float(incoming), float(rootness.get("incoming_symptomness", 0.0) or 0.0))
            local = min(1.0, owned + 0.35 * sibling + 0.20 * min(1.0, metric_count / 6.0))
            symptom = min(1.0, incoming + (0.30 if ctype == "node" else 0.0))
            metric_reason = 0.0
            reason = str(getattr(cand, "reason", "") or "").lower()
            if reason:
                if "cpu" in reason and "cpu" in blob:
                    metric_reason = 1.0
                elif "memory" in reason and any(tok in blob for tok in ("mem", "memory", "rss")):
                    metric_reason = 1.0
                elif any(tok in reason for tok in ("network", "packet", "latency", "loss")) and any(tok in blob for tok in ("network", "packet", "tcp", "retrans", "latency")):
                    metric_reason = 1.0
                elif any(tok in reason for tok in ("read", "write", "disk", "i/o", "io")) and any(tok in blob for tok in ("read", "write", "disk", "io", "fs_")):
                    metric_reason = 1.0
            return {
                "component_type": ctype,
                "local": float(local),
                "causal": float(min(1.0, causal)),
                "symptom": float(symptom),
                "metric_reason": float(metric_reason),
            }

        def _auto_intervention_score(cand: CandidateHypothesis) -> float:
            f = _features(cand)
            has_time = 1.0 if str(getattr(cand, "datetime", "") or "").strip() else 0.0
            return float(
                0.42 * float(f["local"])
                + 0.22 * float(f["metric_reason"])
                + 0.18 * float(f["causal"])
                + 0.08 * has_time
                - 0.25 * float(f["symptom"])
            )

        det_auto_intervention = _auto_intervention_score(deterministic)
        prop_auto_intervention = _auto_intervention_score(proposed)
        llm_top1_intervention = float(llm_details.get("top1_intervention_score", 0.0) or 0.0)
        llm_prop_intervention = float(llm_details.get("selected_intervention_score", 0.0) or 0.0)
        if llm_top1_intervention <= 0.0 and llm_prop_intervention <= 0.0:
            top1_intervention = det_auto_intervention
            prop_intervention = prop_auto_intervention
            intervention_source = "auto_evidence"
        else:
            top1_intervention = 0.55 * llm_top1_intervention + 0.45 * det_auto_intervention
            prop_intervention = 0.55 * llm_prop_intervention + 0.45 * prop_auto_intervention
            intervention_source = "llm_plus_auto"
        intervention_gain = float(prop_intervention - top1_intervention)
        det_features = _features(deterministic)
        prop_features = _features(proposed)
        det_type = str(det_features.get("component_type", "") or "")
        prop_type = str(prop_features.get("component_type", "") or "")
        symptom_over_local = (
            prop_type == "node"
            and det_type in {"container", "pod", "service", "app", "db", "cache", "web"}
            and float(prop_features["local"]) < float(det_features["local"]) + 0.25
            and float(prop_features["metric_reason"]) <= float(det_features["metric_reason"])
        )
        aggressive_intervention = (
            bool(adapter_ok)
            and conf >= 0.45
            and intervention_gain >= 0.05
            and float(prop_features["local"]) >= max(0.12, float(det_features["local"]) - 0.10)
            and float(prop_features["symptom"]) <= float(det_features["symptom"]) + 0.20
            and not symptom_over_local
        )

        def _same(field: str) -> bool:
            return str(getattr(deterministic, field, "") or "") == str(getattr(proposed, field, "") or "")

        def _proposed_matches_challenger(field: str) -> bool:
            if field == "datetime":
                return bool(str(getattr(proposed, "datetime", "") or "")) and str(getattr(proposed, "datetime", "") or "") == str(challenger_profile.get("time_cluster", "") or "")
            key = "reason" if field == "reason" else "component"
            return bool(str(challenger_profile.get(key, "") or "")) and str(getattr(proposed, field, "") or "") == str(challenger_profile.get(key, "") or "")

        def _allow_field(field: str) -> tuple[bool, str]:
            if _same(field):
                return False, "unchanged"
            if field == "component" and symptom_over_local:
                return False, "shared_infra_symptom_over_local_candidate"
            if aggressive_intervention:
                return True, "positive_intervention_gain"
            if conf < max(0.65, min_conf - 0.15):
                return False, "llm_confidence_too_low"
            if score_drop > max_drop and not challenger_sufficient:
                return False, "score_drop_without_sufficient_challenger"
            if field in {"component", "reason"} and not adapter_ok:
                return False, "adapter_rejected_component_reason_pair"
            if field == "component":
                if ({"component_family", "metric_vs_trace", "trace_vs_metric"} & conflict_dims) or _proposed_matches_challenger("component"):
                    return bool(challenger_sufficient and challenger_strength >= 2), "component_conflict_evidence"
                return False, "no_component_field_conflict"
            if field == "reason":
                if ("reason" in conflict_dims) or _proposed_matches_challenger("reason"):
                    return bool(challenger_sufficient and challenger_strength >= 2), "reason_conflict_evidence"
                return False, "no_reason_field_conflict"
            if field == "datetime":
                if {"time_cluster", "metric_vs_trace", "trace_vs_metric"} & conflict_dims:
                    return bool(challenger_sufficient and challenger_strength >= 2), "time_conflict_evidence"
                if score_drop <= max_drop and conf >= min_conf:
                    return True, "high_confidence_low_score_drop_time_change"
                return False, "no_time_field_conflict"
            return False, "unknown_field"

        component = str(getattr(deterministic, "component", "") or "")
        reason = str(getattr(deterministic, "reason", "") or "")
        datetime_s = str(getattr(deterministic, "datetime", "") or "")
        adopted_fields: List[str] = []
        rejected_fields: Dict[str, str] = {}

        if need_comp:
            ok, why = _allow_field("component")
            if ok:
                component = str(getattr(proposed, "component", "") or component)
                adopted_fields.append("component")
            elif why != "unchanged":
                rejected_fields["component"] = why
        if need_reason:
            ok, why = _allow_field("reason")
            if ok:
                reason = str(getattr(proposed, "reason", "") or reason)
                adopted_fields.append("reason")
            elif why != "unchanged":
                rejected_fields["reason"] = why
        if need_time:
            ok, why = _allow_field("datetime")
            if ok:
                datetime_s = str(getattr(proposed, "datetime", "") or datetime_s)
                adopted_fields.append("datetime")
            elif why != "unchanged":
                rejected_fields["datetime"] = why

        merged = CandidateHypothesis(
            component=component,
            reason=reason,
            datetime=datetime_s,
            evidence=list(getattr(proposed, "evidence", []) or getattr(deterministic, "evidence", []) or []),
            score=float(max(float(det_score), float(prop_score) if adopted_fields else float(det_score))),
            source="llm_field_adoption" if adopted_fields else str(getattr(deterministic, "source", "") or "deterministic"),
        )
        details = {
            "adopted_fields": adopted_fields,
            "rejected_fields": rejected_fields,
            "llm_confidence": float(conf),
            "score_drop": float(score_drop),
            "challenger_sufficient": bool(challenger_sufficient),
            "challenger_evidence_strength": int(challenger_strength),
            "conflict_dimensions": sorted(conflict_dims),
            "intervention_source": str(intervention_source),
            "top1_intervention_score": float(top1_intervention),
            "proposed_intervention_score": float(prop_intervention),
            "intervention_gain": float(intervention_gain),
            "deterministic_evidence_features": dict(det_features),
            "proposed_evidence_features": dict(prop_features),
            "aggressive_intervention": bool(aggressive_intervention),
            "symptom_over_local_blocked": bool(symptom_over_local),
        }
        return merged, details

    @staticmethod
    def _uncertainty_gate(*, candidates: List[CandidateHypothesis], context: Mapping[str, Any]) -> Dict[str, Any]:
        top1 = candidates[0] if candidates else None
        top2 = candidates[1] if len(candidates) > 1 else None
        top1_score = float(getattr(top1, "score", 0.0) or 0.0) if top1 is not None else 0.0
        top2_score = float(getattr(top2, "score", 0.0) or 0.0) if top2 is not None else 0.0
        gap = max(0.0, float(top1_score) - float(top2_score))

        gap_th = float(context.get("evidence_score_gap_threshold", 0.15) or 0.15)
        floor = float(context.get("evidence_top1_score_floor", 0.30) or 0.30)
        floor_single = float(context.get("evidence_top1_score_floor_single", 0.50) or 0.50)
        skip_gap_th = float(context.get("gate_skip_gap_threshold", 0.30) or 0.30)

        reasons: List[str] = []
        need_llm = False
        if not candidates:
            need_llm = True
            reasons.append("no_candidates")
        elif len(candidates) == 1:
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

        if need_llm and len(candidates) >= 2 and top1_score >= floor_single and gap >= skip_gap_th:
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
        }

    def _candidates_from_context(self, *, instruction: str, dataset_name: str, task_focus: str, context: Mapping[str, Any]) -> List[CandidateHypothesis]:
        items = list((context or {}).get("candidate_tuples") or [])
        out: List[CandidateHypothesis] = []
        for it in items:
            if not isinstance(it, Mapping):
                continue
            comp = str(it.get("component", "") or "").strip()
            reason = str(it.get("reason", "") or "").strip()
            dt = str(it.get("datetime", "") or it.get("root cause occurrence datetime", "") or "").strip()
            if not comp and not reason and not dt:
                continue
            evidence = [str(x) for x in list(it.get("evidence") or []) if str(x or "").strip()]
            metrics = [str(x) for x in list(it.get("supporting_metrics") or []) if str(x or "").strip()]
            if metrics:
                evidence.append("supporting_metrics=" + ",".join(metrics[:6]))
            rootness = dict(it.get("rootness_features", {}) or {}) if isinstance(it.get("rootness_features", {}), Mapping) else {}
            feature_bits = []
            for key in ("instance_owned_score", "sibling_contrast", "causal_source_score", "outgoing_propagation", "incoming_symptomness"):
                if key in rootness:
                    feature_bits.append(f"{key}={rootness.get(key)}")
            if feature_bits:
                evidence.append("rootness_features=" + ",".join(feature_bits[:5]))
            ctype = str(it.get("component_type", "") or "").strip()
            if ctype:
                evidence.append("component_type=" + ctype)
            topo = dict(it.get("topology_context", {}) or {}) if isinstance(it.get("topology_context", {}), Mapping) else {}
            if topo:
                compact_topo = {
                    "role": str(topo.get("component_role", "") or ""),
                    "shared_infra": bool(topo.get("is_shared_infra", False)),
                    "fanout": int(topo.get("trace_fanout", topo.get("fanout", 0)) or 0),
                    "fanin": int(topo.get("trace_fanin", topo.get("fanin", 0)) or 0),
                    "upstream": [str(x) for x in list(topo.get("trace_upstream", topo.get("upstream", [])) or [])[:3]],
                    "downstream": [str(x) for x in list(topo.get("trace_downstream", topo.get("downstream", [])) or [])[:3]],
                    "note": str(topo.get("rootness_note", "") or "")[:180],
                }
                evidence.append("topology_context=" + json.dumps(compact_topo, ensure_ascii=False, separators=(",", ":")))
            out.append(
                CandidateHypothesis(
                    component=comp,
                    reason=reason,
                    datetime=dt,
                    evidence=evidence,
                    score=float(it.get("score", 0.0) or 0.0),
                    source=str(it.get("source", "basil_tuple") or "basil_tuple"),
                )
            )
        # Preserve the index/ranker order. The candidate pool may already have
        # applied dataset-specific rootness, coverage, and tuple selection.
        # Re-sorting here by raw score can promote high-score symptoms above the
        # deterministic answer, making the LLM defend the wrong top-1.
        challenger_keys = [x for x in list((context or {}).get("llm_challenger_candidate_keys") or []) if isinstance(x, Mapping)]
        if out and challenger_keys:
            top1 = out[0]

            def matches_key(cand: CandidateHypothesis, key: Mapping[str, Any]) -> bool:
                kc = str(key.get("component", "") or "").strip()
                kr = str(key.get("reason", "") or "").strip()
                kd = str(key.get("datetime", "") or "").strip()
                if kc and str(getattr(cand, "component", "") or "") != kc:
                    return False
                if kr and str(getattr(cand, "reason", "") or "") != kr:
                    return False
                if kd and str(getattr(cand, "datetime", "") or "") != kd:
                    return False
                return True

            selected: List[CandidateHypothesis] = [top1]
            for key in challenger_keys:
                for cand in out[1:]:
                    if cand in selected:
                        continue
                    if matches_key(cand, key):
                        selected.append(cand)
                        break

            def sig(cand: CandidateHypothesis) -> tuple[str, str, str]:
                return (
                    str(getattr(cand, "component", "") or ""),
                    str(getattr(cand, "reason", "") or ""),
                    str(getattr(cand, "datetime", "") or ""),
                )

            seen = {sig(x) for x in selected}
            for cand in out:
                if sig(cand) in seen:
                    continue
                selected.append(cand)
                seen.add(sig(cand))
            out = selected
        if out:
            return out
        top_k = int((context or {}).get("llm_verify_top_k", 3) or 3)
        return self.executor.propose_candidates(
            instruction=str(instruction),
            dataset_name=str(dataset_name or ""),
            task_focus=str(task_focus or ""),
            top_k=max(1, top_k),
            context=dict(context or {}),
        )

    def run(
        self,
        *,
        instruction: str,
        dataset_name: str = "",
        max_step: int = 25,
        max_turn: int = 5,
        task_focus: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        context = dict(context or {})
        required_k = max(1, int(context.get("required_root_count", 1) or 1))
        need_comp = bool(context.get("need_comp", True))
        need_reason = bool(context.get("need_reason", True))
        enable_uncertainty_llm = bool(context.get("enable_uncertainty_llm", False))
        deterministic_only = bool(context.get("deterministic_only", False))
        component_level = str(context.get("component_level", "") or "")

        adapter = get_dataset_adapter(str(dataset_name or ""))
        candidates = self._candidates_from_context(
            instruction=str(instruction),
            dataset_name=str(dataset_name or ""),
            task_focus=str(task_focus or ""),
            context=context,
        )
        if adapter is not None and not bool(context.get("candidate_tuples_ranked", False)):
            fn = getattr(adapter, "rerank_candidate_hypotheses", None)
            if callable(fn):
                try:
                    reranked = fn(
                        candidates=list(candidates),
                        context=dict(context or {}),
                        component_level=str(component_level or ""),
                        need_comp=bool(need_comp),
                        need_reason=bool(need_reason),
                        need_time=bool(context.get("need_time", False)),
                    )
                    if isinstance(reranked, list) and reranked:
                        candidates = list(reranked)
                except Exception:
                    pass

        candidates = list(candidates or [])
        if not candidates:
            out = {"1": {"root cause component": "unknown", "root cause reason": "unknown"}}
            return json.dumps(out, ensure_ascii=False), [], []

        trajectory: List[Dict[str, Any]] = []
        prompt: List[str] = []

        if required_k > 1:
            picked: List[CandidateHypothesis] = []
            used: set[tuple[str, str]] = set()
            for cand in candidates:
                key = (str(getattr(cand, "component", "") or ""), str(getattr(cand, "reason", "") or ""))
                if key in used:
                    continue
                used.add(key)
                picked.append(cand)
                if len(picked) >= required_k:
                    break
            if len(picked) < required_k:
                picked.extend([c for c in candidates if c not in picked][: required_k - len(picked)])
            out: Dict[str, Any] = {}
            for i, cand in enumerate(picked[:required_k], start=1):
                out[str(i)] = {
                    "root cause component": str(getattr(cand, "component", "") or ""),
                    "root cause reason": str(getattr(cand, "reason", "") or ""),
                }
                if bool(context.get("need_time", False)):
                    out[str(i)]["root cause occurrence datetime"] = str(getattr(cand, "datetime", "") or "")
            return json.dumps(out, ensure_ascii=False), trajectory, prompt

        top = candidates[0]
        gate = {}
        if (not deterministic_only) and enable_uncertainty_llm:
            gate = self._uncertainty_gate(candidates=candidates, context=context)
            context["uncertainty_gate"] = dict(gate)
            trajectory.append({"step": 1, "action": "uncertainty_gate", "elapsed_seconds": 0.0, "output": dict(gate)})

            if bool(gate.get("need_llm")):
                step_t0 = time.perf_counter()
                try:
                    selected = self.executor.select_best_candidate_with_llm(
                        instruction=str(instruction),
                        candidates=list(candidates),
                        dataset_name=str(dataset_name or ""),
                        task_focus=str(task_focus or ""),
                        context=dict(context or {}),
                        top_k=int(context.get("llm_verify_top_k", 3) or 3),
                    )
                    elapsed = float(time.perf_counter() - step_t0)
                    llm_details = dict(selected.get("details") or {})
                    trajectory.append(
                        {
                            "step": 2,
                            "action": "llm_verify_candidate",
                            "elapsed_seconds": float(elapsed),
                            "output": llm_details,
                        }
                    )
                    proposed = selected.get("candidate")
                    if isinstance(proposed, CandidateHypothesis):
                        min_conf = float(context.get("llm_change_min_confidence", 0.85) or 0.85)
                        max_drop = float(context.get("llm_max_score_drop_vs_det_top1", 0.05) or 0.05)
                        conf = float(llm_details.get("confidence", 0.0) or 0.0)
                        det_score = float(getattr(top, "score", 0.0) or 0.0)
                        prop_score = float(getattr(proposed, "score", 0.0) or 0.0)
                        changed = (
                            str(getattr(proposed, "component", "") or "") != str(getattr(top, "component", "") or "")
                            or str(getattr(proposed, "reason", "") or "") != str(getattr(top, "reason", "") or "")
                            or str(getattr(proposed, "datetime", "") or "") != str(getattr(top, "datetime", "") or "")
                        )
                        adapter_ok = True
                        adapter_veto_reason = ""
                        if changed and adapter is not None:
                            try:
                                adapter_ok, adapter_veto_reason = adapter.validate_llm_candidate_adoption(
                                    proposed_component=str(getattr(proposed, "component", "") or ""),
                                    proposed_reason=str(getattr(proposed, "reason", "") or ""),
                                    deterministic_component=str(getattr(top, "component", "") or ""),
                                    deterministic_reason=str(getattr(top, "reason", "") or ""),
                                    context=dict(context or {}),
                                    need_comp=bool(need_comp),
                                    need_reason=bool(need_reason),
                                )
                            except Exception:
                                adapter_ok = True
                                adapter_veto_reason = ""
                        if changed:
                            merged, field_details = self._field_level_adopt_candidate(
                                deterministic=top,
                                proposed=proposed,
                                llm_details=llm_details,
                                context=dict(context or {}),
                                adapter_ok=bool(adapter_ok),
                                min_conf=float(min_conf),
                                max_drop=float(max_drop),
                                det_score=float(det_score),
                                prop_score=float(prop_score),
                                need_comp=bool(need_comp),
                                need_reason=bool(need_reason),
                                need_time=bool(need_time),
                            )
                            trajectory.append(
                                {
                                    "step": 2,
                                    "action": "llm_field_adoption",
                                    "elapsed_seconds": 0.0,
                                    "output": {
                                        **field_details,
                                        "min_confidence": float(min_conf),
                                        "max_score_drop_vs_det_top1": float(max_drop),
                                        "det_score": float(det_score),
                                        "llm_selected_score": float(prop_score),
                                        "adapter_consistency_ok": bool(adapter_ok),
                                        "adapter_veto_reason": str(adapter_veto_reason or ""),
                                    },
                                }
                            )
                            if list(field_details.get("adopted_fields") or []):
                                top = merged
                            else:
                                trajectory.append(
                                    {
                                        "step": 2,
                                        "action": "llm_veto",
                                        "elapsed_seconds": 0.0,
                                        "output": {
                                            **field_details,
                                            "reason": "no_fields_adopted",
                                            "adapter_consistency_ok": bool(adapter_ok),
                                            "adapter_veto_reason": str(adapter_veto_reason or ""),
                                        },
                                    }
                                )
                except Exception as exc:
                    elapsed = float(time.perf_counter() - step_t0)
                    trajectory.append(
                        {
                            "step": 2,
                            "action": "llm_verify_candidate_failed",
                            "elapsed_seconds": float(elapsed),
                            "output": {"error": str(exc)},
                        }
                    )

        comp_out = str(getattr(top, "component", "") or "")
        reason_out = str(getattr(top, "reason", "") or "")
        dt_out = str(getattr(top, "datetime", "") or "")
        if adapter is not None:
            if need_comp:
                try:
                    comp_out = str(adapter.canonicalize_component(component=str(comp_out), component_level=str(component_level or "")) or comp_out)
                except Exception:
                    pass
            if need_reason:
                try:
                    reason_out = str(adapter.parse_root_cause_label(str(reason_out)) or reason_out)
                except Exception:
                    pass
        out = {
            "1": {
                "root cause component": comp_out if need_comp else "",
                "root cause reason": reason_out if need_reason else "",
            }
        }
        if bool(context.get("need_time", False)):
            out["1"]["root cause occurrence datetime"] = dt_out
        return json.dumps(out, ensure_ascii=False), trajectory, prompt


class CounterfactualRCAAgent:
    def __init__(self, agent_prompt, basic_prompt) -> None:
        self.ap = agent_prompt
        self.bp = basic_prompt

    def run(
        self,
        instruction,
        logger,
        dataset_name: str = "",
        max_step: int = 25,
        max_turn: int = 5,
        task_focus: str = "",
        context: dict = None,
    ):
        logger.info(f"Objective: {instruction}")

        executor = CounterfactualExecutor(
            agent_prompt=self.ap,
            basic_prompt=self.bp,
            logger=logger,
        )
        controller = CounterfactualController(
            executor=executor,
            logger=logger,
        )

        prediction, trajectory, prompt = controller.run(
            instruction=instruction,
            dataset_name=dataset_name,
            max_step=max_step,
            max_turn=max_turn,
            task_focus=task_focus,
            context=context,
        )

        logger.info(f"Result: {prediction}")
        return prediction, trajectory, prompt
