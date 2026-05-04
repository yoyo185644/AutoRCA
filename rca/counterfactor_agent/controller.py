from typing import Any, Dict, List, Tuple
import time

from .executor import CounterfactualExecutor
from .schemas import ControllerState
from .utils import format_prediction_json
from rca.agent.evidence_builder import build_candidate_evidence, build_time_evidence, compute_uncertainty
from rca.time_candidates import TimeCandidate, select_time_candidate


def is_uncertain(candidates: List[Any], context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    scores = [float(getattr(c, "score", 0.0) or 0.0) for c in candidates or []]
    scores = sorted(scores, reverse=True)
    gap_threshold = float(context.get("llm_uncertain_gap_threshold", 1.0) or 1.0)
    confidence_floor = float(context.get("llm_confidence_floor", 1.0) or 1.0)
    time_confidence_floor = float(context.get("llm_time_confidence_floor", 1.0) or 1.0)
    need_time = bool(context.get("need_time", False))
    need_comp = bool(context.get("need_comp", False))
    need_reason = bool(context.get("need_reason", False))
    difficulty = str(context.get("difficulty", "") or "").lower()
    force_midhard = bool(context.get("llm_force_midhard", False))

    reasons: List[str] = []
    top1_score = float(scores[0]) if scores else 0.0
    top2_score = float(scores[1]) if len(scores) >= 2 else 0.0
    gap = float(top1_score - top2_score) if len(scores) >= 2 else 1.0
    if len(scores) >= 2 and gap < gap_threshold:
        reasons.append("top1_top2_gap_below_threshold")
    if top1_score < confidence_floor:
        reasons.append("top1_score_below_confidence_floor")
    required_fields = int(need_time) + int(need_comp) + int(need_reason)
    if force_midhard and len(scores) >= 2 and required_fields >= 2:
        if difficulty == "hard":
            reasons.append("hard_multifield_requires_llm_verification")
        elif difficulty == "middle":
            reasons.append("middle_multifield_requires_llm_verification")

    time_candidates = list(context.get("time_candidates") or [])
    time_confidence = 1.0
    if need_time:
        if time_candidates:
            vals = []
            for cand in time_candidates:
                if isinstance(cand, dict):
                    vals.append(float(cand.get("score", 0.0) or 0.0))
                else:
                    vals.append(float(getattr(cand, "score", 0.0) or 0.0))
            time_confidence = max(vals + [0.0])
        else:
            time_confidence = 0.0
        if time_confidence < time_confidence_floor:
            reasons.append("time_confidence_below_floor")

    details = {
        "is_uncertain": bool(reasons),
        "reasons": reasons,
        "top1_score": float(top1_score),
        "top2_score": float(top2_score),
        "gap": float(gap),
        "gap_threshold": float(gap_threshold),
        "confidence_floor": float(confidence_floor),
        "need_time": bool(need_time),
        "need_comp": bool(need_comp),
        "need_reason": bool(need_reason),
        "difficulty": difficulty,
        "required_fields": int(required_fields),
        "force_midhard": bool(force_midhard),
        "time_confidence": float(time_confidence),
        "time_confidence_floor": float(time_confidence_floor),
    }
    return bool(reasons), details


def is_time_uncertain(
    time_candidates: List[Dict[str, Any]],
    component: str,
    reason: str,
    context: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    def _score(item: Dict[str, Any]) -> float:
        try:
            return float(item.get("score", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _sources(item: Dict[str, Any]) -> set:
        parts = set()
        source = str(item.get("source", "") or "")
        parts.update([s for s in source.split("+") if s])
        evidence = item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
        for src in evidence.get("merged_sources", []) or []:
            parts.update([s for s in str(src).split("+") if s])
        return parts

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
        if old is None or _score(item) > _score(old):
            by_dt[dt] = item

    ordered = sorted(
        by_dt.values(),
        key=lambda x: (_score(x), -int(x.get("_original_rank", 10**9))),
        reverse=True,
    )
    if not ordered:
        return False, {
            "is_uncertain": False,
            "reasons": ["no_time_candidates"],
            "top1_score": 0.0,
            "top2_score": 0.0,
            "gap": 1.0,
        }

    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) >= 2 else {}
    top1_score = _score(top1)
    top2_score = _score(top2) if top2 else 0.0
    gap = top1_score - top2_score if len(ordered) >= 2 else 1.0
    gap_threshold = float(context.get("time_llm_gap_threshold", 0.08) or 0.08)
    confidence_floor = float(context.get("time_llm_confidence_floor", 0.85) or 0.85)
    force = bool(context.get("time_llm_force", False))
    top1_sources = _sources(top1)
    protected_trace_onset = top1_score >= 1.0 and bool(top1_sources & {"trace_stable_onset", "trace_onset"})

    comp = str(component or "")
    reason_s = str(reason or "")
    has_close_component_or_reason_candidate = False
    for item in ordered[1: min(len(ordered), 5)]:
        item_sources = _sources(item)
        c = str(item.get("component", "") or "")
        r = str(item.get("reason", "") or "")
        close = top1_score - _score(item) <= float(context.get("time_llm_competing_drop", 0.12) or 0.12)
        if close and ((comp and c == comp) or (reason_s and r == reason_s) or "reason_metric_onset" in item_sources):
            has_close_component_or_reason_candidate = True
            break

    reasons: List[str] = []
    if force:
        reasons.append("time_llm_force")
    if top1_score < confidence_floor:
        reasons.append("time_top1_score_below_floor")
    if gap < gap_threshold:
        reasons.append("time_top1_top2_gap_below_threshold")
    if not protected_trace_onset and has_close_component_or_reason_candidate:
        reasons.append("close_component_or_reason_time_candidate")

    if protected_trace_onset and not force:
        reasons = [
            r for r in reasons
            if r in {"time_top1_score_below_floor", "time_top1_top2_gap_below_threshold"}
        ]

    details = {
        "is_uncertain": bool(reasons),
        "reasons": reasons,
        "top1_datetime": str(top1.get("datetime", "")),
        "top2_datetime": str(top2.get("datetime", "")) if top2 else "",
        "top1_score": float(top1_score),
        "top2_score": float(top2_score),
        "gap": float(gap),
        "gap_threshold": float(gap_threshold),
        "confidence_floor": float(confidence_floor),
        "protected_trace_onset": bool(protected_trace_onset),
        "has_close_component_or_reason_candidate": bool(has_close_component_or_reason_candidate),
        "force": bool(force),
    }
    return bool(reasons), details


class CounterfactualController:
    """
    Controller 负责：
    - 流程编排
    - step / turn 控制
    - 停止条件
    - 轨迹记录
    """

    def __init__(self, executor: CounterfactualExecutor, logger=None):
        self.executor = executor
        self.logger = logger

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def run(
        self,
        instruction: str,
        dataset_name: str = "",
        max_step: int = 25,
        max_turn: int = 5,
        task_focus: str = "",
        context: Dict[str, Any] = None,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        state = ControllerState(
            instruction=instruction,
            dataset_name=dataset_name,
            task_focus=task_focus,
            max_step=max_step,
            max_turn=max_turn,
        )
        trajectory: List[Dict[str, Any]] = []
        prompt_trace: List[Dict[str, Any]] = []

        self._log(f"[Controller] Start counterfactual RCA. dataset={dataset_name}, task_focus={task_focus}")
        context = context or {}
        need_time = bool(context.get("need_time", False))
        need_comp = bool(context.get("need_comp", True))
        need_reason = bool(context.get("need_reason", True))

        def pick_time(component: str, reason: str = "") -> Tuple[str, Dict[str, Any]]:
            if not need_time:
                return "", {}
            time_candidates = context.get("time_candidates") or []
            if time_candidates:
                selected, details = select_time_candidate(
                    time_candidates,
                    component=component,
                    reason=reason,
                    fallback=str(context.get("time_choice", "") or ""),
                )
                if str(selected or "").strip():
                    return str(selected), dict(details or {})
            if need_comp:
                ft = (context.get("component_first_times") or {}).get(component)
                if str(ft or "").strip():
                    return str(ft), {"source": "component_first_times_fallback", "component": component}
                pt = (context.get("component_peak_times") or {}).get(component)
                if str(pt or "").strip():
                    return str(pt), {"source": "component_peak_times_fallback", "component": component}
                est = str(context.get("estimated_failure_time", "") or "").strip()
                if est:
                    return est, {"source": "estimated_failure_time_fallback"}
            t = str(context.get("time_choice", "") or "").strip()
            if t:
                return t, {"source": "time_choice_fallback"}
            return "", {}

        def pick_time_with_optional_llm(component: str, reason: str = "") -> Tuple[str, Dict[str, Any]]:
            if not need_time:
                return "", {}
            if bool(context.get("enable_time_llm", False)) and not bool(context.get("deterministic_only", False)):
                time_candidates = list(context.get("time_candidates") or [])
                if time_candidates:
                    should_call_time_llm, time_gate = is_time_uncertain(time_candidates, component, reason, context)
                    trajectory.append(
                        {
                            "step": state.step,
                            "action": "time_uncertainty_gate",
                            "output": time_gate,
                            "enable_time_llm": True,
                            "should_call_time_llm": bool(should_call_time_llm),
                        }
                    )
                    self._log(
                        "[Controller] Time LLM gate: "
                        f"{'CALL_LLM' if should_call_time_llm else 'SKIP_LLM'} "
                        f"reasons={time_gate.get('reasons', [])} "
                        f"top1={float(time_gate.get('top1_score', 0.0)):.4f} "
                        f"top2={float(time_gate.get('top2_score', 0.0)):.4f} "
                        f"gap={float(time_gate.get('gap', 0.0)):.4f} "
                        f"protected_trace_onset={bool(time_gate.get('protected_trace_onset', False))}"
                    )
                    if not should_call_time_llm:
                        return pick_time(component, reason)
                    step_t0 = time.perf_counter()
                    try:
                        selected = self.executor.select_time_with_llm(
                            instruction=instruction,
                            component=component,
                            reason=reason,
                            time_candidates=time_candidates,
                            dataset_name=dataset_name,
                            task_focus=task_focus,
                            context=context,
                            top_k=int(context.get("time_llm_top_k", 3) or 3),
                        )
                        elapsed = time.perf_counter() - step_t0
                        details = dict(selected.get("details") or {})
                        details["source"] = "llm_time_verifier"
                        details["elapsed_total_seconds"] = float(elapsed)
                        self._log(
                            "[Controller] Time LLM verifier done: "
                            f"elapsed_seconds={elapsed:.6f}, "
                            f"llm_call_seconds={float(details.get('elapsed_seconds', 0.0)):.6f}, "
                            f"approx_input_tokens={int(details.get('approx_input_tokens', 0) or 0)}, "
                            f"selected_time={selected.get('datetime', '')}, "
                            f"accepted={bool(details.get('accepted', True))}, "
                            f"accepted_selected_id={int(details.get('accepted_selected_id', -1) or -1)}, "
                            f"veto_reason={details.get('veto_reason', '')}"
                        )
                        if str(selected.get("datetime", "") or "").strip():
                            return str(selected.get("datetime")), details
                    except Exception as e:
                        failed_call = dict(getattr(self.executor, "last_llm_call", {}) or {})
                        self._log(f"[Controller] Time LLM verifier failed, fallback to deterministic time: {e}")
                        return_dt, return_details = pick_time(component, reason)
                        return_details = dict(return_details or {})
                        return_details["llm_time_error"] = str(e)
                        return_details["llm_time_attempted"] = True
                        return_details["approx_input_tokens"] = int(failed_call.get("approx_input_tokens", 0) or 0)
                        return_details["elapsed_seconds"] = float(failed_call.get("elapsed_seconds", 0.0) or 0.0)
                        return return_dt, return_details
            return pick_time(component, reason)

        # Step 1: 候选生成
        state.step += 1
        state.turn += 1
        self._log("[Controller] Step 1: proposing candidates...")
        step_t0 = time.perf_counter()
        candidates = self.executor.propose_candidates(
            instruction=instruction,
            dataset_name=dataset_name,
            task_focus=task_focus,
            top_k=5,
            context=context,
        )
        elapsed = time.perf_counter() - step_t0
        state.candidates = candidates

        trajectory.append(
            {
                "step": state.step,
                "action": "propose_candidates",
                "elapsed_seconds": float(elapsed),
                "output": [c.to_dict() for c in candidates],
            }
        )

        if not candidates:
            time_dt, time_selection = pick_time_with_optional_llm("unknown", "unknown")
            prediction = format_prediction_json(
                component="unknown",
                reason="unknown" if need_reason else "",
                confidence=0.0,
                reasoning="No candidate hypothesis could be generated.",
                occurrence_datetime=time_dt,
            )
            state.done = True
            trajectory.append(
                {
                    "step": state.step,
                    "action": "terminate",
                    "reason": "no_candidates",
                    "time_selection": time_selection,
                }
            )
            return prediction, trajectory, str(prompt_trace)

        force_deterministic = bool(context.get("deterministic_only", False))
        enable_uncertainty_llm = bool(context.get("enable_uncertainty_llm", False))
        gate_details: Dict[str, Any] = {"is_uncertain": False, "reasons": ["uncertainty_llm_disabled"]}
        should_call_llm = False
        if enable_uncertainty_llm and not force_deterministic:
            uncertain, gate_details = is_uncertain(candidates, context)
            try:
                all_components = [
                    str(c.component)
                    for c in list(candidates or [])
                    if str(getattr(c, "component", "") or "").strip()
                ]
                evidence_candidates = []
                for i, cand in enumerate(list(candidates or [])[: max(2, int(context.get("llm_verify_top_k", 3) or 3))]):
                    next_score = float(candidates[i + 1].score) if i + 1 < len(candidates) else 0.0
                    evidence_candidates.append(
                        build_candidate_evidence(
                            dataset=dataset_name,
                            instruction=instruction,
                            component=str(cand.component),
                            reason=str(cand.reason),
                            score=float(cand.score),
                            score_gap_vs_next=max(0.0, float(cand.score) - next_score),
                            all_candidate_components=all_components,
                            topology_max_rows=context.get("basil_max_trace_rows"),
                        )
                    )
                evidence_times = []
                for item in list(context.get("time_candidates") or [])[:5]:
                    if not isinstance(item, dict):
                        continue
                    evidence_times.append(
                        build_time_evidence(
                            dataset=dataset_name,
                            instruction=instruction,
                            time_candidate=TimeCandidate(
                                datetime=str(item.get("datetime", "")),
                                source=str(item.get("source", "")),
                                score=float(item.get("score", 0.0) or 0.0),
                                component=str(item.get("component", "") or ""),
                                reason=str(item.get("reason", "") or ""),
                                evidence=dict(item.get("evidence", {}) or {}),
                            ),
                            component=str(candidates[0].component if candidates else ""),
                            reason=str(candidates[0].reason if candidates else ""),
                        )
                    )
                task_fields = []
                if need_time:
                    task_fields.append("time")
                if need_comp:
                    task_fields.append("component")
                if need_reason:
                    task_fields.append("reason")
                evidence_gate = compute_uncertainty(
                    candidates=evidence_candidates,
                    time_candidates=evidence_times,
                    task_fields=task_fields,
                    score_gap_threshold=float(context.get("evidence_score_gap_threshold", 0.15) or 0.15),
                    top1_score_floor=float(context.get("evidence_top1_score_floor", 0.30) or 0.30),
                    time_gap_threshold=float(context.get("evidence_time_gap_threshold", 0.20) or 0.20),
                    conflict_seconds=int(context.get("evidence_time_conflict_seconds", 300) or 300),
                )
                evidence_gate["candidate_evidence_count"] = len(evidence_candidates)
                evidence_gate["time_evidence_count"] = len(evidence_times)
                gate_details = {
                    **gate_details,
                    "evidence_gate": evidence_gate,
                    "reasons": list(dict.fromkeys(list(gate_details.get("reasons", []) or []) + list(evidence_gate.get("reasons", []) or []))),
                    "focus": list(evidence_gate.get("focus", []) or []),
                }
                uncertain = bool(uncertain or evidence_gate.get("need_llm", False))
            except Exception as e:
                gate_details["evidence_gate_error"] = str(e)
            should_call_llm = bool(uncertain)
            self._log(
                "[Controller] LLM gate: "
                f"{'CALL_LLM' if should_call_llm else 'SKIP_LLM'} "
                f"reasons={gate_details.get('reasons', [])} "
                f"top1={gate_details.get('top1_score', 0.0):.4f} "
                f"top2={gate_details.get('top2_score', 0.0):.4f} "
                f"gap={gate_details.get('gap', 0.0):.4f} "
                f"time_conf={gate_details.get('time_confidence', 1.0):.4f}"
            )
            trajectory.append(
                {
                    "step": state.step,
                    "action": "uncertainty_gate",
                    "output": gate_details,
                    "enable_uncertainty_llm": bool(enable_uncertainty_llm),
                    "force_deterministic": bool(force_deterministic),
                    "should_call_llm": bool(should_call_llm),
                }
            )
        context["uncertainty_gate"] = gate_details

        if should_call_llm:
            step_t0 = time.perf_counter()
            try:
                selected = self.executor.select_best_candidate_with_llm(
                    instruction=instruction,
                    candidates=candidates,
                    dataset_name=dataset_name,
                    task_focus=task_focus,
                    context=context,
                    top_k=int(context.get("llm_verify_top_k", 3) or 3),
                )
                top = selected["candidate"]
                llm_details = dict(selected.get("details") or {})
                time_dt, time_selection = pick_time_with_optional_llm(top.component, top.reason)
                prediction = format_prediction_json(
                    component=top.component if need_comp else "",
                    reason=top.reason if need_reason else "",
                    confidence=float(llm_details.get("confidence", top.score) or top.score),
                    reasoning=str(llm_details.get("rationale", "") or "LLM selected from top-k candidates."),
                    occurrence_datetime=time_dt,
                )
                elapsed = time.perf_counter() - step_t0
                self._log(
                    "[Controller] LLM verifier done: "
                    f"elapsed_seconds={elapsed:.6f}, "
                    f"llm_call_seconds={float(llm_details.get('elapsed_seconds', 0.0)):.6f}, "
                    f"approx_input_tokens={int(llm_details.get('approx_input_tokens', 0) or 0)}, "
                    f"selected={top.component}::{top.reason}"
                )
                state.done = True
                trajectory.append(
                    {
                        "step": state.step,
                        "action": "llm_verify_topk",
                        "elapsed_seconds": float(elapsed),
                        "output": llm_details,
                        "time_selection": time_selection,
                    }
                )
                return prediction, trajectory, str(prompt_trace)
            except Exception as e:
                failed_call = dict(getattr(self.executor, "last_llm_call", {}) or {})
                self._log(f"[Controller] LLM verifier failed, fallback to deterministic top-1: {e}")
                trajectory.append(
                    {
                        "step": state.step,
                        "action": "llm_verify_topk_failed",
                        "error": str(e),
                        "output": failed_call,
                    }
                )
                should_call_llm = False

        if force_deterministic or not should_call_llm:
            step_t0 = time.perf_counter()
            top = candidates[0]
            time_dt, time_selection = pick_time_with_optional_llm(top.component, top.reason)
            prediction = format_prediction_json(
                component=top.component if need_comp else "",
                reason=top.reason if need_reason else "",
                confidence=top.score,
                reasoning="Deterministic gate selected the top-ranked hypothesis.",
                occurrence_datetime=time_dt,
            )
            elapsed = time.perf_counter() - step_t0
            state.done = True
            trajectory.append(
                {
                    "step": state.step,
                    "action": "deterministic_finalize",
                    "elapsed_seconds": float(elapsed),
                    "output": top.to_dict(),
                    "time_selection": time_selection,
                }
            )
            return prediction, trajectory, str(prompt_trace)

        # Step 2: 对每个候选进行反事实验证
        for idx, cand in enumerate(candidates):
            if state.step >= state.max_step or state.turn >= state.max_turn:
                self._log("[Controller] Reached max_step or max_turn, stop further checks.")
                break

            state.step += 1
            state.turn += 1
            self._log(f"[Controller] Step {state.step}: build CF question for candidate #{idx + 1}")

            step_t0 = time.perf_counter()
            cf_info = self.executor.build_counterfactual_question(
                instruction=instruction,
                candidate=cand,
                dataset_name=dataset_name,
                context=context,
            )
            elapsed = time.perf_counter() - step_t0
            counterfactual_question = cf_info.get("counterfactual_question", "")
            expected_effect = cf_info.get("expected_effect", "")

            trajectory.append(
                {
                    "step": state.step,
                    "action": "build_counterfactual_question",
                    "elapsed_seconds": float(elapsed),
                    "candidate": cand.to_dict(),
                    "output": cf_info,
                }
            )

            if state.step >= state.max_step:
                break

            state.step += 1
            self._log(f"[Controller] Step {state.step}: validate candidate #{idx + 1}")

            step_t0 = time.perf_counter()
            check = self.executor.validate_candidate(
                instruction=instruction,
                candidate=cand,
                counterfactual_question=counterfactual_question,
                expected_effect=expected_effect,
                dataset_name=dataset_name,
                context=context,
            )
            elapsed = time.perf_counter() - step_t0
            state.checks.append(check)

            trajectory.append(
                {
                    "step": state.step,
                    "action": "validate_candidate",
                    "elapsed_seconds": float(elapsed),
                    "candidate": cand.to_dict(),
                    "output": check.to_dict(),
                }
            )

            # Early Stop
            if check.judgement == "support" and check.confidence >= 0.85:
                self._log("[Controller] Early stop: high-confidence counterfactual support.")
                trajectory.append(
                    {
                        "step": state.step,
                        "action": "early_stop",
                        "reason": "high_confidence_support",
                        "candidate": cand.to_dict(),
                        "confidence": check.confidence,
                    }
                )
                break

        # Step 3: 基于反事实结果重新排序
        if state.step < state.max_step:
            state.step += 1
            self._log(f"[Controller] Step {state.step}: refine candidates...")
            step_t0 = time.perf_counter()
            refined = self.executor.refine_candidates(state.checks, top_k=3)
            elapsed = time.perf_counter() - step_t0
            trajectory.append(
                {
                    "step": state.step,
                    "action": "refine_candidates",
                    "elapsed_seconds": float(elapsed),
                    "output": [c.to_dict() for c in refined],
                }
            )
        else:
            refined = state.candidates[:3]

        # Step 4: 最终决策
        if state.step < state.max_step:
            state.step += 1
        self._log(f"[Controller] Step {state.step}: finalize result...")

        step_t0 = time.perf_counter()
        final = self.executor.finalize(state.checks, refined)
        time_dt, time_selection = pick_time_with_optional_llm(final["component"], final.get("reason", ""))
        elapsed = time.perf_counter() - step_t0

        prediction = format_prediction_json(
            component=final["component"],
            reason=final["reason"] if need_reason else "",
            confidence=final["confidence"],
            reasoning=final["reasoning"],
            occurrence_datetime=time_dt,
        )

        trajectory.append(
            {
                "step": state.step,
                "action": "finalize",
                "elapsed_seconds": float(elapsed),
                "output": final,
                "time_selection": time_selection,
            }
        )

        state.done = True
        self._log(f"[Controller] Done. prediction={prediction}")
        return prediction, trajectory, str(prompt_trace)
