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
            out.append(
                CandidateHypothesis(
                    component=comp,
                    reason=reason,
                    datetime=dt,
                    evidence=[str(x) for x in list(it.get("evidence") or [])],
                    score=float(it.get("score", 0.0) or 0.0),
                    source=str(it.get("source", "basil_tuple") or "basil_tuple"),
                )
            )
        out.sort(key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True)
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
        if adapter is not None:
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
                        if changed and ((not adapter_ok) or conf < min_conf or (det_score - prop_score) > max_drop):
                            trajectory.append(
                                {
                                    "step": 2,
                                    "action": "llm_veto",
                                    "elapsed_seconds": 0.0,
                                    "output": {
                                        "min_confidence": float(min_conf),
                                        "max_score_drop_vs_det_top1": float(max_drop),
                                        "llm_confidence": float(conf),
                                        "det_score": float(det_score),
                                        "llm_selected_score": float(prop_score),
                                        "adapter_consistency_ok": bool(adapter_ok),
                                        "adapter_veto_reason": str(adapter_veto_reason or ""),
                                    },
                                }
                            )
                        else:
                            top = proposed
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
