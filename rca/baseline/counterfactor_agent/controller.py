from typing import Any, Dict, List, Tuple

from .executor import CounterfactualExecutor
from .schemas import ControllerState
from .utils import format_prediction_json


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

        def pick_time(component: str) -> str:
            if not need_time:
                return ""
            t = str(context.get("time_choice", "") or "").strip()
            if t:
                return t
            if need_comp:
                ft = (context.get("component_first_times") or {}).get(component)
                if str(ft or "").strip():
                    return str(ft)
                pt = (context.get("component_peak_times") or {}).get(component)
                if str(pt or "").strip():
                    return str(pt)
                est = str(context.get("estimated_failure_time", "") or "").strip()
                if est:
                    return est
            return ""

        # Step 1: 候选生成
        state.step += 1
        state.turn += 1
        self._log("[Controller] Step 1: proposing candidates...")
        candidates = self.executor.propose_candidates(
            instruction=instruction,
            dataset_name=dataset_name,
            task_focus=task_focus,
            top_k=5,
            context=context,
        )
        state.candidates = candidates

        trajectory.append(
            {
                "step": state.step,
                "action": "propose_candidates",
                "output": [c.to_dict() for c in candidates],
            }
        )

        if not candidates:
            prediction = format_prediction_json(
                component="unknown",
                reason="unknown" if need_reason else "",
                confidence=0.0,
                reasoning="No candidate hypothesis could be generated.",
                occurrence_datetime=pick_time("unknown"),
            )
            state.done = True
            trajectory.append(
                {
                    "step": state.step,
                    "action": "terminate",
                    "reason": "no_candidates",
                }
            )
            return prediction, trajectory, str(prompt_trace)

        if bool(context.get("deterministic_only", False)):
            top = candidates[0]
            prediction = format_prediction_json(
                component=top.component if need_comp else "",
                reason=top.reason if need_reason else "",
                confidence=top.score,
                reasoning="Deterministic-only mode: selected the top-ranked hypothesis.",
                occurrence_datetime=pick_time(top.component),
            )
            state.done = True
            trajectory.append(
                {
                    "step": state.step,
                    "action": "deterministic_finalize",
                    "output": top.to_dict(),
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

            cf_info = self.executor.build_counterfactual_question(
                instruction=instruction,
                candidate=cand,
                dataset_name=dataset_name,
                context=context,
            )
            counterfactual_question = cf_info.get("counterfactual_question", "")
            expected_effect = cf_info.get("expected_effect", "")

            trajectory.append(
                {
                    "step": state.step,
                    "action": "build_counterfactual_question",
                    "candidate": cand.to_dict(),
                    "output": cf_info,
                }
            )

            if state.step >= state.max_step:
                break

            state.step += 1
            self._log(f"[Controller] Step {state.step}: validate candidate #{idx + 1}")

            check = self.executor.validate_candidate(
                instruction=instruction,
                candidate=cand,
                counterfactual_question=counterfactual_question,
                expected_effect=expected_effect,
                dataset_name=dataset_name,
                context=context,
            )
            state.checks.append(check)

            trajectory.append(
                {
                    "step": state.step,
                    "action": "validate_candidate",
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
            refined = self.executor.refine_candidates(state.checks, top_k=3)
            trajectory.append(
                {
                    "step": state.step,
                    "action": "refine_candidates",
                    "output": [c.to_dict() for c in refined],
                }
            )
        else:
            refined = state.candidates[:3]

        # Step 4: 最终决策
        if state.step < state.max_step:
            state.step += 1
        self._log(f"[Controller] Step {state.step}: finalize result...")

        final = self.executor.finalize(state.checks, refined)

        prediction = format_prediction_json(
            component=final["component"],
            reason=final["reason"] if need_reason else "",
            confidence=final["confidence"],
            reasoning=final["reasoning"],
            occurrence_datetime=pick_time(final["component"]),
        )

        trajectory.append(
            {
                "step": state.step,
                "action": "finalize",
                "output": final,
            }
        )

        state.done = True
        self._log(f"[Controller] Done. prediction={prediction}")
        return prediction, trajectory, str(prompt_trace)
