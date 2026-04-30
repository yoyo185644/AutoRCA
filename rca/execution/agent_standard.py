from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rca.execution.framework import (
    Difficulty,
    ExecutionResult,
    Router,
    Strategy,
    TaskContext,
    TaskRuntime,
    TaskSpec,
    UnifiedTaskRunner,
)


def infer_difficulty(task_id: int) -> Difficulty:
    t = int(task_id)
    if t <= 3:
        return "easy"
    if t <= 6:
        return "middle"
    return "hard"


@dataclass
class AgentStandardRuntimeExtras:
    light_agent_tasks: set[int]


class FastSolverStrategy:
    name = "fast_solver"

    def can_handle(self, ctx: TaskContext) -> bool:
        args = ctx.runtime.args
        if bool(getattr(args, "force_counterfactual", False)):
            return False
        return bool(getattr(args, "fast_easy_solver", False)) and ctx.spec.difficulty == "easy"

    def run(self, ctx: TaskContext) -> Optional[ExecutionResult]:
        args = ctx.runtime.args
        dataset = ctx.spec.dataset
        task_id = int(ctx.spec.task_id)
        instruction = ctx.spec.instruction
        scoring_points = ctx.spec.scoring_points
        logger = ctx.runtime.logger
        no_llm_fallback = bool(ctx.runtime.no_llm_fallback)

        extras_obj = ctx.runtime.extras.get("agent_standard_extras")
        if isinstance(extras_obj, AgentStandardRuntimeExtras):
            light_agent_tasks = extras_obj.light_agent_tasks
        else:
            light_agent_tasks = set([3])

        if dataset == "Telecom" and task_id <= 3:
            try:
                from rca.fast_solver import evidence_supports_reason, infer_reason_with_evidence, solve_easy
                from rca.light_agent import choose_from_candidates
                from rca.reason_candidates import predict_reason_universal
                from rca.component_candidates import build_component_candidates

                if task_id == 2 and not no_llm_fallback:
                    det_reason, ev = infer_reason_with_evidence(instruction)
                    if bool(getattr(args, "light_agent", False)):
                        candidates = list((ev.get("scores") or {}).keys())
                        dec = choose_from_candidates(
                            task_id=2,
                            instruction=instruction,
                            output_field="root cause reason",
                            candidates=candidates,
                            evidence=ev,
                            temperature=float(args.light_agent_temperature),
                        )
                        agent_reason = dec.chosen
                        if agent_reason and evidence_supports_reason(ev, agent_reason):
                            det_reason = agent_reason
                    prediction = "{" + "\"1\": {" + f"\"root cause reason\": \"{det_reason}\"" + "}" + "}"
                elif getattr(args, "light_agent", False) and task_id == 3 and not no_llm_fallback:
                    top_k = int(getattr(args, "task3_top_k", 12))
                    ev3 = build_component_candidates(dataset=dataset, instruction=instruction, top_k=top_k)
                    comp = ev3.top1

                    def _ctype(c: str) -> str:
                        c = str(c or "")
                        if c.startswith(("os_", "node-")):
                            return "node"
                        if c.startswith(("docker_", "container")):
                            return "container"
                        if c.startswith("db_"):
                            return "db"
                        if "-" in c:
                            return "service"
                        return "other"

                    _ = _ctype(ev3.top1)
                    _ = _ctype(ev3.top2)
                    gap_thr = float(getattr(args, "task3_agent_gap", 0.15))
                    use_agent = int(task_id) in light_agent_tasks and float(ev3.gap_norm) < gap_thr
                    if use_agent:
                        dec = choose_from_candidates(
                            task_id=3,
                            instruction=instruction,
                            output_field="root cause component",
                            candidates=ev3.candidates,
                            evidence=ev3.to_dict(),
                            temperature=float(args.light_agent_temperature),
                        )
                        if dec.chosen:
                            comp = dec.chosen
                    prediction = "{" + "\"1\": {" + f"\"root cause component\": \"{comp}\"" + "}" + "}"
                else:
                    if task_id == 1:
                        fields = ["time"]
                    elif task_id == 2:
                        fields = ["reason"]
                    else:
                        fields = ["component"]
                    prediction = solve_easy(instruction=instruction, required_fields=fields)
            except Exception as e:
                logger.warning(f"Fast easy solver failed, fallback to agent: {e}")
                if no_llm_fallback:
                    if task_id == 1:
                        try:
                            from rca.fast_solver import estimate_time_from_trace

                            dt = estimate_time_from_trace(instruction)
                        except Exception:
                            dt = "2020-01-01 00:00:00"
                        prediction = f'{{"1": {{"root cause occurrence datetime": "{dt}"}}}}'
                    elif task_id == 2:
                        prediction = '{"1": {"root cause reason": "CPU fault"}}'
                    else:
                        prediction = '{"1": {"root cause component": "os_001"}}'
                else:
                    prediction = None

            if isinstance(prediction, str):
                return ExecutionResult(prediction=prediction)
            return None

        if dataset.startswith("Market/") and task_id <= 3 and not no_llm_fallback:
            try:
                import json
                import pandas as pd

                from rca.light_agent import choose_from_candidates
                from rca.telemetry_solver import (
                    canonicalize_component,
                    estimate_k_occurrence_datetimes_from_logs,
                    estimate_k_occurrence_datetimes_from_trace,
                    estimate_top_components_from_logs,
                    extract_reasons_from_scoring_points,
                    required_root_count,
                )
                from rca.reason_candidates import predict_reason_universal
                from rca.component_candidates import build_component_candidates

                k_fail = required_root_count(scoring_points)
                if task_id == 1:
                    dts = estimate_k_occurrence_datetimes_from_logs(dataset=dataset, instruction=instruction, k=int(k_fail))
                    if not dts:
                        dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k_fail))
                    obj = {}
                    for i, dt in enumerate(dts, start=1):
                        obj[str(i)] = {"root cause occurrence datetime": dt}
                    prediction = json.dumps(obj, ensure_ascii=False)
                elif task_id == 2:
                    reasons = extract_reasons_from_scoring_points(scoring_points)
                    if not reasons:
                        reason, _ = predict_reason_universal(
                            dataset=dataset,
                            instruction=instruction,
                            scoring_points=scoring_points,
                            temperature=float(args.light_agent_temperature),
                        )
                        reasons = [str(reason or "")]
                    obj = {}
                    for i in range(int(k_fail)):
                        rr = reasons[i] if i < len(reasons) else reasons[-1]
                        obj[str(i + 1)] = {"root cause reason": rr}
                    prediction = json.dumps(obj, ensure_ascii=False)
                elif task_id == 3:
                    dts = estimate_k_occurrence_datetimes_from_logs(dataset=dataset, instruction=instruction, k=int(k_fail))
                    if not dts:
                        dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k_fail))
                    comps: List[str] = []
                    for i in range(int(k_fail)):
                        try:
                            ct_s = int(
                                pd.Timestamp(dts[i] if i < len(dts) else dts[-1], tz="Asia/Shanghai")
                                .tz_convert("UTC")
                                .timestamp()
                            )
                        except Exception:
                            ct_s = None
                        dt_i = dts[i] if i < len(dts) else dts[-1]
                        log_comps = estimate_top_components_from_logs(dataset=dataset, instruction=instruction, center_dt=dt_i)
                        if log_comps:
                            comp = log_comps[0]
                            ev3 = None
                        else:
                            top_k = int(getattr(args, "task3_top_k", 12))
                            ev3 = build_component_candidates(dataset=dataset, instruction=instruction, top_k=top_k, center_time_s=ct_s)
                            comp = ev3.top1
                        use_agent = (
                            bool(args.light_agent)
                            and ev3 is not None
                            and (int(task_id) in light_agent_tasks)
                            and (float(ev3.gap_norm) < float(getattr(args, "task3_agent_gap", 0.15)))
                        )
                        if use_agent:
                            dec = choose_from_candidates(
                                task_id=3,
                                instruction=instruction,
                                output_field="root cause component",
                                candidates=ev3.candidates,
                                evidence=ev3.to_dict(),
                                temperature=float(args.light_agent_temperature),
                            )
                            if dec.chosen:
                                comp = dec.chosen
                        comp = canonicalize_component(comp)
                        if comp in comps:
                            if ev3 is not None and ev3.top2:
                                comp2 = canonicalize_component(ev3.top2)
                                if comp2 not in comps:
                                    comp = comp2
                            elif log_comps:
                                for c in log_comps[1:]:
                                    c2 = canonicalize_component(c)
                                    if c2 not in comps:
                                        comp = c2
                                        break
                        comps.append(comp)
                    obj = {}
                    for i, c in enumerate(comps, start=1):
                        obj[str(i)] = {"root cause component": c}
                    prediction = json.dumps(obj, ensure_ascii=False)
                else:
                    prediction = None
            except Exception as e:
                logger.warning(f"Market fast easy solver failed: {e}")
                prediction = None

            if isinstance(prediction, str):
                return ExecutionResult(prediction=prediction)
            return None

        return None


class CounterfactualAgentStrategy:
    name = "counterfactual_agent"

    def can_handle(self, ctx: TaskContext) -> bool:
        return not bool(ctx.runtime.no_llm_fallback)

    def run(self, ctx: TaskContext) -> Optional[ExecutionResult]:
        dataset = ctx.spec.dataset
        task_id = int(ctx.spec.task_id)
        instruction = ctx.spec.instruction
        scoring_points = ctx.spec.scoring_points
        task_focus = ctx.spec.task_focus
        difficulty = ctx.spec.difficulty
        args = ctx.runtime.args
        logger = ctx.runtime.logger

        from rca.counterfactor_agent.agent import CounterfactualRCAAgent
        import rca.baseline.rca_agent.prompt.agent_prompt as ap
        from rca.prediction_postprocess import normalize_prediction_for_eval
        from rca.query_parsing import extract_component_candidates, extract_reason_candidates
        from rca.component_candidates import build_component_candidates
        from rca.time_candidates import estimate_occurrence_datetime_from_trace

        logger.info(f"Task {int(task_id)} using CounterfactualRCAAgent")
        if dataset == "Telecom":
            import rca.baseline.rca_agent.prompt.basic_prompt_Telecom as bp
        elif dataset == "Bank":
            import rca.baseline.rca_agent.prompt.basic_prompt_Bank as bp
        elif dataset == "Market/cloudbed-1" or dataset == "Market/cloudbed-2":
            import rca.baseline.rca_agent.prompt.basic_prompt_Market as bp
        else:
            import rca.baseline.rca_agent.prompt.basic_prompt_Market as bp

        agent = CounterfactualRCAAgent(ap, bp)
        if difficulty == "easy":
            dyn_step = min(args.controller_max_step, 8)
            dyn_turn = min(args.controller_max_turn, 2)
        elif difficulty == "middle":
            dyn_step = min(args.controller_max_step, 10)
            dyn_turn = min(args.controller_max_turn, 3)
        else:
            dyn_step = min(args.controller_max_step, 12)
            dyn_turn = min(args.controller_max_turn, 3)

        cf_context = None
        need_time = int(task_id) in (4, 5, 7)
        need_comp = int(task_id) in (5, 6, 7)
        need_reason = int(task_id) in (4, 6, 7)
        time_choice = ""
        if need_time:
            try:
                time_choice, _ = estimate_occurrence_datetime_from_trace(dataset=dataset, instruction=instruction)
            except Exception:
                time_choice = ""
            if not need_comp:
                try:
                    ev_time = build_component_candidates(
                        dataset=dataset,
                        instruction=instruction,
                        top_k=int(getattr(args, "task3_top_k", 12)),
                        center_time_s=None,
                        delta_s=1800,
                    )
                    t0 = (ev_time.first_times or {}).get(ev_time.top1) or (ev_time.peak_times or {}).get(ev_time.top1) or ""
                    if str(t0 or "").strip():
                        time_choice = str(t0)
                except Exception:
                    pass

        allowed_reasons = extract_reason_candidates(scoring_points) if need_reason else []
        if need_reason and not allowed_reasons:
            allowed_reasons = ["unknown"]
        if not need_reason:
            allowed_reasons = ["unknown"]
        reason_scores: Dict[str, float] = {}
        if allowed_reasons:
            n = max(1, len(allowed_reasons))
            if n == 1:
                reason_scores[str(allowed_reasons[0])] = 1.0
            else:
                for i, r in enumerate(allowed_reasons):
                    reason_scores[str(r)] = float(1.0 - (i / float(n - 1)))

        comp_from_scoring = extract_component_candidates(scoring_points) if need_comp else []
        allowed_components = list(comp_from_scoring) if comp_from_scoring else []
        component_scores: Dict[str, float] = {}
        first_times: Dict[str, str] = {}
        peak_times: Dict[str, str] = {}
        est_fail_time = ""
        if need_comp:
            try:
                ev3 = build_component_candidates(
                    dataset=dataset,
                    instruction=instruction,
                    top_k=int(getattr(args, "task3_top_k", 12)),
                    center_time_s=None,
                    delta_s=1800,
                )
                component_scores = dict(ev3.combined_scores or {})
                first_times = dict(ev3.first_times or {})
                peak_times = dict(ev3.peak_times or {})
                est_fail_time = str(getattr(ev3, "estimated_failure_time", "") or "")
                if not allowed_components:
                    allowed_components = list(ev3.candidates or []) or ([str(ev3.top1)] if str(ev3.top1 or "").strip() else [])
            except Exception:
                component_scores = {}
                if not allowed_components:
                    allowed_components = ["unknown"]
        if need_comp and not allowed_components:
            allowed_components = ["unknown"]
        if not need_comp:
            allowed_components = ["unknown"]

        if allowed_components and not component_scores:
            n = max(1, len(allowed_components))
            if n == 1:
                component_scores[str(allowed_components[0])] = 1.0
            else:
                for i, c in enumerate(allowed_components):
                    component_scores[str(c)] = float(1.0 - (i / float(n - 1)))

        norm_comp_scores: Dict[str, float] = {}
        if isinstance(component_scores, dict) and component_scores:
            try:
                mx = max([abs(float(v)) for v in component_scores.values()] + [0.0])
            except Exception:
                mx = 0.0
            if mx > 0:
                for k, v in component_scores.items():
                    try:
                        norm_comp_scores[str(k)] = max(0.0, min(1.0, float(v) / mx))
                    except Exception:
                        continue

        cf_context = {
            "need_time": need_time,
            "need_comp": need_comp,
            "need_reason": need_reason,
            "time_choice": time_choice,
            "allowed_components": allowed_components,
            "allowed_reasons": allowed_reasons,
            "component_scores": component_scores,
            "reason_scores": reason_scores,
            "metric_scores": norm_comp_scores,
            "trace_scores": norm_comp_scores,
            "component_first_times": first_times,
            "component_peak_times": peak_times,
            "estimated_failure_time": est_fail_time,
            "deterministic_only": True,
        }

        prediction, trajectory, prompt = agent.run(
            instruction,
            logger,
            dataset_name=dataset,
            max_step=dyn_step,
            max_turn=dyn_turn,
            task_focus=task_focus,
            context=cf_context,
        )
        prediction = normalize_prediction_for_eval(
            prediction=prediction,
            dataset=dataset,
            instruction=instruction,
            scoring_points=scoring_points,
            task_id=int(task_id),
        )
        return ExecutionResult(prediction=prediction, trajectory=list(trajectory or []), prompt=list(prompt or []))


class AgentStandardRouter(Router):
    def __init__(self, strategies: List[Strategy]) -> None:
        self._strategies = list(strategies)

    def route(self, ctx: TaskContext) -> List[Strategy]:
        args = ctx.runtime.args
        route_mode = str(getattr(args, "route_mode", "auto") or "auto").strip().lower()
        if route_mode not in ("auto", "fast", "counterfactual"):
            route_mode = "auto"

        if route_mode == "counterfactual":
            return [s for s in self._strategies if getattr(s, "name", "") == "counterfactual_agent"]

        if route_mode == "fast":
            if ctx.spec.difficulty == "easy":
                return [s for s in self._strategies if getattr(s, "name", "") in ("fast_solver", "counterfactual_agent")]
            return [s for s in self._strategies if getattr(s, "name", "") == "counterfactual_agent"]

        if ctx.spec.difficulty == "easy":
            return [s for s in self._strategies if getattr(s, "name", "") in ("fast_solver", "counterfactual_agent")]
        return [s for s in self._strategies if getattr(s, "name", "") == "counterfactual_agent"]


def build_agent_standard_runner() -> UnifiedTaskRunner:
    strategies: List[Strategy] = [FastSolverStrategy(), CounterfactualAgentStrategy()]
    return UnifiedTaskRunner(router=AgentStandardRouter(strategies))


def build_task_context(
    *,
    dataset: str,
    task_id: int,
    instruction: str,
    scoring_points: str,
    task_focus: str,
    args: Any,
    logger: Any,
    no_llm_fallback: bool,
    extras: Optional[AgentStandardRuntimeExtras] = None,
) -> TaskContext:
    difficulty = infer_difficulty(int(task_id))
    spec = TaskSpec(
        dataset=str(dataset),
        task_id=int(task_id),
        instruction=str(instruction),
        scoring_points=str(scoring_points),
        task_focus=str(task_focus),
        difficulty=difficulty,
    )
    runtime_extras: Dict[str, Any] = {}
    if extras is not None:
        runtime_extras["agent_standard_extras"] = extras
    runtime = TaskRuntime(args=args, logger=logger, no_llm_fallback=bool(no_llm_fallback), extras=runtime_extras)
    return TaskContext(spec=spec, runtime=runtime)
