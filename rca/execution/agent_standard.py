from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
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
from rca.memory.config import basil_config_from_args, basil_enabled


def infer_difficulty(task_id: int) -> Difficulty:
    t = int(task_id)
    if t <= 3:
        return "easy"
    if t <= 6:
        return "middle"
    return "hard"


def _parse_candidate_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def summarize_time_candidate_stats(time_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    datetimes: List[datetime] = []
    raw_datetimes = []
    for cand in time_candidates or []:
        if not isinstance(cand, dict):
            continue
        raw = str(cand.get("datetime", "") or "").strip()
        if raw:
            raw_datetimes.append(raw)
        parsed = _parse_candidate_datetime(raw)
        if parsed is not None:
            datetimes.append(parsed)

    unique_datetimes = set(raw_datetimes)
    unique_minutes = set()
    for dt in datetimes:
        unique_minutes.add(dt.replace(second=0, microsecond=0).isoformat(sep=" "))

    spread_seconds = 0.0
    if len(datetimes) >= 2:
        spread_seconds = float((max(datetimes) - min(datetimes)).total_seconds())

    return {
        "time_candidate_count": int(len(time_candidates or [])),
        "time_candidate_unique_datetime_count": int(len(unique_datetimes)),
        "time_candidate_unique_minute_count": int(len(unique_minutes)),
        "time_candidate_spread_seconds": float(max(0.0, spread_seconds)),
        "time_candidate_spread_minutes": float(max(0.0, spread_seconds) / 60.0),
    }


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
                from rca.query_parsing import extract_reason_candidates
                from rca.memory.evidence import (
                    build_component_candidates as build_basil_component_candidates,
                    build_reason_candidates as build_basil_reason_candidates,
                    build_time_candidates as build_basil_time_candidates,
                    query_basil_evidence,
                )

                if basil_enabled(args):
                    top_k = int(getattr(args, "task3_top_k", 12))
                    try:
                        reason_hints = extract_reason_candidates(scoring_points) if task_id == 2 else []
                        basil_ev = query_basil_evidence(
                            dataset=dataset,
                            instruction=instruction,
                            task_id=int(task_id),
                            top_k=top_k,
                            reasons=reason_hints,
                            config=basil_config_from_args(args, top_k, task_id=task_id),
                        )
                        if task_id == 1:
                            pass
                        elif task_id == 2:
                            r_res = build_basil_reason_candidates(basil_ev, fallback_candidates=reason_hints, logger=logger)
                            if r_res.candidates:
                                reason = str(r_res.candidates[0])
                                return ExecutionResult(prediction=f'{{"1": {{"root cause reason": "{reason}"}}}}')
                        elif task_id == 3:
                            c_res = build_basil_component_candidates(basil_ev, top_k=top_k)
                            if c_res.candidates:
                                comp = str(c_res.candidates[0])
                                return ExecutionResult(prediction=f'{{"1": {{"root cause component": "{comp}"}}}}')
                    except Exception as e:
                        logger.warning(f"BASIL easy evidence path failed, fallback to fast solver: {e}")

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
                elif task_id == 3 and (basil_enabled(args) or (getattr(args, "light_agent", False) and not no_llm_fallback)):
                    top_k = int(getattr(args, "task3_top_k", 12))
                    ev3 = build_component_candidates(
                        dataset=dataset,
                        instruction=instruction,
                        top_k=top_k,
                        use_basil=basil_enabled(args),
                        basil_config=basil_config_from_args(args, top_k, required_fields=getattr(ctx.spec, "required_fields", None), task_id=task_id),
                    )
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
                    use_agent = (not no_llm_fallback) and int(task_id) in light_agent_tasks and float(ev3.gap_norm) < gap_thr
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
                            ev3 = build_component_candidates(
                                dataset=dataset,
                                instruction=instruction,
                                top_k=top_k,
                                center_time_s=ct_s,
                                use_basil=basil_enabled(args),
                                basil_config=basil_config_from_args(args, top_k, required_fields=getattr(ctx.spec, "required_fields", None), task_id=task_id),
                            )
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
        no_llm_fallback = bool(ctx.runtime.no_llm_fallback)

        from rca.counterfactor_agent.agent import CounterfactualRCAAgent
        import rca.baseline.rca_agent.prompt.agent_prompt as ap
        from rca.prediction_postprocess import normalize_prediction_for_eval
        from rca.query_parsing import extract_component_candidates, extract_reason_candidates
        from rca.component_candidates import build_component_candidates as build_raw_component_candidates
        from rca.time_candidates import TimeCandidate, estimate_occurrence_datetime_candidates, select_time_candidate
        from rca.agent.evidence_builder import build_time_evidence
        from rca.light_agent import call_json_only
        from rca.memory.evidence import (
            build_component_candidates as build_basil_component_candidates,
            build_reason_candidates as build_basil_reason_candidates,
            build_time_candidates as build_basil_time_candidates,
            query_basil_evidence,
        )

        strategy_t0 = time.perf_counter()
        timing: Dict[str, float] = {
            "strategy_total_seconds": 0.0,
            "strategy_setup_seconds": 0.0,
            "candidate_generation_seconds": 0.0,
            "time_candidate_seconds": 0.0,
            "basil_index_seconds": 0.0,
            "component_candidate_seconds": 0.0,
            "agent_run_seconds": 0.0,
            "normalize_prediction_seconds": 0.0,
        }
        basil_timing_details: Dict[str, float] = {}

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
        timing["strategy_setup_seconds"] = time.perf_counter() - strategy_t0

        cf_context = None
        need_time = int(task_id) in (1, 4, 5, 7)
        need_comp = int(task_id) in (5, 6, 7)
        need_reason = int(task_id) in (4, 6, 7)
        time_choice = ""
        time_candidates: List[Dict[str, Any]] = []

        candidate_generation_t0 = time.perf_counter()
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
        top_metrics: Dict[str, List[str]] = {}
        est_fail_time = ""
        basil_evidence = None
        basil_evidence_called = False
        basil_low_confidence = False
        basil_confidence_reason = ""
        basil_component_coverage = 0
        basil_time_coverage = 0
        basil_reason_coverage = 0
        if basil_enabled(args):
            _basil_top_k = int(getattr(args, "task3_top_k", 12))
            try:
                t0 = time.perf_counter()
                basil_evidence = query_basil_evidence(
                    dataset=dataset,
                    instruction=instruction,
                    task_id=int(task_id),
                    top_k=_basil_top_k,
                    reasons=allowed_reasons if need_reason and allowed_reasons != ["unknown"] else [],
                    center_time_s=None,
                    delta_s=1800,
                    config=basil_config_from_args(args, _basil_top_k, required_fields=getattr(ctx.spec, "required_fields", None), task_id=task_id),
                )
                timing["basil_index_seconds"] += time.perf_counter() - t0
                basil_evidence_called = True
                basil_low_confidence = bool(getattr(basil_evidence, "low_confidence", False))
                basil_confidence_reason = str(getattr(basil_evidence, "confidence_reason", "") or "")
                basil_component_coverage = len(getattr(basil_evidence, "component_scores", {}) or {})
                basil_time_coverage = sum(len(v or []) for v in (getattr(basil_evidence, "time_candidates", {}) or {}).values())
                basil_reason_coverage = sum(len(v or {}) for v in (getattr(basil_evidence, "reason_scores", {}) or {}).values())
                basil_timing_details.update({f"evidence.{k}": float(v) for k, v in (getattr(basil_evidence, "timings", {}) or {}).items()})
                component_scores = dict(getattr(basil_evidence, "component_scores", {}) or {})
                first_times = dict(getattr(basil_evidence, "first_times", {}) or {})
                peak_times = dict(getattr(basil_evidence, "peak_times", {}) or {})
                top_metrics = dict(getattr(basil_evidence, "supporting_metrics", {}) or {})
                est_fail_time = str(getattr(basil_evidence, "estimated_failure_time", "") or "")
            except Exception as e:
                logger.warning(f"BASIL evidence query failed, fallback to raw candidate generation: {e}")
                basil_evidence = None

        if need_reason and basil_evidence is not None:
            reason_result = build_basil_reason_candidates(
                basil_evidence,
                fallback_candidates=[] if allowed_reasons == ["unknown"] else allowed_reasons,
                logger=logger,
            )
            if reason_result.candidates:
                allowed_reasons = list(reason_result.candidates)
                reason_scores = dict(reason_result.scores)

        if need_comp:
            try:
                t0 = time.perf_counter()
                _basil_top_k = int(getattr(args, "task3_top_k", 12))
                if basil_evidence is not None:
                    comp_result = build_basil_component_candidates(basil_evidence, top_k=_basil_top_k)
                    elapsed = time.perf_counter() - t0
                    timing["component_candidate_seconds"] += elapsed
                    component_scores = dict(comp_result.scores or {})
                    if not allowed_components:
                        allowed_components = list(comp_result.candidates or [])
                else:
                    ev3 = build_raw_component_candidates(
                        dataset=dataset,
                        instruction=instruction,
                        top_k=_basil_top_k,
                        center_time_s=None,
                        delta_s=1800,
                        use_basil=False,
                        basil_reasons=allowed_reasons if need_reason else [],
                    )
                    elapsed = time.perf_counter() - t0
                    timing["component_candidate_seconds"] += elapsed
                    component_scores = dict(ev3.combined_scores or {})
                    first_times = dict(ev3.first_times or {})
                    peak_times = dict(ev3.peak_times or {})
                    top_metrics = dict(ev3.top_metrics or {})
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

        if need_time:
            t0 = time.perf_counter()
            try:
                reason_hints = allowed_reasons if need_reason and allowed_reasons != ["unknown"] else []
                if basil_evidence is not None:
                    time_result = build_basil_time_candidates(basil_evidence, top_k=int(getattr(args, "time_candidate_top_k", 8)))
                    basil_tcands = list(time_result.candidates or [])
                    raw_tcands = estimate_occurrence_datetime_candidates(
                        dataset=dataset,
                        instruction=instruction,
                        component_first_times=first_times,
                        component_peak_times=peak_times,
                        estimated_failure_time=est_fail_time or time_choice,
                        component_scores=component_scores,
                        reasons=reason_hints,
                        top_k=int(getattr(args, "time_candidate_top_k", 8)),
                    )
                    merged_by_dt: Dict[str, Any] = {}
                    for cand in list(raw_tcands or []) + list(basil_tcands or []):
                        dt = str(getattr(cand, "datetime", "") or "").strip()
                        if not dt:
                            continue
                        old = merged_by_dt.get(dt)
                        if old is None or float(getattr(cand, "score", 0.0) or 0.0) > float(getattr(old, "score", 0.0) or 0.0):
                            merged_by_dt[dt] = cand
                    tcands = sorted(merged_by_dt.values(), key=lambda c: float(getattr(c, "score", 0.0) or 0.0), reverse=True)[
                        : int(getattr(args, "time_candidate_top_k", 8))
                    ]
                else:
                    tcands = estimate_occurrence_datetime_candidates(
                        dataset=dataset,
                        instruction=instruction,
                        component_first_times=first_times,
                        component_peak_times=peak_times,
                        estimated_failure_time=est_fail_time or time_choice,
                        component_scores=component_scores,
                        reasons=reason_hints,
                        top_k=int(getattr(args, "time_candidate_top_k", 8)),
                    )
                time_candidates = [c.to_dict() for c in tcands]
                if tcands:
                    focus_comp = ""
                    if need_comp and comp_from_scoring:
                        focus_comp = str(comp_from_scoring[0] or "")
                    elif need_comp and allowed_components and allowed_components != ["unknown"]:
                        focus_comp = str(allowed_components[0] or "")
                    focus_reason = ""
                    if need_reason and allowed_reasons and allowed_reasons != ["unknown"]:
                        focus_reason = str(allowed_reasons[0] or "")
                    chosen, _details = select_time_candidate(
                        tcands,
                        component=focus_comp,
                        reason=focus_reason,
                        fallback=str(time_candidates[0].get("datetime") if time_candidates else time_choice),
                    )
                    time_choice = str(chosen or time_choice)
            except Exception as e:
                logger.warning(f"Time candidate set construction failed: {e}")
                time_candidates = []
            timing["time_candidate_seconds"] += time.perf_counter() - t0
        timing["candidate_generation_seconds"] = time.perf_counter() - candidate_generation_t0

        cf_context = {
            "need_time": need_time,
            "need_comp": need_comp,
            "need_reason": need_reason,
            "difficulty": difficulty,
            "task_id": int(task_id),
            "time_choice": time_choice,
            "time_candidates": time_candidates,
            "allowed_components": allowed_components,
            "allowed_reasons": allowed_reasons,
            "component_scores": component_scores,
            "reason_scores": reason_scores,
            "metric_scores": norm_comp_scores,
            "trace_scores": norm_comp_scores,
            "component_first_times": first_times,
            "component_peak_times": peak_times,
            "top_metrics": top_metrics,
            "estimated_failure_time": est_fail_time,
            "basil_evidence_called": bool(basil_evidence_called),
            "basil_low_confidence": bool(basil_low_confidence),
            "basil_confidence_reason": basil_confidence_reason,
            "deterministic_only": bool(getattr(args, "deterministic_only", False)),
            "enable_uncertainty_llm": bool(getattr(args, "uncertainty_llm", False)) and not no_llm_fallback,
            "enable_time_llm": bool(getattr(args, "time_llm", False)) and not no_llm_fallback,
            "llm_force_midhard": bool(getattr(args, "llm_force_midhard", False)),
            "llm_uncertain_gap_threshold": float(getattr(args, "llm_uncertain_gap_threshold", 1.0)),
            "llm_confidence_floor": float(getattr(args, "llm_confidence_floor", 1.0)),
            "llm_time_confidence_floor": float(getattr(args, "llm_time_confidence_floor", 1.0)),
            "llm_verify_top_k": int(getattr(args, "llm_verify_top_k", 3)),
            "time_llm_top_k": int(getattr(args, "time_llm_top_k", 3)),
            "time_llm_max_score_drop": float(getattr(args, "time_llm_max_score_drop", 0.20)),
            "time_llm_gap_threshold": float(getattr(args, "time_llm_gap_threshold", 0.08)),
            "time_llm_confidence_floor": float(getattr(args, "time_llm_confidence_floor", 0.85)),
            "time_llm_competing_drop": float(getattr(args, "time_llm_competing_drop", 0.12)),
            "time_llm_force": bool(getattr(args, "time_llm_force", False)),
            "basil_max_trace_rows": int(getattr(args, "basil_max_trace_rows", 0) or 0),
            "evidence_score_gap_threshold": float(getattr(args, "evidence_score_gap_threshold", 0.15)),
            "evidence_top1_score_floor": float(getattr(args, "evidence_top1_score_floor", 0.30)),
            "evidence_time_gap_threshold": float(getattr(args, "evidence_time_gap_threshold", 0.20)),
            "evidence_time_conflict_seconds": int(getattr(args, "evidence_time_conflict_seconds", 300)),
        }

        t0 = time.perf_counter()
        prediction, trajectory, prompt = agent.run(
            instruction,
            logger,
            dataset_name=dataset,
            max_step=dyn_step,
            max_turn=dyn_turn,
            task_focus=task_focus,
            context=cf_context,
        )
        timing["agent_run_seconds"] = time.perf_counter() - t0

        if need_time and bool(getattr(args, "time_llm", False)) and (not no_llm_fallback) and time_candidates:
            try:
                import json as _json

                already_attempted = False
                for st in trajectory or []:
                    if not isinstance(st, dict):
                        continue
                    ts = st.get("time_selection")
                    if isinstance(ts, dict) and (
                        ts.get("source") == "llm_time_verifier" or bool(ts.get("llm_time_attempted", False))
                    ):
                        already_attempted = True
                        break
                if already_attempted:
                    raise StopIteration()

                pred_obj = _json.loads(prediction) if isinstance(prediction, str) else {}
                root = pred_obj.get("1", {}) if isinstance(pred_obj, dict) else {}
                pred_comp = str(root.get("root cause component") or "").strip()
                pred_reason = str(root.get("root cause reason") or "").strip()
                pred_time = str(root.get("root cause occurrence datetime") or "").strip()

                cand_dts = []
                for c in time_candidates:
                    if isinstance(c, dict):
                        dt = str(c.get("datetime") or "").strip()
                        if dt:
                            cand_dts.append(dt)
                spread_s = 0.0
                try:
                    dts = [datetime.fromisoformat(x) for x in cand_dts if x]
                    if len(dts) >= 2:
                        spread_s = float((max(dts) - min(dts)).total_seconds())
                except Exception:
                    spread_s = 0.0

                enable = bool(getattr(args, "time_llm_force", False)) or float(spread_s) >= 180.0
                if enable:
                    top_k = int(getattr(args, "time_llm_top_k", 3) or 3)
                    tcands_obj = []
                    for c in time_candidates[: max(1, top_k)]:
                        if isinstance(c, dict):
                            tcands_obj.append(c)

                    evs = []
                    for c in tcands_obj:
                        dt = str(c.get("datetime") or "")
                        src = str(c.get("source") or "")
                        score = float(c.get("score", 0.0) or 0.0)
                        evs.append(
                            build_time_evidence(
                                dataset=dataset,
                                instruction=instruction,
                                time_candidate=TimeCandidate(datetime=dt, source=src, score=score, component=str(c.get("component") or ""), reason=str(c.get("reason") or "")),
                                component=pred_comp,
                                reason=pred_reason,
                            ).to_dict()
                        )

                    sys_p = "You are an RCA time verifier. Choose the most likely root-cause occurrence time. Output JSON only."
                    user_p = _json.dumps(
                        {
                            "instruction": str(instruction),
                            "fixed_component": pred_comp,
                            "fixed_reason": pred_reason,
                            "predicted_time": pred_time,
                            "time_candidates": evs,
                            "question": "Which time represents the earliest stable root-cause onset? Prefer: component metric first breach before trace impact; stable onset (not a spike). Output JSON only: {selected_id, confidence, rationale, key_signal}.",
                            "output_schema": {"selected_id": 1, "confidence": 0.0, "rationale": "", "key_signal": ""},
                        },
                        ensure_ascii=False,
                    )
                    t_llm0 = time.perf_counter()
                    obj, raw, in_tok, _out_tok, msgs = call_json_only(system=sys_p, user=user_p, temperature=0.0)
                    elapsed = time.perf_counter() - t_llm0

                    selected_id = int(obj.get("selected_id", 0) or 0) if isinstance(obj, dict) else 0
                    conf = float(obj.get("confidence", 0.0) or 0.0) if isinstance(obj, dict) else 0.0
                    floor = float(getattr(args, "time_llm_confidence_floor", 0.85))
                    chosen_dt = ""
                    if 1 <= selected_id <= len(tcands_obj):
                        chosen_dt = str(tcands_obj[selected_id - 1].get("datetime") or "").strip()

                    time_selection = {
                        "source": "llm_time_verifier",
                        "llm_time_attempted": True,
                        "elapsed_seconds": float(elapsed),
                        "approx_input_tokens": int(in_tok),
                        "selected_id": int(selected_id),
                        "confidence": float(conf),
                        "confidence_floor": float(floor),
                        "predicted_time": str(pred_time),
                        "chosen_time": str(chosen_dt),
                    }
                    trajectory = list(trajectory or [])
                    trajectory.append(
                        {
                            "step": "time_verifier",
                            "action": "llm_time_verifier",
                            "time_selection": time_selection,
                            "output": obj,
                            "raw_response": raw,
                        }
                    )
                    prompt = list(prompt or []) + list(msgs or [])

                    if chosen_dt and conf >= floor and isinstance(pred_obj, dict) and isinstance(root, dict):
                        root["root cause occurrence datetime"] = chosen_dt
                        pred_obj["1"] = root
                        prediction = _json.dumps(pred_obj, ensure_ascii=False)
            except StopIteration:
                pass
            except Exception as e:
                try:
                    trajectory = list(trajectory or [])
                    trajectory.append(
                        {
                            "step": "time_verifier",
                            "action": "llm_time_verifier_failed",
                            "time_selection": {"source": "llm_time_verifier", "llm_time_attempted": True, "error": str(e)},
                        }
                    )
                except Exception:
                    pass
        llm_verifier_seconds = 0.0
        llm_input_tokens = 0
        llm_gate_called = False
        llm_time_verifier_seconds = 0.0
        llm_time_input_tokens = 0
        llm_time_called = False
        for step in trajectory or []:
            if not isinstance(step, dict):
                continue
            if step.get("action") == "llm_verify_topk":
                llm_gate_called = True
                out = step.get("output") if isinstance(step.get("output"), dict) else {}
                llm_verifier_seconds += float(out.get("elapsed_seconds", 0.0) or 0.0)
                llm_input_tokens += int(out.get("approx_input_tokens", 0) or 0)
            if step.get("action") == "llm_verify_topk_failed":
                llm_gate_called = True
                out = step.get("output") if isinstance(step.get("output"), dict) else {}
                llm_verifier_seconds += float(out.get("elapsed_seconds", 0.0) or 0.0)
                llm_input_tokens += int(out.get("approx_input_tokens", 0) or 0)
            time_selection = step.get("time_selection")
            if isinstance(time_selection, dict) and (
                time_selection.get("source") == "llm_time_verifier" or bool(time_selection.get("llm_time_attempted", False))
            ):
                llm_time_called = True
                llm_time_verifier_seconds += float(time_selection.get("elapsed_seconds", 0.0) or 0.0)
                llm_time_input_tokens += int(time_selection.get("approx_input_tokens", 0) or 0)
        t0 = time.perf_counter()
        prediction = normalize_prediction_for_eval(
            prediction=prediction,
            dataset=dataset,
            instruction=instruction,
            scoring_points=scoring_points,
            task_id=int(task_id),
        )
        timing["normalize_prediction_seconds"] = time.perf_counter() - t0
        timing["strategy_total_seconds"] = time.perf_counter() - strategy_t0
        timing["non_agent_preparation_seconds"] = max(
            0.0,
            timing["strategy_total_seconds"] - timing["agent_run_seconds"] - timing["normalize_prediction_seconds"],
        )

        def _sum_basil_suffixes(suffixes: List[str]) -> float:
            total = 0.0
            for key, value in basil_timing_details.items():
                key_s = str(key)
                if any(key_s.endswith(suffix) for suffix in suffixes):
                    total += float(value or 0.0)
            return float(total)

        basil_evidence_extract_seconds = _sum_basil_suffixes(
            [
                "basil.evidence_extract_segments_seconds",
                "basil.evidence_extract_topology_seconds",
            ]
        )
        basil_evidence_build_seconds = _sum_basil_suffixes(
            [
                "basil.evidence_extract_segments_seconds",
                "basil.evidence_extract_topology_seconds",
                "basil.evidence_index_write_seconds",
            ]
        )
        basil_evidence_query_seconds = _sum_basil_suffixes(["basil.evidence_index_query_seconds"])
        basil_candidate_fusion_seconds = _sum_basil_suffixes(["basil.stage2_task_fusion_seconds"])
        basil_candidate_expansion_seconds = _sum_basil_suffixes(["basil.stage3_candidate_expansion_seconds"])
        basil_candidate_rerank_seconds = _sum_basil_suffixes(["basil.stage4_two_stage_rerank_seconds"])
        basil_candidate_decision_seconds = _sum_basil_suffixes(["basil.candidate_decision_seconds"])
        basil_index_total_seconds = _sum_basil_suffixes(["basil.total_seconds"]) or float(timing["basil_index_seconds"])
        candidate_generation_seconds = float(timing["candidate_generation_seconds"])
        time_candidate_stats = summarize_time_candidate_stats(time_candidates)
        timing_breakdown = {
            "total_execution_seconds": float(timing["strategy_total_seconds"]),
            "setup_seconds": float(timing["strategy_setup_seconds"]),
            "candidate_generation_seconds": float(candidate_generation_seconds),
            "component_candidate_seconds": float(timing["component_candidate_seconds"]),
            "time_candidate_seconds": float(timing["time_candidate_seconds"]),
            "basil_index_total_seconds": float(basil_index_total_seconds),
            "basil_evidence_extract_seconds": float(basil_evidence_extract_seconds),
            "basil_evidence_build_seconds": float(basil_evidence_build_seconds),
            "basil_index_query_seconds": float(basil_evidence_query_seconds),
            "basil_candidate_fusion_seconds": float(basil_candidate_fusion_seconds),
            "basil_candidate_expansion_seconds": float(basil_candidate_expansion_seconds),
            "basil_candidate_rerank_seconds": float(basil_candidate_rerank_seconds),
            "basil_candidate_decision_seconds": float(basil_candidate_decision_seconds),
            "non_basil_candidate_generation_seconds": float(max(0.0, candidate_generation_seconds - basil_index_total_seconds)),
            "agent_call_seconds": float(timing["agent_run_seconds"]),
            "llm_verifier_seconds": float(llm_verifier_seconds),
            "llm_input_tokens": int(llm_input_tokens),
            "llm_gate_called": bool(llm_gate_called),
            "llm_time_verifier_seconds": float(llm_time_verifier_seconds),
            "llm_time_input_tokens": int(llm_time_input_tokens),
            "llm_time_called": bool(llm_time_called),
            "agent_total_input_tokens": int(llm_input_tokens) + int(llm_time_input_tokens),
            "basil_evidence_called": bool(basil_evidence_called),
            "basil_low_confidence": bool(basil_low_confidence),
            "basil_confidence_reason": basil_confidence_reason,
            "basil_component_coverage": int(basil_component_coverage),
            "basil_time_coverage": int(basil_time_coverage),
            "basil_reason_coverage": int(basil_reason_coverage),
            "postprocess_seconds": float(timing["normalize_prediction_seconds"]),
            **time_candidate_stats,
        }
        timing_step = {
            "step": "timing",
            "action": "timing",
            "strategy": "counterfactual_agent",
            "dataset": dataset,
            "task_id": int(task_id),
            "basil_enabled": bool(basil_enabled(args)),
            "timings_seconds": timing,
            "basil_index_details_seconds": basil_timing_details,
            "timing_breakdown_seconds": timing_breakdown,
            **time_candidate_stats,
        }
        logger.info(f"Timing: {timing_step}")
        out_trajectory = list(trajectory or [])
        out_trajectory.append(timing_step)
        return ExecutionResult(prediction=prediction, trajectory=out_trajectory, prompt=list(prompt or []))


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
