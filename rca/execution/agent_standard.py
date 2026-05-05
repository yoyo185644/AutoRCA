from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from rca.baseline.rca_agent.prompt.agent_prompt import rules as _AGENT_RULES
from rca.baseline.rca_agent.prompt.basic_prompt_Bank import cand as _BANK_CAND
from rca.baseline.rca_agent.prompt.basic_prompt_Market import cand as _MARKET_CAND
from rca.baseline.rca_agent.prompt.basic_prompt_Telecom import cand as _TELECOM_CAND
from rca.counterfactor_agent.agent import CounterfactualRCAAgent
from rca.execution.framework import Difficulty, ExecutionResult, TaskContext, TaskRuntime, TaskSpec, UnifiedTaskRunner
from rca.memory.evidence import build_component_candidates, build_reason_candidates, build_time_candidates, query_basil_evidence
from rca.time_candidates import estimate_occurrence_datetime_candidates


@dataclass(frozen=True)
class AgentStandardRuntimeExtras:
    light_agent_tasks: Set[int] = field(default_factory=set)


def infer_difficulty(task_id: int) -> Difficulty:
    t = int(task_id)
    if 1 <= t <= 3:
        return "easy"
    if 4 <= t <= 6:
        return "middle"
    return "hard"


def _required_fields(task_id: int) -> Tuple[bool, bool, bool]:
    t = int(task_id)
    need_time = t in (1, 4, 5, 7)
    need_reason = t in (2, 4, 6, 7)
    need_comp = t in (3, 5, 6, 7)
    return need_time, need_comp, need_reason


def _basic_prompt_for_dataset(dataset: str) -> str:
    d = str(dataset or "")
    if d.startswith("Market/") or d == "Market":
        return str(_MARKET_CAND)
    if d == "Bank":
        return str(_BANK_CAND)
    return str(_TELECOM_CAND)


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
    spec = TaskSpec(
        dataset=str(dataset),
        task_id=int(task_id),
        instruction=str(instruction),
        scoring_points=str(scoring_points),
        task_focus=str(task_focus),
        difficulty=infer_difficulty(int(task_id)),
    )
    rt = TaskRuntime(
        args=args,
        logger=logger,
        no_llm_fallback=bool(no_llm_fallback),
        extras={"agent_standard": extras} if extras is not None else {},
    )
    return TaskContext(spec=spec, runtime=rt)


class _AgentStandardRouter:
    def __init__(self, strategy: "_CounterfactualStrategy") -> None:
        self._strategy = strategy

    def route(self, ctx: TaskContext):
        return [self._strategy]


class _CounterfactualStrategy:
    name = "counterfactual_agent"

    def can_handle(self, ctx: TaskContext) -> bool:
        return True

    def run(self, ctx: TaskContext) -> Optional[ExecutionResult]:
        args = ctx.runtime.args
        logger = ctx.runtime.logger
        spec = ctx.spec

        t0 = time.perf_counter()
        timing: Dict[str, float] = {
            "total_execution_seconds": 0.0,
            "setup_seconds": 0.0,
            "candidate_generation_seconds": 0.0,
            "component_candidate_seconds": 0.0,
            "time_candidate_seconds": 0.0,
            "basil_evidence_build_seconds": 0.0,
            "basil_index_query_seconds": 0.0,
            "basil_index_total_seconds": 0.0,
            "agent_call_seconds": 0.0,
            "postprocess_seconds": 0.0,
        }
        basil_index_details: Dict[str, float] = {}

        setup_t0 = time.perf_counter()
        agent = CounterfactualRCAAgent(agent_prompt=str(_AGENT_RULES), basic_prompt=_basic_prompt_for_dataset(spec.dataset))
        timing["setup_seconds"] = float(time.perf_counter() - setup_t0)

        need_time, need_comp, need_reason = _required_fields(int(spec.task_id))

        candgen_t0 = time.perf_counter()
        basil_evidence = None
        basil_called = False
        if bool(getattr(args, "basil_index", False)):
            try:
                cfg = {k: v for k, v in dict(vars(args)).items() if str(k).startswith("basil_")}
            except Exception:
                cfg = {}
            try:
                basil_evidence = query_basil_evidence(
                    dataset=str(spec.dataset),
                    instruction=str(spec.instruction),
                    task_id=int(spec.task_id),
                    top_k=int(getattr(args, "task3_top_k", 12) or 12),
                    reasons=[],
                    center_time_s=None,
                    delta_s=1800,
                    config=cfg,
                )
                basil_called = True
                basil_index_details.update({f"basil.{k}": float(v) for k, v in (getattr(basil_evidence, "timings", {}) or {}).items()})
            except Exception:
                basil_evidence = None

        allowed_components: Optional[List[str]] = None
        allowed_reasons: Optional[List[str]] = None
        component_scores: Dict[str, float] = {}
        reason_scores: Dict[str, float] = {}
        component_first_times: Dict[str, str] = {}
        component_peak_times: Dict[str, str] = {}
        estimated_failure_time: str = ""

        if basil_evidence is not None:
            if need_comp:
                comp_t0 = time.perf_counter()
                comp_res = build_component_candidates(basil_evidence, top_k=int(getattr(args, "task3_top_k", 12) or 12))
                timing["component_candidate_seconds"] += float(time.perf_counter() - comp_t0)
                allowed_components = list(comp_res.candidates or [])
                component_scores = dict(comp_res.scores or {})
                component_first_times = dict(getattr(basil_evidence, "first_times", {}) or {})
                component_peak_times = dict(getattr(basil_evidence, "peak_times", {}) or {})
                estimated_failure_time = str(getattr(basil_evidence, "estimated_failure_time", "") or "")

            if need_reason:
                r_res = build_reason_candidates(basil_evidence, fallback_candidates=None, logger=None)
                allowed_reasons = list(r_res.candidates or [])
                reason_scores = dict(r_res.scores or {})

        time_candidates: List[Any] = []
        time_choice = ""
        if need_time:
            try:
                from rca.component_candidates import parse_time_window_generic

                _id, _start, _end = parse_time_window_generic(spec.instruction)
                time_choice = _start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_choice = ""

            time_t0 = time.perf_counter()
            try:
                focus_components: List[str] = []
                if allowed_components and need_comp:
                    focus_components = [str(allowed_components[0])]
                if basil_evidence is not None:
                    basil_tc = build_time_candidates(
                        basil_evidence,
                        top_k=int(getattr(args, "time_candidate_top_k", 8) or 8),
                        focus_components=focus_components,
                    )
                    time_candidates.extend(list(basil_tc.candidates or []))
                time_candidates.extend(
                    estimate_occurrence_datetime_candidates(
                        dataset=str(spec.dataset),
                        instruction=str(spec.instruction),
                        component_first_times=component_first_times,
                        component_peak_times=component_peak_times,
                        estimated_failure_time=estimated_failure_time,
                        component_scores=component_scores,
                        reasons=allowed_reasons or [],
                        top_k=int(getattr(args, "time_candidate_top_k", 8) or 8),
                    )
                )
            except Exception:
                pass
            timing["time_candidate_seconds"] += float(time.perf_counter() - time_t0)

        timing["candidate_generation_seconds"] = float(time.perf_counter() - candgen_t0)

        if basil_called:
            timing["basil_evidence_build_seconds"] = float(basil_index_details.get("basil.evidence_extract_segments_seconds", 0.0) or 0.0)
            timing["basil_index_query_seconds"] = float(basil_index_details.get("basil.evidence_index_query_seconds", 0.0) or 0.0)
            timing["basil_index_total_seconds"] = float(basil_index_details.get("basil.total_seconds", 0.0) or 0.0)

        context: Dict[str, Any] = {
            "need_time": bool(need_time),
            "need_comp": bool(need_comp),
            "need_reason": bool(need_reason),
            "time_choice": str(time_choice),
            "time_candidates": list(time_candidates),
            "component_first_times": dict(component_first_times),
            "component_peak_times": dict(component_peak_times),
            "estimated_failure_time": str(estimated_failure_time),
            "allowed_components": allowed_components,
            "allowed_reasons": allowed_reasons,
            "component_scores": dict(component_scores),
            "reason_scores": dict(reason_scores),
            "deterministic_only": bool(getattr(args, "deterministic_only", False)),
            "enable_uncertainty_llm": bool(getattr(args, "uncertainty_llm", False)),
            "enable_time_llm": bool(getattr(args, "time_llm", False)),
            "time_llm_top_k": int(getattr(args, "time_llm_top_k", 3) or 3),
            "evidence_score_gap_threshold": float(getattr(args, "evidence_score_gap_threshold", 0.15) or 0.15),
            "evidence_top1_score_floor": float(getattr(args, "evidence_top1_score_floor", 0.30) or 0.30),
            "evidence_top1_score_floor_single": float(getattr(args, "evidence_top1_score_floor_single", 0.50) or 0.50),
            "gate_skip_gap_threshold": float(getattr(args, "gate_skip_gap_threshold", 0.30) or 0.30),
            "basil_max_trace_rows": int(getattr(args, "basil_max_trace_rows", 0) or 0) or None,
        }

        agent_t0 = time.perf_counter()
        prediction, trajectory, prompt = agent.run(
            instruction=str(spec.instruction),
            logger=logger,
            dataset_name=str(spec.dataset),
            max_step=int(getattr(args, "controller_max_step", 25) or 25),
            max_turn=int(getattr(args, "controller_max_turn", 5) or 5),
            task_focus=str(spec.task_focus),
            context=context,
        )
        timing["agent_call_seconds"] = float(time.perf_counter() - agent_t0)

        timing["total_execution_seconds"] = float(time.perf_counter() - t0)
        if isinstance(trajectory, list):
            trajectory.append(
                {
                    "step": "timing",
                    "action": "timing",
                    "strategy": self.name,
                    "dataset": str(spec.dataset),
                    "task_id": int(spec.task_id),
                    "basil_enabled": bool(getattr(args, "basil_index", False)),
                    "timings_seconds": dict(timing),
                    "basil_index_details_seconds": dict(basil_index_details),
                    "timing_breakdown_seconds": dict(timing),
                }
            )

        return ExecutionResult(prediction=str(prediction), trajectory=list(trajectory or []), prompt=list(prompt or []))


def build_agent_standard_runner() -> UnifiedTaskRunner:
    strategy = _CounterfactualStrategy()
    return UnifiedTaskRunner(router=_AgentStandardRouter(strategy))

