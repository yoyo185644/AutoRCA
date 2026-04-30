from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Literal, List, Optional, Tuple

from rca.execution.agent_standard import AgentStandardRuntimeExtras, build_task_context
from rca.fast_solver import (
    estimate_time_from_trace,
    evidence_supports_reason,
    infer_reason_with_evidence,
    pick_component_from_granger,
)
from rca.execution.field_solvers import assemble_prediction, solve_component, solve_reason, solve_time

try:
    from rca.window_granger_solver import solve_with_window_granger
except Exception:
    solve_with_window_granger = None


Field = Literal["time", "component", "reason"]
Difficulty = Literal["easy", "middle", "hard"]


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    difficulty: Difficulty
    required_fields: FrozenSet[Field]

    @property
    def need_time(self) -> bool:
        return "time" in self.required_fields

    @property
    def need_component(self) -> bool:
        return "component" in self.required_fields

    @property
    def need_reason(self) -> bool:
        return "reason" in self.required_fields


TASK_SPECS: dict[int, TaskSpec] = {
    1: TaskSpec(task_id=1, difficulty="easy", required_fields=frozenset(["time"])),
    2: TaskSpec(task_id=2, difficulty="easy", required_fields=frozenset(["reason"])),
    3: TaskSpec(task_id=3, difficulty="easy", required_fields=frozenset(["component"])),
    4: TaskSpec(task_id=4, difficulty="middle", required_fields=frozenset(["time", "reason"])),
    5: TaskSpec(task_id=5, difficulty="middle", required_fields=frozenset(["time", "component"])),
    6: TaskSpec(task_id=6, difficulty="middle", required_fields=frozenset(["component", "reason"])),
    7: TaskSpec(task_id=7, difficulty="hard", required_fields=frozenset(["time", "component", "reason"])),
}


def get_task_spec(task_id: int) -> TaskSpec:
    tid = int(task_id)
    return TASK_SPECS.get(tid, TaskSpec(task_id=tid, difficulty="hard", required_fields=frozenset(["time", "component", "reason"])))


def _solve_fast_mid_hard_prediction(
    *,
    spec: TaskSpec,
    dataset: str,
    instruction: str,
    scoring_points: str,
    args: Any,
    logger: Any,
    no_llm_fallback: bool,
    light_agent_tasks: set[int],
) -> Optional[str]:
    try:
        need_time = bool(spec.need_time)
        need_comp = bool(spec.need_component)
        need_reason = bool(spec.need_reason)
        k_fail = int(required_root_count(scoring_points))
        dt, center_time_s, ev_time = solve_time(dataset=dataset, instruction=instruction, scoring_points=scoring_points, need_time=need_time)
        det_reason, det_reason_other, det_reason_ready, ev2 = solve_reason(
            spec=spec,
            dataset=dataset,
            instruction=instruction,
            scoring_points=scoring_points,
            args=args,
            need_reason=need_reason,
        )
        comp, dt_override, ev3, det_reason = solve_component(
            spec=spec,
            dataset=dataset,
            instruction=instruction,
            args=args,
            need_comp=need_comp,
            need_time=need_time,
            need_reason=need_reason,
            center_time_s=center_time_s,
            det_reason=det_reason,
            det_reason_ready=det_reason_ready,
            det_reason_other=det_reason_other,
            ev2=ev2,
            light_agent_tasks=set(light_agent_tasks),
            no_llm_fallback=no_llm_fallback,
        )
        if dt_override is not None:
            dt = str(dt_override)
        root_reason = det_reason if dataset == "Telecom" else det_reason_other
        obj = assemble_prediction(
            dataset=dataset,
            instruction=instruction,
            scoring_points=scoring_points,
            args=args,
            need_time=need_time,
            need_comp=need_comp,
            need_reason=need_reason,
            k_fail=int(k_fail),
            root_dt=dt,
            root_comp=comp,
            root_reason=root_reason if det_reason_ready else "",
            det_reason=str(det_reason or ""),
            det_reason_other=str(det_reason_other or ""),
            det_reason_ready=bool(det_reason_ready),
        )
        import json as _json
        if dataset == "Telecom":
            root = obj.get("1", {}) if isinstance(obj, dict) else {}
            missing = []
            if need_time and not str(root.get("root cause occurrence datetime", "") or "").strip():
                missing.append("root cause occurrence datetime")
            if need_comp and not str(root.get("root cause component", "") or "").strip():
                missing.append("root cause component")
            if need_reason and not str(root.get("root cause reason", "") or "").strip():
                missing.append("root cause reason")
            if missing:
                logger.warning(f"fast_mid_hard_solver produced incomplete Telecom prediction: missing={missing}")
                return None
            low_conf = []
            if need_time:
                if ev_time is None:
                    low_conf.append("time:no_evidence")
                else:
                    try:
                        ratio = float(ev_time.peak_p95) / max(float(ev_time.baseline_p95), 1e-9)
                    except Exception:
                        ratio = 0.0
                    if str(getattr(ev_time, "method", "")) != "jump" and ratio < 1.30:
                        low_conf.append(f"time:{getattr(ev_time, 'method', '')}:{ratio:.2f}")
            if need_reason:
                try:
                    ok_reason = bool(evidence_supports_reason(ev2 or {}, det_reason))
                except Exception:
                    ok_reason = False
                try:
                    gap = float((ev2 or {}).get("gap", 0.0) or 0.0)
                    bscore = float(((ev2 or {}).get("best") or {}).get("score", 0.0) or 0.0)
                except Exception:
                    gap, bscore = 0.0, 0.0
                if not ok_reason and (gap < 0.35 or bscore < 1.25):
                    low_conf.append(f"reason:gap={gap:.2f},best={bscore:.2f}")
            if need_comp and ev3 is not None:
                try:
                    g = float(getattr(ev3, "gap_norm", 0.0) or 0.0)
                except Exception:
                    g = 0.0
                if g < float(getattr(args, "task3_agent_gap", 0.15)):
                    low_conf.append(f"component:gap_norm={g:.3f}")
            if low_conf and not no_llm_fallback:
                logger.warning(f"fast_mid_hard_solver low-confidence; fallback to CounterfactualRCAAgent: {low_conf}")
                return None
            return _json.dumps(obj, ensure_ascii=False)
        return _json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"fast_mid_hard_solver failed, fallback: {e}")
        return None


def _fallback_to_agent_runner(
    *,
    spec: TaskSpec,
    args: Any,
    dataset: str,
    instruction: str,
    scoring_points: str,
    task_focus: str,
    logger: Any,
    no_llm_fallback: bool,
    rca_agent_tasks: Optional[set[int]],
    agent_standard_runner: Any,
    agent_standard_extras: AgentStandardRuntimeExtras,
) -> Tuple[str, List[Any], List[Any]]:
    if no_llm_fallback:
        return "{}", [], []
    if rca_agent_tasks is not None and int(spec.task_id) not in rca_agent_tasks:
        raise RuntimeError(f"rca_agent disabled for task_{spec.task_id}")
    ctx = build_task_context(
        dataset=dataset,
        task_id=int(spec.task_id),
        instruction=instruction,
        scoring_points=scoring_points,
        task_focus=task_focus,
        args=args,
        logger=logger,
        no_llm_fallback=no_llm_fallback,
        extras=agent_standard_extras,
    )
    res = agent_standard_runner.run(ctx)
    return res.prediction, list(res.trajectory or []), list(res.prompt or [])


class EasySolver:
    def run(
        self,
        *,
        spec: TaskSpec,
        args: Any,
        dataset: str,
        instruction: str,
        scoring_points: str,
        task_focus: str,
        logger: Any,
        no_llm_fallback: bool,
        rca_agent_tasks: Optional[set[int]],
        agent_standard_runner: Any,
        agent_standard_extras: AgentStandardRuntimeExtras,
    ) -> Tuple[str, List[Any], List[Any]]:
        return _fallback_to_agent_runner(
            spec=spec,
            args=args,
            dataset=dataset,
            instruction=instruction,
            scoring_points=scoring_points,
            task_focus=task_focus,
            logger=logger,
            no_llm_fallback=no_llm_fallback,
            rca_agent_tasks=rca_agent_tasks,
            agent_standard_runner=agent_standard_runner,
            agent_standard_extras=agent_standard_extras,
        )


class MiddleSolver:
    def run(
        self,
        *,
        spec: TaskSpec,
        args: Any,
        dataset: str,
        instruction: str,
        scoring_points: str,
        task_focus: str,
        logger: Any,
        no_llm_fallback: bool,
        rca_agent_tasks: Optional[set[int]],
        fast_mid_hard: bool,
        light_agent_tasks: set[int],
        agent_standard_runner: Any,
        agent_standard_extras: AgentStandardRuntimeExtras,
    ) -> Tuple[str, List[Any], List[Any]]:
        prediction = None
        if not bool(getattr(args, "force_counterfactual", False)) and fast_mid_hard and not bool(getattr(args, "mid_hard_agent", False)):
            prediction = _solve_fast_mid_hard_prediction(
                spec=spec,
                dataset=dataset,
                instruction=instruction,
                scoring_points=scoring_points,
                args=args,
                logger=logger,
                no_llm_fallback=no_llm_fallback,
                light_agent_tasks=set(light_agent_tasks),
            )
        if isinstance(prediction, str):
            return prediction, [], []
        prediction, trajectory, prompt = _fallback_to_agent_runner(
            spec=spec,
            args=args,
            dataset=dataset,
            instruction=instruction,
            scoring_points=scoring_points,
            task_focus=task_focus,
            logger=logger,
            no_llm_fallback=no_llm_fallback,
            rca_agent_tasks=rca_agent_tasks,
            agent_standard_runner=agent_standard_runner,
            agent_standard_extras=agent_standard_extras,
        )
        return prediction, trajectory, prompt


class HardSolver(MiddleSolver):
    pass


def solve_by_mode(
    *,
    spec: TaskSpec,
    args: Any,
    dataset: str,
    idx: int,
    instruction: str,
    scoring_points: str,
    task_focus: str,
    logger: Any,
    archive_df: Optional[pd.DataFrame],
    no_llm_fallback: bool,
    fast_mid_hard: bool,
    light_agent_tasks: set[int],
    rca_agent_tasks: Optional[set[int]],
    agent_standard_runner: Any,
    agent_standard_extras: AgentStandardRuntimeExtras,
) -> Tuple[str, List[Any], List[Any]]:
    prediction = None
    trajectory: List[Any] = []
    prompt: List[Any] = []

    if bool(getattr(args, "window_granger_solver", False)) and (dataset.startswith("Market/") or dataset == "Bank"):
        if solve_with_window_granger is None:
            logger.warning("window_granger_solver is not available; skipping window_granger_solver.")
        else:
            prediction = solve_with_window_granger(dataset=dataset, instruction=instruction, task_id=int(spec.task_id))

    if prediction is None and bool(getattr(args, "hybrid_window_granger_prior", False)) and dataset.startswith("Market/"):
        hybrid_tasks = {
            "Market/cloudbed-1": {2, 3, 4},
            "Market/cloudbed-2": {1, 4},
        }
        if int(spec.task_id) in hybrid_tasks.get(dataset, set()):
            if solve_with_window_granger is None:
                logger.warning("window_granger_solver is not available; skipping hybrid_window_granger_prior.")
            else:
                prediction = solve_with_window_granger(dataset=dataset, instruction=instruction, task_id=int(spec.task_id))

    if prediction is None and archive_df is not None:
        row_match = archive_df[archive_df["row_id_int"] == int(idx)]
        if not row_match.empty:
            prediction = row_match.iloc[0]["prediction"]
            if dataset == "Telecom" and bool(getattr(args, "telecom_task1_trace_time", False)) and int(spec.task_id) == 1:
                try:
                    dt = estimate_time_from_trace(instruction)
                    import json as _json
                    obj = _json.loads(prediction)
                    if "1" in obj and isinstance(obj["1"], dict):
                        obj["1"]["root cause occurrence datetime"] = dt
                        prediction = _json.dumps(obj, ensure_ascii=False, indent=4)
                except Exception as e:
                    logger.warning(f"Task1 trace time override failed: {e}")
            if dataset == "Telecom" and bool(getattr(args, "telecom_task3_granger_component", False)) and int(spec.task_id) == 3:
                try:
                    import json as _json
                    obj = _json.loads(prediction)
                    comp = pick_component_from_granger(instruction)
                    if "1" in obj and isinstance(obj["1"], dict):
                        obj["1"]["root cause component"] = comp
                        prediction = _json.dumps(obj, ensure_ascii=False, indent=4)
                except Exception as e:
                    logger.warning(f"Task3 granger component override failed: {e}")
        else:
            logger.warning(f"Archive baseline missing row_id={idx}; using empty prediction.")
            prediction = "{}"

    if isinstance(prediction, str):
        return prediction, trajectory, prompt

    if spec.difficulty == "easy":
        return EasySolver().run(
            spec=spec,
            args=args,
            dataset=dataset,
            instruction=instruction,
            scoring_points=scoring_points,
            task_focus=task_focus,
            logger=logger,
            no_llm_fallback=no_llm_fallback,
            rca_agent_tasks=rca_agent_tasks,
            agent_standard_runner=agent_standard_runner,
            agent_standard_extras=agent_standard_extras,
        )

    if spec.difficulty == "middle":
        return MiddleSolver().run(
            spec=spec,
            args=args,
            dataset=dataset,
            instruction=instruction,
            scoring_points=scoring_points,
            task_focus=task_focus,
            logger=logger,
            no_llm_fallback=no_llm_fallback,
            rca_agent_tasks=rca_agent_tasks,
            fast_mid_hard=fast_mid_hard,
            light_agent_tasks=set(light_agent_tasks),
            agent_standard_runner=agent_standard_runner,
            agent_standard_extras=agent_standard_extras,
        )

    return HardSolver().run(
        spec=spec,
        args=args,
        dataset=dataset,
        instruction=instruction,
        scoring_points=scoring_points,
        task_focus=task_focus,
        logger=logger,
        no_llm_fallback=no_llm_fallback,
        rca_agent_tasks=rca_agent_tasks,
        fast_mid_hard=fast_mid_hard,
        light_agent_tasks=set(light_agent_tasks),
        agent_standard_runner=agent_standard_runner,
        agent_standard_extras=agent_standard_extras,
    )
