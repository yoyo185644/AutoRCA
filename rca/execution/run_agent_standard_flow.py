from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

from rca.agent.agent_standard import AgentStandardRuntimeExtras, build_task_context, infer_difficulty
from rca.execution.framework import Difficulty, UnifiedTaskRunner


@dataclass(frozen=True)
class MinimalTaskSpec:
    task_id: int
    difficulty: Difficulty


def get_task_spec(task_id: int) -> MinimalTaskSpec:
    return MinimalTaskSpec(task_id=int(task_id), difficulty=infer_difficulty(int(task_id)))


def solve_by_mode(
    *,
    spec: Any,
    args: Any,
    dataset: str,
    idx: int,
    instruction: str,
    task_focus: str,
    logger: Any,
    archive_df: Any = None,
    no_llm_fallback: bool = False,
    fast_mid_hard: bool = False,
    light_agent_tasks: Optional[Set[int]] = None,
    rca_agent_tasks: Any = None,
    agent_standard_runner: Optional[UnifiedTaskRunner] = None,
    agent_standard_extras: Optional[AgentStandardRuntimeExtras] = None,
) -> Tuple[str, List[Any], List[Any]]:
    task_id = int(getattr(spec, "task_id", 0) or 0)
    if task_id <= 0:
        try:
            task_id = int(getattr(spec, "id", 0) or 0)
        except Exception:
            task_id = 0
    if task_id <= 0:
        task_id = 0

    extras = agent_standard_extras
    if extras is None and light_agent_tasks is not None:
        extras = AgentStandardRuntimeExtras(light_agent_tasks=set(light_agent_tasks))

    if agent_standard_runner is None:
        raise ValueError("agent_standard_runner is required")

    ctx = build_task_context(
        dataset=str(dataset),
        task_id=int(task_id),
        instruction=str(instruction),
        task_focus=str(task_focus),
        args=args,
        logger=logger,
        no_llm_fallback=bool(no_llm_fallback),
        extras=extras,
    )
    res = agent_standard_runner.run(ctx)
    return str(res.prediction), list(res.trajectory or []), list(res.prompt or [])
