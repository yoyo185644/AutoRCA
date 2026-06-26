from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol


Difficulty = Literal["easy", "middle", "hard"]


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    task_id: int
    instruction: str
    task_focus: str
    difficulty: Difficulty


@dataclass
class TaskRuntime:
    args: Any
    logger: Any
    no_llm_fallback: bool
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    prediction: str
    trajectory: List[Any] = field(default_factory=list)
    prompt: List[Any] = field(default_factory=list)


@dataclass
class TaskContext:
    spec: TaskSpec
    runtime: TaskRuntime


class Strategy(Protocol):
    name: str

    def can_handle(self, ctx: TaskContext) -> bool: ...

    def run(self, ctx: TaskContext) -> Optional[ExecutionResult]: ...


class Router(Protocol):
    def route(self, ctx: TaskContext) -> List[Strategy]: ...


class UnifiedTaskRunner:
    def __init__(self, router: Router) -> None:
        self._router = router

    def run(self, ctx: TaskContext) -> ExecutionResult:
        for strategy in self._router.route(ctx):
            if not strategy.can_handle(ctx):
                continue
            res = strategy.run(ctx)
            if res is not None and isinstance(res.prediction, str):
                return res
        return ExecutionResult(prediction="{}")
