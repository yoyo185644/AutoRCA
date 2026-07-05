from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from rca.skill.dataset_skill import DatasetSkill


@dataclass(frozen=True)
class CoreModulePlan:
    """BASIL core modules selected for a dataset/query."""

    typed_channel_recall: bool
    metric_reason_evidence: bool
    log_reason_evidence: bool
    causal_topology: bool
    causal_time_selector: bool
    bounded_llm_rerank: bool
    trace_evidence: bool

    def to_dict(self) -> Dict[str, bool]:
        return {
            "typed_channel_recall": bool(self.typed_channel_recall),
            "metric_reason_evidence": bool(self.metric_reason_evidence),
            "log_reason_evidence": bool(self.log_reason_evidence),
            "causal_topology": bool(self.causal_topology),
            "causal_time_selector": bool(self.causal_time_selector),
            "bounded_llm_rerank": bool(self.bounded_llm_rerank),
            "trace_evidence": bool(self.trace_evidence),
        }


class CorePlanner:
    """Select BASIL core modules from a DatasetSkill.

    The planner keeps algorithm selection centralized.  Dataset adapters should
    expose dataset semantics and capability declarations; they should not decide
    independent BASIL pipelines per dataset.
    """

    def __init__(self, skill: DatasetSkill) -> None:
        self.skill = skill

    @staticmethod
    def _resolve(value: bool | None, default: bool) -> bool:
        return bool(default if value is None else value)

    def plan_for_task(
        self,
        *,
        task_id: int,
        need_time: bool,
        need_comp: bool,
        need_reason: bool,
        context: Mapping[str, Any] | None = None,
    ) -> CoreModulePlan:
        sources = self.skill.evidence_sources
        modules = self.skill.modules
        needs_any_evidence = bool(need_time or need_comp or need_reason)

        metric_reason = bool(sources.metric and need_reason)
        log_reason = bool(sources.log and need_reason)
        topology = bool(sources.topology and sources.trace and (need_comp or need_time))
        causal_time = bool(need_time and (sources.metric or sources.trace or sources.log))

        return CoreModulePlan(
            typed_channel_recall=self._resolve(
                modules.typed_channel_recall,
                bool(sources.metric and (need_comp or need_reason)),
            ),
            metric_reason_evidence=self._resolve(
                modules.metric_reason_evidence,
                metric_reason,
            ),
            log_reason_evidence=self._resolve(
                modules.log_reason_evidence,
                log_reason,
            ),
            causal_topology=self._resolve(
                modules.causal_topology,
                topology,
            ),
            causal_time_selector=self._resolve(
                modules.causal_time_selector,
                causal_time,
            ),
            bounded_llm_rerank=self._resolve(
                modules.bounded_llm_rerank,
                needs_any_evidence,
            ),
            trace_evidence=self._resolve(
                modules.trace_evidence,
                bool(sources.trace),
            ),
        )
