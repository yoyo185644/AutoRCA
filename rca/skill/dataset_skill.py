from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class EvidenceSources:
    """Evidence sources exposed by a dataset.

    This is a capability declaration, not an algorithm decision.  The core
    planner consumes it to decide which BASIL modules can run.
    """

    metric: bool = True
    trace: bool = True
    log: bool = False
    topology: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "EvidenceSources":
        row = dict(data or {})
        return cls(
            metric=bool(row.get("metric", True)),
            trace=bool(row.get("trace", True)),
            log=bool(row.get("log", False)),
            topology=bool(row.get("topology", row.get("trace", True))),
        )

    def to_dict(self) -> Dict[str, bool]:
        return {
            "metric": bool(self.metric),
            "trace": bool(self.trace),
            "log": bool(self.log),
            "topology": bool(self.topology),
        }


@dataclass(frozen=True)
class SkillModules:
    """Optional module declarations from the dataset skill.

    A value of ``None`` means "let CorePlanner infer it from evidence_sources".
    """

    typed_channel_recall: bool | None = None
    metric_reason_evidence: bool | None = None
    log_reason_evidence: bool | None = None
    causal_topology: bool | None = None
    causal_time_selector: bool | None = None
    bounded_llm_rerank: bool | None = None
    trace_evidence: bool | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "SkillModules":
        row = dict(data or {})

        def opt_bool(key: str) -> bool | None:
            return None if key not in row else bool(row.get(key))

        return cls(
            typed_channel_recall=opt_bool("typed_channel_recall"),
            metric_reason_evidence=opt_bool("metric_reason_evidence"),
            log_reason_evidence=opt_bool("log_reason_evidence"),
            causal_topology=opt_bool("causal_topology"),
            causal_time_selector=opt_bool("causal_time_selector"),
            bounded_llm_rerank=opt_bool("bounded_llm_rerank"),
            trace_evidence=opt_bool("trace_evidence"),
        )


@dataclass(frozen=True)
class DatasetSkill:
    """Declarative dataset semantics for BASIL.

    The skill is allowed to describe schema, taxonomy, component types, metric
    ownership and evidence availability.  It should not implement candidate
    ranking, time selection, or LLM adoption logic.
    """

    dataset: str
    timezone: str = "Asia/Shanghai"
    adapter: str = ""
    evidence_sources: EvidenceSources = field(default_factory=EvidenceSources)
    modules: SkillModules = field(default_factory=SkillModules)
    component_types: Dict[str, Any] = field(default_factory=dict)
    metric_ownership: Dict[str, Any] = field(default_factory=dict)
    reason_taxonomy: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, dataset: str, config: Mapping[str, Any] | None) -> "DatasetSkill":
        cfg = dict(config or {})
        skill_cfg = dict(cfg.get("skill", {}) or {})
        telemetry_cfg = dict(cfg.get("telemetry", {}) or {})

        evidence_cfg = dict(skill_cfg.get("evidence_sources", {}) or {})
        if "metric" not in evidence_cfg:
            evidence_cfg["metric"] = True
        if "trace" not in evidence_cfg:
            evidence_cfg["trace"] = bool(telemetry_cfg.get("trace_file", True))
        if "log" not in evidence_cfg:
            log_files = telemetry_cfg.get("log_files", {})
            evidence_cfg["log"] = bool(log_files)
        if "topology" not in evidence_cfg:
            evidence_cfg["topology"] = bool(evidence_cfg.get("trace", True))

        return cls(
            dataset=str(cfg.get("dataset", dataset) or dataset),
            timezone=str(cfg.get("timezone", "Asia/Shanghai") or "Asia/Shanghai"),
            adapter=str(cfg.get("adapter", "") or ""),
            evidence_sources=EvidenceSources.from_mapping(evidence_cfg),
            modules=SkillModules.from_mapping(skill_cfg.get("modules", {})),
            component_types=dict(skill_cfg.get("component_types", {}) or {}),
            metric_ownership=dict(skill_cfg.get("metric_ownership", {}) or {}),
            reason_taxonomy=dict(cfg.get("reason_taxonomy", {}) or skill_cfg.get("reason_taxonomy", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "timezone": self.timezone,
            "adapter": self.adapter,
            "evidence_sources": self.evidence_sources.to_dict(),
            "component_types": dict(self.component_types),
            "metric_ownership": dict(self.metric_ownership),
            "reason_taxonomy_size": len(self.reason_taxonomy),
        }
