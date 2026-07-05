"""Dataset skill abstraction for BASIL.

The skill layer describes dataset semantics and available evidence sources.
It is intentionally separate from BASIL's RCA algorithms: core modules decide
how evidence is indexed, recalled, ranked, and optionally reranked by an LLM.
"""

from rca.skill.core_planner import CoreModulePlan, CorePlanner
from rca.skill.dataset_skill import DatasetSkill, EvidenceSources, SkillModules

__all__ = [
    "CoreModulePlan",
    "CorePlanner",
    "DatasetSkill",
    "EvidenceSources",
    "SkillModules",
]
