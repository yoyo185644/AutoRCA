# BASIL Dataset Skill Refactor

## Goal

The refactor separates dataset semantics from the BASIL RCA algorithm.

Dataset-specific code should describe how to read and interpret a dataset.
BASIL core code should decide how to build evidence, recall candidates, rank
root causes, select time candidates, and invoke bounded LLM reranking.

## Layers

```text
Dataset Skill
  declares schema, evidence sources, component types, metric ownership,
  reason taxonomy, and parser/canonicalization hooks

Core Planner
  maps Dataset Skill capabilities to enabled BASIL core modules

BASIL Core Modules
  execute evidence extraction, index query, candidate recall, joint ranking,
  uncertainty gating, and bounded LLM reranking
```

## Dataset Skill Boundary

Dataset Skill may define:

- telemetry schema and timezone
- evidence source availability: metric / trace / log / topology
- component canonicalization
- component semantic type
- metric ownership vocabulary
- reason taxonomy and reason keywords
- trace/log parser behavior

Dataset Skill should not implement:

- candidate ranking policy
- rootness ranking policy
- time selection policy
- LLM gate policy
- LLM adoption policy

## Core Module Plan

`CorePlanner` currently selects these modules:

- `typed_channel_recall`
- `metric_reason_evidence`
- `log_reason_evidence`
- `causal_topology`
- `causal_time_selector`
- `bounded_llm_rerank`
- `trace_evidence`

The plan is available at runtime in `context["basil_core_modules"]`; the
dataset declaration is available in `context["dataset_skill"]`.

## Migration Strategy

This commit is the safe structural phase:

1. Add `DatasetSkill`.
2. Add `CorePlanner`.
3. Attach both to every `DatasetAdapter`.
4. Expose the plan in the indexed evidence agent context.
5. Keep existing adapter ranking behavior intact.

Next phases can move one algorithm at a time from adapters into BASIL core:

1. typed-channel recall
2. reason evidence scorer
3. causal change-point time selector
4. topology-aware rootness ranker
5. bounded LLM gate/adoption guard

Each migration should be validated with Telecom, Bank, Market/cloudbed-1, and
Market/cloudbed-2 before removing adapter-specific code.
