from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

from rca.common.text import service_base


@dataclass
class BasilCoreOutput:
    """Candidate-pool output from the unified BASIL core."""

    deterministic_pool: List[Dict[str, Any]]
    deterministic: List[Dict[str, Any]]
    llm_prompt_pool: List[Dict[str, Any]]
    llm_only_extras: List[Dict[str, Any]] = field(default_factory=list)
    pool_diagnostics: Dict[str, Any] = field(default_factory=dict)
    coverage_plan: Dict[str, Any] = field(default_factory=dict)
    intervention_plan: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


class BasilCorePipeline:
    """Unified candidate generation/ranking pipeline.

    Dataset adapters provide semantic hooks only.  This pipeline owns the
    shared algorithmic flow:

    1. initial index top-k retrieval
    2. typed-channel supplement
    3. task-aware dimension completion
    4. candidate expansion
    5. deterministic rerank
    6. prompt-pool challenger expansion
    """

    def __init__(
        self,
        *,
        adapter: Any,
        need_time: bool,
        need_comp: bool,
        need_reason: bool,
        task_id: int,
        k: int,
        args: Any,
        result: Any,
        context: Dict[str, Any],
        deterministic_select: Callable[..., List[Dict[str, Any]]],
        root_order: Callable[[Sequence[Mapping[str, Any]]], List[Dict[str, Any]]],
        adapter_rerank: Callable[..., List[Dict[str, Any]]],
        attach_topology: Callable[..., List[Dict[str, Any]]],
        llm_extra: Callable[..., List[Dict[str, Any]]],
        diagnostics: Callable[..., Dict[str, Any]],
        coverage: Callable[..., Dict[str, Any]],
        intervention: Callable[..., Dict[str, Any]],
    ) -> None:
        self.adapter = adapter
        self.need_time = bool(need_time)
        self.need_comp = bool(need_comp)
        self.need_reason = bool(need_reason)
        self.task_id = int(task_id)
        self.k = max(1, int(k))
        self.args = args
        self.result = result
        self.context = dict(context or {})
        self.deterministic_select = deterministic_select
        self.root_order = root_order
        self.adapter_rerank = adapter_rerank
        self.attach_topology = attach_topology
        self.llm_extra = llm_extra
        self.diagnostics = diagnostics
        self.coverage = coverage
        self.intervention = intervention

    def build(self) -> BasilCoreOutput:
        pool_k = self._pool_k()
        pool = self.deterministic_select(result=self.result, k=int(pool_k))
        pool = [dict(x) for x in list(pool or []) if isinstance(x, Mapping)]

        pool = self._merge_time_dimension(pool, pool_k=pool_k)
        pool = self._replace_reason_surface_if_needed(pool, pool_k=pool_k)
        pool = self._expand_candidates(pool, pool_k=pool_k)
        pool = self.adapter_rerank(
            result=self.result,
            task_id=int(self.task_id),
            context=self.context,
            selected=pool,
            k=int(pool_k),
        )
        pool = [dict(x) for x in list(pool or []) if isinstance(x, Mapping)]
        if self.need_time and not self.need_comp and not self.need_reason:
            pool = self._sort_time_only(pool)

        self.context["candidate_tuples"] = [dict(x) for x in pool[:50] if isinstance(x, Mapping)]
        pool = self.attach_topology(pool, adapter=self.adapter, context=self.context)

        extras = self.llm_extra(
            pool,
            self.context,
            need_time=self.need_time,
            need_comp=self.need_comp,
            need_reason=self.need_reason,
            max_extra=max(0, int((self.context or {}).get("llm_max_prompt_candidates", 6) or 6) * 3),
        )
        prompt_pool = list(pool or []) + list(extras or [])
        if extras:
            prompt_pool = self.attach_topology(prompt_pool, adapter=self.adapter, context=self.context)
        self.context["candidate_tuples"] = [dict(x) for x in list(prompt_pool or [])[:80] if isinstance(x, Mapping)]
        self.context["llm_only_challenger_count"] = int(len(extras or []))
        self.context["candidate_tuples_ranked"] = True

        deterministic = list(pool or [])[: self.k]
        if self.need_comp and not self.need_reason and not self.need_time:
            deterministic = self.root_order(deterministic)[: self.k]

        pool_diag = self.diagnostics(
            prompt_pool,
            context=self.context,
            need_time=self.need_time,
            need_comp=self.need_comp,
            need_reason=self.need_reason,
        )
        coverage_plan = self.coverage(
            prompt_pool,
            self.context,
            need_time=self.need_time,
            need_comp=self.need_comp,
            need_reason=self.need_reason,
            max_keys=int((self.context or {}).get("llm_max_prompt_candidates", 6) or 6),
        )
        pool_diag["coverage_aware_agent"] = dict(coverage_plan.get("coverage") or {})
        self.context["coverage_aware_agent"] = dict(coverage_plan.get("coverage") or {})

        intervention_plan = self.intervention(
            prompt_pool,
            self.context,
            need_time=self.need_time,
            need_comp=self.need_comp,
            need_reason=self.need_reason,
            max_keys=int((self.context or {}).get("llm_max_prompt_candidates", 6) or 6),
        )
        pool_diag["intervention_agent"] = {
            "call_llm": bool(intervention_plan.get("call_llm")),
            "reason": str(intervention_plan.get("reason", "") or ""),
            "profiles": list(intervention_plan.get("profiles") or [])[:6],
        }
        self.context["intervention_agent"] = dict(pool_diag["intervention_agent"])
        return BasilCoreOutput(
            deterministic_pool=pool,
            deterministic=deterministic,
            llm_prompt_pool=[dict(x) for x in list(prompt_pool or []) if isinstance(x, Mapping)],
            llm_only_extras=[dict(x) for x in list(extras or []) if isinstance(x, Mapping)],
            pool_diagnostics=pool_diag,
            coverage_plan=dict(coverage_plan or {}),
            intervention_plan=dict(intervention_plan or {}),
            context=dict(self.context or {}),
        )

    def _pool_k(self) -> int:
        pool_k = max(self.k, 5)
        if self.adapter is None:
            return int(pool_k)
        try:
            return int(
                self.adapter.index_candidate_pool_k(
                    task_id=int(self.task_id),
                    k=int(self.k),
                    default_pool_k=int(pool_k),
                    context=dict(self.context or {}),
                )
            )
        except Exception:
            return int(pool_k)

    def _merge_time_dimension(self, pool: List[Dict[str, Any]], *, pool_k: int) -> List[Dict[str, Any]]:
        if not self.need_time:
            return pool
        time_pool = [dict(x) for x in list((self.context or {}).get("time_candidates") or []) if isinstance(x, Mapping)]
        existing_dt = {str(x.get("datetime", "") or "") for x in pool if isinstance(x, Mapping)}
        merged: List[Dict[str, Any]] = []
        for item in time_pool:
            dt = str(item.get("datetime", "") or "").strip()
            if not dt or dt in existing_dt:
                continue
            comp = str(item.get("component", "") or "").strip()
            reason = str(item.get("reason", "") or "").strip()
            if self.need_comp and not comp:
                continue
            if self.need_reason and self.need_comp and not reason:
                continue
            merged.append(
                {
                    "component": comp,
                    "reason": reason,
                    "datetime": dt,
                    "score": float(item.get("score", 0.0) or 0.0),
                    "source": str(item.get("source", "indexed_time_candidate") or "indexed_time_candidate"),
                    "evidence": dict(item.get("evidence", {}) or {}) if isinstance(item.get("evidence", {}), Mapping) else {},
                }
            )
            existing_dt.add(dt)
            if len(merged) >= int(pool_k):
                break

        if (not self.need_comp and not self.need_reason) or len({str(x.get("datetime", "") or "") for x in pool}) <= 1:
            return sorted(list(pool or []) + merged, key=self._time_score, reverse=True)[: max(int(pool_k), len(merged))]
        return pool

    def _replace_reason_surface_if_needed(self, pool: List[Dict[str, Any]], *, pool_k: int) -> List[Dict[str, Any]]:
        if not self.need_reason or self.need_comp:
            return pool
        reason_scores = dict((self.context or {}).get("reason_scores") or {})
        allowed_reasons = [str(x).strip() for x in list((self.context or {}).get("allowed_reasons") or []) if str(x).strip()]
        if not reason_scores and not allowed_reasons:
            return pool
        default_dt = str((self.context or {}).get("time_choice", "") or "").strip()
        if self.need_time and not default_dt:
            for item in list((self.context or {}).get("time_candidates") or []):
                if isinstance(item, Mapping) and str(item.get("datetime", "") or "").strip():
                    default_dt = str(item.get("datetime", "") or "").strip()
                    break
        max_reason_score = max([abs(float(v or 0.0)) for v in reason_scores.values()] + [1.0])
        ordered = list(dict.fromkeys(list(allowed_reasons) + list(reason_scores.keys())))
        out: List[Dict[str, Any]] = []
        for rank, reason in enumerate(ordered, start=1):
            score = float(reason_scores.get(reason, 0.0) or 0.0) / max(max_reason_score, 1e-9)
            if reason not in reason_scores:
                score = max(score, 0.05 / float(rank))
            out.append(
                {
                    "component": "",
                    "reason": reason,
                    "datetime": default_dt if self.need_time else "",
                    "score": float(score),
                    "source": "indexed_reason_surface",
                }
            )
        return sorted(out, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)[: max(int(pool_k), len(out))]

    def _expand_candidates(self, pool: List[Dict[str, Any]], *, pool_k: int) -> List[Dict[str, Any]]:
        if self.adapter is None:
            return pool
        try:
            expanded = self.adapter.expand_index_candidate_tuples(
                selected=list(pool),
                context=dict(self.context or {}),
                task_id=int(self.task_id),
                k=int(pool_k),
            )
            if isinstance(expanded, list) and expanded:
                self.context["candidate_tuples"] = [dict(x) for x in expanded[:50] if isinstance(x, Mapping)]
                return [dict(x) for x in expanded if isinstance(x, Mapping)]
        except Exception:
            pass
        return pool

    def _sort_time_only(self, pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted([dict(x) for x in list(pool or []) if isinstance(x, Mapping)], key=self._time_score, reverse=True)

    def _time_score(self, item: Mapping[str, Any]) -> float:
        if self.adapter is not None:
            try:
                return float(self.adapter._score_time_candidate(dict(item), "", ""))  # noqa: SLF001
            except Exception:
                pass
        try:
            return float(item.get("score", 0.0) or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def component_family(component: str) -> str:
        comp = str(component or "").strip().lower()
        if not comp:
            return ""
        if comp.startswith("db_"):
            return "db"
        if comp.startswith(("docker_", "container")):
            return "container"
        if comp.startswith(("os_", "node-")):
            return "node"
        if comp.startswith("redis"):
            return "cache"
        return service_base(comp) or comp

    @staticmethod
    def time_cluster(dt: str) -> str:
        text = str(dt or "").strip()
        if not text:
            return ""
        match = re.search(r"(\d{2}):(\d{2})(?::(\d{2}))?", text)
        if not match:
            return text
        hour = int(match.group(1))
        minute = int(match.group(2))
        return f"{hour:02d}:{(minute // 5) * 5:02d}"
