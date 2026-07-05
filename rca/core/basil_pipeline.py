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
        self._ensure_query_window_context()
        pool = self.deterministic_select(result=self.result, k=int(pool_k))
        pool = [dict(x) for x in list(pool or []) if isinstance(x, Mapping)]

        pool = self._pre_supplement_candidates(pool)
        pool = self._merge_time_dimension(pool, pool_k=pool_k)
        pool = self._bind_reason_time_candidates(pool, pool_k=pool_k)
        pool = self._expand_component_reason_cross_product(pool, pool_k=pool_k)
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

    def _pre_supplement_candidates(self, pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expose typed-channel supplements before task-dimension binding.

        Some adapters can cheaply recover component-local telemetry candidates
        from the current incident.  Running that hook before reason/time binding
        lets the shared core form evidence-backed tuples from those candidates
        instead of binding only the initial top-k retrieval surface.
        """
        if self.adapter is None or not (self.need_comp or self.need_reason):
            return pool
        try:
            date_dir = str((self.context or {}).get("date_dir", "") or getattr(self.result, "date_dir", "") or "")
            supplement = self.adapter.supplement_candidate_pool(
                list(pool or []),
                dict(self.context or {}),
                date_dir=date_dir,
            )
        except Exception:
            supplement = []
        if not supplement:
            return pool
        merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for item in list(pool or []) + [dict(x) for x in list(supplement or []) if isinstance(x, Mapping)]:
            if not isinstance(item, Mapping):
                continue
            key = (
                str(item.get("component", "") or ""),
                str(item.get("reason", "") or ""),
                str(item.get("datetime", "") or ""),
            )
            prev = merged.get(key)
            if prev is None or float(prev.get("score", 0.0) or 0.0) < float(item.get("score", 0.0) or 0.0):
                merged[key] = dict(item)
        rows = list(merged.values())
        rows.sort(key=self._evidence_backed_score, reverse=True)
        return rows

    def _ensure_query_window_context(self) -> None:
        if (
            str((self.context or {}).get("date_dir", "") or "").strip()
            and str((self.context or {}).get("query_start_local", "") or "").strip()
            and str((self.context or {}).get("query_end_local", "") or "").strip()
        ):
            return
        instruction = str((self.context or {}).get("instruction", "") or "")
        if not instruction:
            return
        try:
            from rca.adapter.utils.component_candidates import parse_time_window_generic
            from rca.common.io import resolve_date_dir

            raw_date, start, end = parse_time_window_generic(instruction)
            dataset = str((self.context or {}).get("dataset", "") or getattr(self.result, "dataset", "") or "")
            date_dir = str(resolve_date_dir(dataset, raw_date)) if dataset and raw_date else str(raw_date or "")
            if date_dir and not str((self.context or {}).get("date_dir", "") or "").strip():
                self.context["date_dir"] = date_dir
            if start is not None and not str((self.context or {}).get("query_start_local", "") or "").strip():
                self.context["query_start_local"] = start.strftime("%Y-%m-%d %H:%M:%S")
            if end is not None and not str((self.context or {}).get("query_end_local", "") or "").strip():
                self.context["query_end_local"] = end.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return

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

    def _bind_reason_time_candidates(self, pool: List[Dict[str, Any]], *, pool_k: int) -> List[Dict[str, Any]]:
        """Create evidence-backed reason/time rows instead of global empty rows.

        For reason-only and reason+time tasks, the old generic path produced
        ``component=''`` candidates with no metrics.  Those can dominate with
        score=1.0 and break time/reason alignment.  This binder keeps the
        output fields component-free later, but ranks reasons through
        component-local metric support first.
        """
        if not self.need_reason or self.need_comp:
            return pool
        reason_scores = dict((self.context or {}).get("reason_scores") or {})
        allowed = [str(x).strip() for x in list((self.context or {}).get("allowed_reasons") or []) if str(x).strip()]
        reasons = list(dict.fromkeys(list(allowed) + list(reason_scores.keys())))
        if not reasons:
            return pool

        comp_rows = self._component_evidence_rows(pool)
        time_rows = [dict(x) for x in list((self.context or {}).get("time_candidates") or []) if isinstance(x, Mapping)]
        default_dt = str((self.context or {}).get("time_choice", "") or "").strip()
        if self.need_time and not default_dt:
            for item in time_rows:
                dt = str(item.get("datetime", "") or "").strip()
                if dt:
                    default_dt = dt
                    break

        max_reason = max([abs(float(v or 0.0)) for v in reason_scores.values()] + [1.0])
        bound: List[Dict[str, Any]] = []
        for comp_item in comp_rows[: max(12, int(pool_k) * 2)]:
            comp = str(comp_item.get("component", "") or "").strip()
            metrics = self._metrics_for_component(comp, comp_item)
            if not comp or not metrics:
                continue
            for reason in reasons:
                support = self._reason_support(component=comp, reason=reason, metrics=metrics)
                if support <= 0.0:
                    continue
                dt = self._best_time_for_component_reason(comp, reason, default_dt, time_rows)
                base = float(comp_item.get("score", 0.0) or 0.0)
                rs = float(reason_scores.get(reason, 0.0) or 0.0) / max(max_reason, 1e-9)
                score = 0.52 * support + 0.28 * rs + 0.20 * max(0.0, min(1.0, base))
                bound.append(
                    {
                        "component": comp,
                        "reason": reason,
                        "datetime": dt if self.need_time else "",
                        "score": float(score),
                        "source": "core_reason_time_binder",
                        "supporting_metrics": metrics[:8],
                        "component_type": self._component_type(comp),
                        "reason_metric_support": float(support),
                    }
                )

        if not bound:
            # Keep reason surface as fallback, but make empty-evidence rows
            # weaker than any evidence-backed candidate.
            rows = self._replace_reason_surface_if_needed(pool, pool_k=pool_k)
            for item in rows:
                if not str(item.get("component", "") or "").strip() and not item.get("supporting_metrics"):
                    item["score"] = min(float(item.get("score", 0.0) or 0.0), 0.42)
                    item["source"] = f"{item.get('source', '')}+weak_empty_reason_surface"
            return rows

        bound.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return bound[: max(int(pool_k), self.k)]

    def _expand_component_reason_cross_product(self, pool: List[Dict[str, Any]], *, pool_k: int) -> List[Dict[str, Any]]:
        """Expand component candidates across taxonomy reasons with evidence.

        This generalizes the old Market-specific component×reason expansion.
        It is active for tasks that require both component and reason.
        """
        if not (self.need_comp and self.need_reason):
            return pool
        if self.adapter is None:
            return pool
        try:
            taxonomy = [str(x).strip() for x in list(self.adapter.basil_reason_taxonomy() or []) if str(x).strip()]
        except Exception:
            taxonomy = []
        if not taxonomy:
            return pool

        merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for item in pool:
            if not isinstance(item, Mapping):
                continue
            key = (str(item.get("component", "") or ""), str(item.get("reason", "") or ""), str(item.get("datetime", "") or ""))
            merged[key] = dict(item)

        comp_rows = self._component_evidence_rows(pool)
        for comp_item in comp_rows[: max(12, int(pool_k) * 2)]:
            comp = str(comp_item.get("component", "") or "").strip()
            metrics = self._metrics_for_component(comp, comp_item)
            if not comp or not metrics:
                continue
            base = max(0.0, min(1.0, float(comp_item.get("score", 0.0) or 0.0)))
            for reason in taxonomy:
                support = self._reason_support(component=comp, reason=reason, metrics=metrics)
                if support <= 0.0:
                    continue
                dt = str(comp_item.get("datetime", "") or "")
                if self.need_time:
                    dt = self._best_time_for_component_reason(comp, reason, dt, list((self.context or {}).get("time_candidates") or []))
                score = 0.58 * support + 0.32 * base
                key = (comp, reason, dt)
                prev = merged.get(key)
                row = {
                    "component": comp,
                    "reason": reason,
                    "datetime": dt,
                    "score": float(score),
                    "source": "core_component_reason_expand",
                    "supporting_metrics": metrics[:8],
                    "component_type": self._component_type(comp),
                    "reason_metric_support": float(support),
                }
                if prev is None or float(prev.get("score", 0.0) or 0.0) < score:
                    merged[key] = row

        rows = list(merged.values())
        rows.sort(key=self._evidence_backed_score, reverse=True)
        return rows[: max(int(pool_k) * 2, 30)]

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

    def _component_evidence_rows(self, pool: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in list(pool or []):
            if not isinstance(item, Mapping):
                continue
            comp = str(item.get("component", "") or "").strip()
            if not comp or comp in seen:
                continue
            row = dict(item)
            metrics = self._metrics_for_component(comp, row)
            if not metrics:
                continue
            row["supporting_metrics"] = metrics
            rows.append(row)
            seen.add(comp)
        top_metrics = dict((self.context or {}).get("top_metrics") or {})
        comp_scores = dict((self.context or {}).get("component_scores") or {})
        for comp, metrics in top_metrics.items():
            c = str(comp or "").strip()
            if not c or c in seen:
                continue
            rows.append(
                {
                    "component": c,
                    "reason": "",
                    "datetime": "",
                    "score": float(comp_scores.get(c, 0.0) or 0.0),
                    "source": "core_component_metric_surface",
                    "supporting_metrics": [str(x) for x in list(metrics or []) if str(x).strip()][:8],
                    "component_type": self._component_type(c),
                }
            )
            seen.add(c)
        rows.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return rows

    def _metrics_for_component(self, component: str, item: Mapping[str, Any]) -> List[str]:
        metrics = [str(x) for x in list(item.get("supporting_metrics", []) or []) if str(x).strip()]
        if metrics:
            return metrics[:8]
        top_metrics = dict((self.context or {}).get("top_metrics") or {})
        return [str(x) for x in list(top_metrics.get(str(component), []) or []) if str(x).strip()][:8]

    def _reason_support(self, *, component: str, reason: str, metrics: Sequence[str]) -> float:
        if self.adapter is not None:
            fn = getattr(self.adapter, "_reason_metric_support_score", None)
            if callable(fn):
                try:
                    return max(0.0, min(1.0, float(fn(component=component, reason=reason, metrics=list(metrics)))))
                except Exception:
                    pass
            fn2 = getattr(self.adapter, "_reason_metric_support", None)
            if callable(fn2):
                try:
                    return max(0.0, min(1.0, float(fn2(component=component, reason=reason, top_metrics=list(metrics)))))
                except Exception:
                    pass
            try:
                keywords = [str(x).lower() for x in list(self.adapter.reason_metric_keywords(str(reason)) or []) if str(x).strip()]
            except Exception:
                keywords = []
            if keywords:
                blob = " ".join(str(x).lower() for x in list(metrics or []))
                hits = len({k for k in keywords if k in blob})
                support = float(hits) / float(max(1, min(4, len(set(keywords)))))
                try:
                    expected = set(self.adapter.reason_compatible_component_types(str(reason)) or [])
                    ctype = str(self.adapter.component_semantic_type(str(component)) or "")
                    if expected and ctype:
                        support *= 1.15 if ctype in expected else 0.55
                except Exception:
                    pass
                return max(0.0, min(1.0, support))
        return 0.0

    def _component_type(self, component: str) -> str:
        if self.adapter is None:
            return self.component_family(component)
        try:
            return str(self.adapter.component_semantic_type(str(component)) or "")
        except Exception:
            return self.component_family(component)

    def _best_time_for_component_reason(
        self,
        component: str,
        reason: str,
        fallback: str,
        time_rows: Sequence[Mapping[str, Any]],
    ) -> str:
        if not self.need_time:
            return ""
        best_dt = str(fallback or "").strip()
        best_score = -1.0
        for item in list(time_rows or []):
            if not isinstance(item, Mapping):
                continue
            dt = str(item.get("datetime", "") or "").strip()
            if not dt:
                continue
            c0 = str(item.get("component", "") or "").strip()
            r0 = str(item.get("reason", "") or "").strip()
            score = float(item.get("score", 0.0) or 0.0)
            if c0 and c0 == component:
                score += 0.50
            elif c0:
                score -= 0.08
            if r0 and r0 == reason:
                score += 0.25
            if score > best_score:
                best_score = score
                best_dt = dt
        return best_dt

    def _evidence_backed_score(self, item: Mapping[str, Any]) -> float:
        score = float(item.get("score", 0.0) or 0.0)
        metrics = list(item.get("supporting_metrics", []) or [])
        support = float(item.get("reason_metric_support", 0.0) or 0.0)
        if not str(item.get("component", "") or "").strip() and not metrics:
            score *= 0.35
        if metrics:
            score += 0.12
        if support:
            score += 0.18 * max(0.0, min(1.0, support))
        return float(score)

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
