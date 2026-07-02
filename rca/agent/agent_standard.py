from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from rca.common.tasks import required_root_count_from_instruction, task_need_flags
from rca.adapter.registry import get_dataset_adapter
from rca.execution.field_solvers import assemble_prediction, solve_component, solve_reason, solve_time
from rca.execution.framework import ExecutionResult, Strategy, TaskContext, TaskRuntime, TaskSpec, UnifiedTaskRunner
from rca.execution.indexed_evidence_agent import IndexedEvidenceAgentRunner
from rca.execution.indexed_evidence_agent import _collect_agent_stats, _load_counterfactual_prompts, _safe_prediction_to_tuples, _safe_prediction_to_tuples_constrained
from rca.index.config import apply_adaptive_profile_config, apply_dataset_basil_config, basil_config_from_args, basil_enabled
from rca.index.query import basil_query_config_from_mapping, query_basil_components
from rca.execution.telemetry_solver import canonicalize_prediction
from rca.execution.telemetry_solver import estimate_k_occurrence_datetimes_from_trace
from rca.adapter.utils.time_candidates import estimate_occurrence_datetime_candidates
from rca.adapter.utils.component_candidates import build_component_candidates, parse_time_window_generic
from rca.counterfactor_agent.agent import CounterfactualRCAAgent


@dataclass(frozen=True)
class AgentStandardRuntimeExtras:
    light_agent_tasks: Set[int] = field(default_factory=set)


def infer_difficulty(task_id: int) -> str:
    tid = int(task_id or 0)
    if tid <= 3:
        return "easy"
    if tid <= 6:
        return "middle"
    return "hard"


def build_task_context(
    *,
    dataset: str,
    task_id: int,
    instruction: str,
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
        task_focus=str(task_focus),
        difficulty=infer_difficulty(int(task_id)),
    )
    rt = TaskRuntime(
        args=args,
        logger=logger,
        no_llm_fallback=bool(no_llm_fallback),
        extras={"agent_standard": extras or AgentStandardRuntimeExtras()},
    )
    return TaskContext(spec=spec, runtime=rt)


def _task_fields(task_id: int) -> List[str]:
    need_time, need_comp, need_reason = task_need_flags(int(task_id))
    fields: List[str] = []
    if need_time:
        fields.append("time")
    if need_comp:
        fields.append("component")
    if need_reason:
        fields.append("reason")
    return fields


def _prediction_from_tuples(
    items: Iterable[Dict[str, Any]],
    *,
    task_id: int,
    k: int,
    adapter: Any = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    need_time, need_comp, need_reason = task_need_flags(int(task_id))
    items_list = [dict(x) for x in list(items or []) if isinstance(x, Mapping)]
    if adapter is not None and bool(need_comp) and bool(need_reason):
        try:
            if adapter.__class__.__name__ == "MarketAdapter":
                adjusted = adapter.postprocess_output_tuples(
                    items_list,
                    context=dict(context or {}),
                    need_comp=True,
                    need_reason=True,
                    need_time=bool(need_time),
                )
                if isinstance(adjusted, list) and adjusted:
                    items_list = [dict(x) for x in adjusted if isinstance(x, Mapping)]
        except Exception:
            pass
    obj: Dict[str, Dict[str, str]] = {}
    for item in items_list[: max(1, int(k))]:
        root: Dict[str, str] = {}
        if need_time:
            dt = str((item or {}).get("datetime") or (item or {}).get("occurrence_datetime") or "").strip()
            if dt:
                root["root cause occurrence datetime"] = dt
        if need_comp:
            comp = str((item or {}).get("component") or "").strip()
            if comp and adapter is not None:
                try:
                    comp = str(adapter.parse_component_id(comp) or comp)
                except Exception:
                    pass
            if comp:
                root["root cause component"] = comp
        if need_reason:
            reason = str((item or {}).get("reason") or "").strip()
            if reason and adapter is not None:
                try:
                    reason = str(adapter.parse_root_cause_label(reason) or reason)
                except Exception:
                    pass
            if reason:
                root["root cause reason"] = reason
        if root:
            obj[str(len(obj) + 1)] = root
        if len(obj) >= max(1, int(k)):
            break
    return json.dumps(obj or {"1": {}}, ensure_ascii=False)


def _tuples_from_prediction(prediction: str) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(str(prediction or ""))
    except Exception:
        return []
    if not isinstance(obj, Mapping):
        return []
    rows: List[Dict[str, Any]] = []
    for key in sorted(obj.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        root = obj.get(key)
        if not isinstance(root, Mapping):
            continue
        rows.append(
            {
                "component": str(root.get("root cause component", "") or ""),
                "reason": str(root.get("root cause reason", "") or ""),
                "datetime": str(root.get("root cause occurrence datetime", "") or ""),
            }
        )
    return rows


def _timing_step(
    *,
    strategy: str,
    dataset: str,
    task_id: int,
    total_seconds: float,
    setup_seconds: float = 0.0,
    candidate_generation_seconds: float = 0.0,
    component_candidate_seconds: float = 0.0,
    time_candidate_seconds: float = 0.0,
    basil_timings: Optional[Dict[str, float]] = None,
    agent_call_seconds: float = 0.0,
    postprocess_seconds: float = 0.0,
    llm_input_tokens: int = 0,
    llm_called: bool = False,
    candidate_pool_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bt = dict(basil_timings or {})
    evidence_build = float(bt.get("basil.stage1_evidence_index_seconds", 0.0) or 0.0)
    index_query = float(bt.get("basil.evidence_index_query_seconds", 0.0) or 0.0)
    index_total = float(bt.get("basil.total_seconds", 0.0) or 0.0)
    decision = float(bt.get("basil.candidate_decision_seconds", 0.0) or 0.0)
    breakdown = {
        "total_execution_seconds": float(total_seconds),
        "setup_seconds": float(setup_seconds),
        "candidate_generation_seconds": float(candidate_generation_seconds),
        "component_candidate_seconds": float(component_candidate_seconds),
        "time_candidate_seconds": float(time_candidate_seconds),
        "basil_evidence_build_seconds": float(evidence_build),
        "basil_index_query_seconds": float(index_query),
        "basil_index_total_seconds": float(index_total),
        "basil_candidate_decision_seconds": float(decision),
        "agent_call_seconds": float(agent_call_seconds),
        "llm_input_tokens": int(llm_input_tokens),
        "llm_gate_called": bool(llm_called),
        "postprocess_seconds": float(postprocess_seconds),
    }
    diag = dict(candidate_pool_diagnostics or {})
    for key in [
        "candidate_pool_count",
        "candidate_component_count",
        "candidate_reason_count",
        "candidate_time_count",
        "candidate_component_family_count",
        "candidate_time_cluster_count",
        "llm_max_prompt_candidates",
        "llm_verify_count",
        "llm_changed_count",
        "llm_veto_count",
        "llm_adopted_count",
        "needs_component",
        "needs_reason",
        "needs_time",
        "candidate_pool_preview",
    ]:
        if key in diag:
            breakdown[key] = diag.get(key)
    return {
        "step": "timing",
        "action": "timing",
        "strategy": str(strategy),
        "dataset": str(dataset),
        "task_id": int(task_id),
        "timing_breakdown_seconds": breakdown,
    }


class BasilIndexStrategy:
    name = "basil_index"

    def can_handle(self, ctx: TaskContext) -> bool:
        return bool(basil_enabled(ctx.runtime.args))

    def run(self, ctx: TaskContext) -> Optional[ExecutionResult]:
        spec = ctx.spec
        args = ctx.runtime.args
        logger = ctx.runtime.logger
        t_total = time.perf_counter()
        top_k = int(getattr(args, "task3_top_k", 12) or 12)
        k_fail = max(1, int(required_root_count_from_instruction(spec.instruction)))
        cfg_map = basil_config_from_args(args, top_k, required_fields=_task_fields(spec.task_id), task_id=spec.task_id)
        cfg_map = apply_adaptive_profile_config(cfg_map, dataset=spec.dataset, args=args)
        top_k = int(cfg_map.get("top_k", top_k) or top_k)
        cfg_map = apply_dataset_basil_config(cfg_map, dataset=spec.dataset, args=args)
        adapter = get_dataset_adapter(spec.dataset)
        component_level = ""
        if adapter:
            try:
                component_level = ""
                cfg_map.update(
                    adapter.basil_query_config_overrides(
                        task_id=int(spec.task_id),
                        need_time=task_need_flags(spec.task_id)[0],
                        need_comp=task_need_flags(spec.task_id)[1],
                        need_reason=task_need_flags(spec.task_id)[2],
                        component_level=component_level,
                    )
                )
            except Exception:
                pass
        top_k = int(cfg_map.get("top_k", top_k) or top_k)
        cfg = basil_query_config_from_mapping(cfg_map, top_k=top_k)

        t_query = time.perf_counter()
        query_reasons: List[str] = []
        if adapter is not None:
            reason_taxonomy_fn = getattr(adapter, "basil_reason_taxonomy", None)
            if callable(reason_taxonomy_fn):
                try:
                    query_reasons = [str(x) for x in reason_taxonomy_fn() if str(x).strip()]
                except Exception:
                    query_reasons = []
        if not query_reasons and adapter is not None and adapter.__class__.__name__ == "MarketAdapter":
            taxonomy = getattr(adapter, "config", {}).get("reason_taxonomy", {})
            if isinstance(taxonomy, dict):
                query_reasons = [str(x) for x in taxonomy.keys() if str(x).strip()]
        result = query_basil_components(
            dataset=spec.dataset,
            instruction=spec.instruction,
            top_k=top_k,
            center_time_s=None,
            delta_s=1800,
            reasons=query_reasons,
            config=cfg,
        )
        query_seconds = float(time.perf_counter() - t_query)
        try:
            setattr(result, "_component_level", component_level)
        except Exception:
            pass

        t_agent = time.perf_counter()
        decision = IndexedEvidenceAgentRunner().run(
            dataset=spec.dataset,
            instruction=spec.instruction,
            result=result,
            task_id=int(spec.task_id),
            args=args,
            no_llm_fallback=bool(ctx.runtime.no_llm_fallback),
            logger=logger,
        )
        agent_seconds = float(time.perf_counter() - t_agent)

        t_post = time.perf_counter()
        pred_context = {
            "candidate_tuples": list((decision.candidate_pool_diagnostics or {}).get("candidate_pool_preview") or [])
        }
        pred = _prediction_from_tuples(
            decision.selected_tuples,
            task_id=int(spec.task_id),
            k=k_fail,
            adapter=adapter,
            context=pred_context,
        )
        pred = canonicalize_prediction(pred)
        if adapter is not None and adapter.__class__.__name__ == "MarketAdapter":
            pred_rows = _tuples_from_prediction(pred)
            if pred_rows:
                pred = _prediction_from_tuples(
                    pred_rows,
                    task_id=int(spec.task_id),
                    k=k_fail,
                    adapter=adapter,
                    context=pred_context,
                )
        post_seconds = float(time.perf_counter() - t_post)
        total_seconds = float(time.perf_counter() - t_total)

        trajectory: List[Any] = []
        trajectory.extend(list(decision.trajectory or []))
        trajectory.append(
            _timing_step(
                strategy=self.name,
                dataset=spec.dataset,
                task_id=int(spec.task_id),
                total_seconds=total_seconds,
                candidate_generation_seconds=query_seconds,
                component_candidate_seconds=query_seconds,
                basil_timings=dict(getattr(result.stats, "timings", {}) or {}),
                agent_call_seconds=agent_seconds,
                postprocess_seconds=post_seconds,
                llm_input_tokens=int(decision.input_tokens),
                llm_called=bool(decision.llm_called),
                candidate_pool_diagnostics=dict(decision.candidate_pool_diagnostics or {}),
            )
        )
        return ExecutionResult(prediction=str(pred), trajectory=trajectory, prompt=list(decision.prompt or []))


class FieldSolverStrategy:
    name = "field_solver"

    def can_handle(self, ctx: TaskContext) -> bool:
        return True

    def run(self, ctx: TaskContext) -> Optional[ExecutionResult]:
        spec = ctx.spec
        args = ctx.runtime.args
        t_total = time.perf_counter()
        need_time, need_comp, need_reason = task_need_flags(int(spec.task_id))
        k_fail = max(1, int(required_root_count_from_instruction(spec.instruction)))
        extras = ctx.runtime.extras.get("agent_standard") if isinstance(ctx.runtime.extras, dict) else None
        light_agent_tasks = set(getattr(extras, "light_agent_tasks", set()) or set())

        adapter_for_non_basil = get_dataset_adapter(spec.dataset)
        is_bank = adapter_for_non_basil is not None and adapter_for_non_basil.__class__.__name__ == "BankAdapter"
        is_market = adapter_for_non_basil is not None and adapter_for_non_basil.__class__.__name__ == "MarketAdapter"
        is_telecom = adapter_for_non_basil is not None and adapter_for_non_basil.__class__.__name__ == "TelecomAdapter"
        use_non_basil_counterfactual = (
            (bool(is_bank) or bool(is_market) or bool(is_telecom))
            and (not basil_enabled(args))
            and bool(getattr(args, "uncertainty_llm", False))
            and (not bool(getattr(args, "deterministic_only", False)))
            and (not bool(ctx.runtime.no_llm_fallback))
            and (not (bool(need_time) and (not bool(need_comp)) and (not bool(need_reason))))
        )

        if use_non_basil_counterfactual:
            t_candidates = time.perf_counter()
            root_dt, center_time_s, _ev_time = solve_time(
                dataset=spec.dataset,
                instruction=spec.instruction,
                need_time=bool(need_time),
            )
            candidate_seconds = float(time.perf_counter() - t_candidates)

            adapter = adapter_for_non_basil
            component_level = ""

            hint_components: List[str] = []
            hint_reasons: List[str] = []
            taxonomy_reasons: List[str] = []
            if adapter is not None:
                try:
                    taxonomy_reasons = [str(x) for x in list(adapter.basil_reason_taxonomy() or []) if str(x).strip()]
                except Exception:
                    taxonomy_reasons = []

            time_candidates: List[Dict[str, Any]] = []
            if bool(need_time):
                try:
                    # Multi-signal fusion: trace onset/peak + reason-aware metric onset + ±1min neighbors
                    _fused_cands = estimate_occurrence_datetime_candidates(
                        dataset=spec.dataset,
                        instruction=spec.instruction,
                        reasons=list(taxonomy_reasons) or [],
                        top_k=max(16, int(getattr(args, "time_candidate_top_k", 8) or 8) * 2),
                    )
                    for cand in _fused_cands[: max(1, int(getattr(args, "time_candidate_top_k", 8) or 8))]:
                        time_candidates.append({
                            "datetime": str(cand.datetime),
                            "source": str(cand.source),
                            "score": float(cand.score),
                            "component": str(cand.component or ""),
                            "reason": str(cand.reason or ""),
                            "evidence": dict(cand.evidence or {}),
                        })
                except Exception:
                    # Fallback to legacy trace-only candidates
                    try:
                        k_time = max(8, int(k_fail) * 4)
                        dts = estimate_k_occurrence_datetimes_from_trace(dataset=spec.dataset, instruction=spec.instruction, k=int(k_time))
                    except Exception:
                        dts = []
                    for rank, dt in enumerate(list(dts or [])[: max(1, int(getattr(args, "time_candidate_top_k", 8) or 8))], start=1):
                        time_candidates.append({
                            "datetime": str(dt),
                            "source": "trace_p95_duration",
                            "score": float(1.0 - (rank - 1) / float(max(1, len(dts) - 1))) if len(dts) > 1 else 1.0,
                            "component": "",
                            "reason": "",
                            "evidence": {},
                        })

            comp_scores: Dict[str, float] = {}
            comp_candidates: List[str] = []
            top_metrics: Dict[str, Any] = {}
            if bool(need_comp):
                try:
                    ev3_fresh = build_component_candidates(
                        dataset=spec.dataset,
                        instruction=spec.instruction,
                        top_k=int(getattr(args, "task3_top_k", 12) or 12),
                        center_time_s=center_time_s,
                        use_basil=False,
                        basil_config=basil_config_from_args(
                            args,
                            int(getattr(args, "task3_top_k", 12) or 12),
                            required_fields=None,
                            task_id=int(spec.task_id),
                        ),
                        basil_reasons=[],
                    )
                    comp_scores = dict(getattr(ev3_fresh, "combined_scores", {}) or {})
                    comp_candidates = [str(x) for x in list(getattr(ev3_fresh, "candidates", []) or []) if str(x).strip()]
                    top_metrics = dict(getattr(ev3_fresh, "top_metrics", {}) or {})
                except Exception:
                    comp_scores = {}
                    comp_candidates = []
                    top_metrics = {}

            if adapter is not None and comp_candidates:
                cc = []
                for c in comp_candidates:
                    try:
                        cc.append(str(adapter.parse_component_id(c) or c))
                    except Exception:
                        cc.append(str(c))
                comp_candidates = [x for x in dict.fromkeys(cc) if x]

            reasons = list(taxonomy_reasons) if bool(need_reason) else []
            if adapter is not None and reasons:
                rr = []
                for r in reasons:
                    try:
                        rr.append(str(adapter.parse_root_cause_label(r) or r))
                    except Exception:
                        rr.append(str(r))
                reasons = [x for x in dict.fromkeys(rr) if x]
            if not reasons and bool(need_reason):
                reasons = ["unknown"]

            reason_scores: Dict[str, float] = {}
            if reasons:
                n = max(1, len(reasons))
                for i, r in enumerate(reasons):
                    reason_scores[str(r)] = float(1.0 - (i / float(max(1, n - 1)))) if n > 1 else 1.0

            max_comp = max([abs(float(v)) for v in comp_scores.values()] + [0.0]) if comp_scores else 0.0
            if max_comp <= 1e-12:
                max_comp = 1.0

            pool: List[Dict[str, Any]] = []
            top_components = comp_candidates[: max(12, int(k_fail) * 4)] if comp_candidates else (hint_components or ["unknown"])
            top_reasons = reasons[: max(8, int(getattr(args, "basil_candidate_reason_top_r", 3) or 3))] if reasons else (hint_reasons or ["unknown"])
            for comp in top_components:
                cs = float(comp_scores.get(comp, 0.0) or 0.0) / float(max_comp)
                for r in top_reasons if bool(need_reason) else [""]:
                    rs = float(reason_scores.get(str(r), 0.0) or 0.0) if bool(need_reason) else 0.0
                    score = float(0.85 * cs + 0.15 * rs) if bool(need_comp) else float(rs)
                    pool.append(
                        {
                            "component": str(comp),
                            "reason": str(r),
                            "datetime": "",
                            "score": float(score),
                            "source": "non_basil_pool",
                            "evidence": [str(x) for x in list(top_metrics.get(str(comp), []) or [])[:6] if str(x).strip()],
                        }
                    )
            pool.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            pool = pool[: max(20, int(k_fail) * 12)]

            context = {
                "dataset": str(spec.dataset or ""),
                "need_time": bool(need_time),
                "need_comp": bool(need_comp),
                "need_reason": bool(need_reason),
                "difficulty": str(spec.difficulty or ""),
                "required_root_count": int(k_fail),
                "allowed_components": list(comp_candidates),
                "allowed_reasons": list(reasons),
                "candidate_tuples": [dict(x) for x in list(pool or [])],
                "component_scores": dict(comp_scores),
                "reason_scores": dict(reason_scores),
                "time_candidates": list(time_candidates),
                "time_choice": str(root_dt or ""),
                "top_metrics": dict(top_metrics),
                "basil_supporting_metrics": dict(top_metrics),
                "metric_scores": dict(comp_scores),
                "matched_metrics": dict(top_metrics),
                "component_level": str(component_level or ""),
                "enable_uncertainty_llm": True,
                "enable_time_llm": bool(getattr(args, "time_llm", False)),
                "deterministic_only": False,
                "llm_uncertain_gap_threshold": float(getattr(args, "evidence_score_gap_threshold", 0.15) or 0.15),
                "llm_confidence_floor": float(getattr(args, "evidence_top1_score_floor", 0.30) or 0.30),
                "llm_time_confidence_floor": float(getattr(args, "evidence_top1_score_floor", 0.30) or 0.30),
                "llm_verify_top_k": max(3, int(k_fail) * 3, int(getattr(args, "basil_candidate_reason_top_r", 3) or 3)),
                "llm_max_prompt_candidates": (
                    min(8, max(len(reasons), int(getattr(args, "basil_candidate_reason_top_r", 3) or 3)))
                    if bool(need_reason) and not bool(need_comp)
                    else min(8, max(5, int(k_fail) * 4))
                ),
                "evidence_score_gap_threshold": float(getattr(args, "evidence_score_gap_threshold", 0.15) or 0.15),
                "evidence_top1_score_floor": float(getattr(args, "evidence_top1_score_floor", 0.30) or 0.30),
                "evidence_top1_score_floor_single": float(getattr(args, "evidence_top1_score_floor_single", 0.50) or 0.50),
                "gate_skip_gap_threshold": float(getattr(args, "gate_skip_gap_threshold", 0.30) or 0.30),
                "evidence_time_gap_threshold": 0.20,
                "basil_max_trace_rows": int(getattr(args, "basil_max_trace_rows", 0) or 0),
            }

            # Inject reason_evidence so CounterfactualRCAAgent can do evidence-grounded
            # reason reranking (fixes CPU-bias for Telecom non-basil path).
            if bool(need_reason) and adapter is not None:
                try:
                    from rca.execution.light_agent import _format_reason_evidence
                    from rca.adapter.utils.reason_candidates import predict_reason_universal
                    raw = adapter.solve_reason(
                        spec=None,
                        instruction=str(spec.instruction or ""),
                        scoring_points="",
                        args=args,
                        need_reason=True,
                    )
                    raw_ev = raw[3] if (raw is not None and len(raw) >= 4) else None
                    if raw_ev is None:
                        _, raw_ev = predict_reason_universal(
                            dataset=str(spec.dataset or ""),
                            instruction=str(spec.instruction or ""),
                            scoring_points="",
                        )
                    if isinstance(raw_ev, dict):
                        context["reason_evidence"] = _format_reason_evidence(raw_ev)
                except Exception:
                    pass

            agent_prompt, basic_prompt = _load_counterfactual_prompts(spec.dataset)
            agent = CounterfactualRCAAgent(agent_prompt=agent_prompt, basic_prompt=basic_prompt)
            t_agent = time.perf_counter()
            try:
                prediction, agent_traj, prompt = agent.run(
                    instruction=spec.instruction,
                    logger=ctx.runtime.logger,
                    dataset_name=spec.dataset,
                    max_step=int(getattr(args, "controller_max_step", 25) or 25),
                    max_turn=int(getattr(args, "controller_max_turn", 5) or 5),
                    task_focus=_task_focus(int(spec.task_id)),
                    context=context,
                )
            except Exception as exc:
                prediction, agent_traj, prompt = str(exc), [], []
            agent_seconds = float(time.perf_counter() - t_agent)

            if is_market:
                selected = _safe_prediction_to_tuples_constrained(
                    str(prediction or ""),
                    [dict(x) for x in list(pool or [])],
                    int(k_fail),
                    task_id=int(spec.task_id),
                    dataset=str(spec.dataset or ""),
                    adapter=adapter,
                    context=context,
                )
            else:
                selected = _safe_prediction_to_tuples(str(prediction or ""), [dict(x) for x in pool[: max(1, int(k_fail))]], int(k_fail))
            if adapter is not None and bool(need_time):
                picked: List[Dict[str, Any]] = []
                for item in selected:
                    comp = str(item.get("component", "") or "")
                    rr = str(item.get("reason", "") or "")
                    try:
                        override = adapter.select_time_candidate_override(
                            time_candidates=list(time_candidates),
                            component=str(comp),
                            reason=str(rr),
                            need_comp=bool(need_comp),
                            need_reason=bool(need_reason),
                            need_time=bool(need_time),
                            fallback=str(root_dt or ""),
                        )
                    except Exception:
                        override = None
                    if override is not None and str(override[0] or "").strip():
                        item = dict(item)
                        item["datetime"] = str(override[0])
                    elif time_candidates:
                        item = dict(item)
                        item["datetime"] = str(time_candidates[0].get("datetime", "") or "")
                    picked.append(item)
                selected = picked

            pred = _prediction_from_tuples(selected, task_id=int(spec.task_id), k=int(k_fail), adapter=adapter)
            pred = canonicalize_prediction(pred)

            llm_called, input_tokens, _reasons, llm_stats = _collect_agent_stats(agent_traj)
            total_seconds = float(time.perf_counter() - t_total)
            trajectory: List[Any] = []
            trajectory.extend(list(agent_traj or []))
            trajectory.append(
                _timing_step(
                    strategy=self.name,
                    dataset=spec.dataset,
                    task_id=int(spec.task_id),
                    total_seconds=total_seconds,
                    candidate_generation_seconds=candidate_seconds,
                    component_candidate_seconds=candidate_seconds if need_comp else 0.0,
                    time_candidate_seconds=candidate_seconds if need_time else 0.0,
                    agent_call_seconds=agent_seconds,
                    postprocess_seconds=0.0,
                    llm_input_tokens=int(input_tokens),
                    llm_called=bool(llm_called),
                    candidate_pool_diagnostics=dict(llm_stats or {}),
                )
            )
            prompt_items = list(prompt or []) if isinstance(prompt, list) else ([prompt] if str(prompt or "").strip() else [])
            return ExecutionResult(prediction=str(pred), trajectory=trajectory, prompt=prompt_items)

        t_candidates = time.perf_counter()
        root_dt, center_time_s, _ev_time = solve_time(
            dataset=spec.dataset,
            instruction=spec.instruction,
            need_time=bool(need_time),
        )
        det_reason, det_reason_other, det_reason_ready, ev2 = solve_reason(
            spec=spec,
            dataset=spec.dataset,
            instruction=spec.instruction,
            args=args,
            need_reason=bool(need_reason),
        )
        root_comp, dt_override, ev3, root_reason = solve_component(
            spec=spec,
            dataset=spec.dataset,
            instruction=spec.instruction,
            args=args,
            need_comp=bool(need_comp),
            need_time=bool(need_time),
            need_reason=bool(need_reason),
            center_time_s=center_time_s,
            det_reason=det_reason,
            det_reason_ready=bool(det_reason_ready),
            det_reason_other=det_reason_other,
            ev2=ev2,
            light_agent_tasks=light_agent_tasks,
            no_llm_fallback=bool(ctx.runtime.no_llm_fallback),
        )
        if dt_override:
            root_dt = str(dt_override)
        candidate_seconds = float(time.perf_counter() - t_candidates)

        t_post = time.perf_counter()
        adapter = get_dataset_adapter(spec.dataset)
        pred_obj = None
        if adapter:
            try:
                pred_obj = adapter.assemble_prediction(
                    spec=spec,
                    instruction=spec.instruction,
                    need_time=bool(need_time),
                    need_comp=bool(need_comp),
                    need_reason=bool(need_reason),
                    k_fail=int(k_fail),
                    root_dt=str(root_dt or ""),
                    root_comp=str(root_comp or ""),
                    root_reason=str(root_reason or det_reason or ""),
                    det_reason=str(det_reason or ""),
                    det_reason_other=str(det_reason_other or ""),
                    det_reason_ready=bool(det_reason_ready),
                )
            except Exception:
                pred_obj = None
        if pred_obj is None:
            pred_obj = assemble_prediction(
                spec=spec,
                dataset=spec.dataset,
                instruction=spec.instruction,
                args=args,
                need_time=bool(need_time),
                need_comp=bool(need_comp),
                need_reason=bool(need_reason),
                k_fail=int(k_fail),
                root_dt=str(root_dt or ""),
                root_comp=str(root_comp or ""),
                root_reason=str(root_reason or det_reason or ""),
                det_reason=str(det_reason or ""),
                det_reason_other=str(det_reason_other or ""),
                det_reason_ready=bool(det_reason_ready),
            )
        pred = canonicalize_prediction(json.dumps(pred_obj, ensure_ascii=False))
        post_seconds = float(time.perf_counter() - t_post)
        total_seconds = float(time.perf_counter() - t_total)
        trajectory = [
            _timing_step(
                strategy=self.name,
                dataset=spec.dataset,
                task_id=int(spec.task_id),
                total_seconds=total_seconds,
                candidate_generation_seconds=candidate_seconds,
                component_candidate_seconds=candidate_seconds if need_comp else 0.0,
                time_candidate_seconds=candidate_seconds if need_time else 0.0,
                postprocess_seconds=post_seconds,
            )
        ]
        return ExecutionResult(prediction=str(pred), trajectory=trajectory, prompt=[])


class AgentStandardRouter:
    def __init__(self) -> None:
        self._strategies: List[Strategy] = [BasilIndexStrategy(), FieldSolverStrategy()]

    def route(self, ctx: TaskContext) -> List[Strategy]:
        return list(self._strategies)


def build_agent_standard_runner() -> UnifiedTaskRunner:
    return UnifiedTaskRunner(AgentStandardRouter())
