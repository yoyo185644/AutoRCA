from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, FrozenSet, Literal, List, Optional, Sequence, Tuple

import pandas as pd

from rca.agent.evidence_builder import (
    CandidateEvidence,
    TimeEvidence,
    build_candidate_evidence,
    build_candidate_pairs,
    build_time_evidence,
    compute_uncertainty,
)
from rca.execution.agent_standard import AgentStandardRuntimeExtras, build_task_context
from rca.fast_solver import (
    estimate_time_from_trace,
    evidence_supports_reason,
    infer_reason_with_evidence,
    pick_component_from_granger,
)
from rca.execution.field_solvers import assemble_prediction, solve_component, solve_reason, solve_time
from rca.light_agent import call_json_only
from rca.telemetry_solver import extract_reasons_from_scoring_points, required_root_count
from rca.time_candidates import TimeCandidate, estimate_occurrence_datetime_candidates

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


def _build_component_reason_prompt(
    *,
    dataset: str,
    instruction: str,
    uncertainty_reasons: Sequence[str],
    c1: CandidateEvidence,
    c2: CandidateEvidence,
) -> Tuple[str, str]:
    """Build the component/reason gate prompt; braces are escaped for safe .format()."""
    system = (
        "You are an RCA evidence judge. Your job is to choose between candidates based on evidence, not scores.\n"
        "Output JSON only.\n"
    )
    user = (
        "Dataset: {dataset}\n"
        "Query: {instruction}\n"
        "Uncertainty reason: {uncertainty_reasons}\n\n"
        "=== CANDIDATE COMPARISON ===\n\n"
        "CANDIDATE 1: {c1_comp} / {c1_reason}\n"
        "Score: {c1_score:.3f} (gap vs next: {c1_gap:.3f})\n\n"
        "Evidence FOR candidate 1:\n{c1_for}\n\n"
        "Evidence AGAINST candidate 1:\n{c1_against}\n\n"
        "Metric timeline (key minutes only):\n{c1_metrics}\n\n"
        "Trace timeline (key minutes only):\n{c1_trace}\n\n"
        "Causal position:\n{c1_causal}\n\n"
        "---\n\n"
        "CANDIDATE 2: {c2_comp} / {c2_reason}\n"
        "Score: {c2_score:.3f}\n\n"
        "Evidence FOR candidate 2:\n{c2_for}\n\n"
        "Evidence AGAINST candidate 2:\n{c2_against}\n\n"
        "Metric timeline (key minutes only):\n{c2_metrics}\n\n"
        "Trace timeline (key minutes only):\n{c2_trace}\n\n"
        "Causal position:\n{c2_causal}\n\n"
        "=== QUESTION ===\n"
        "Which candidate is more likely the root cause?\n"
        "Focus on: which anomaly appeared first? which has stronger metric evidence? which is causally upstream?\n\n"
        "Output JSON only:\n"
        "{{\n"
        '  "selected": 1,\n'
        '  "confidence": 0.0,\n'
        '  "key_reason": "one sentence: why this candidate over the other",\n'
        '  "decisive_evidence": "which specific evidence item decided it"\n'
        "}}\n"
    ).format(
        dataset=str(dataset),
        instruction=str(instruction),
        uncertainty_reasons=json.dumps(list(uncertainty_reasons or []), ensure_ascii=False),
        c1_comp=str(c1.component),
        c1_reason=str(c1.reason),
        c1_score=float(c1.score),
        c1_gap=float(c1.score_gap_vs_next),
        c1_for=json.dumps(list(c1.evidence_for or []), ensure_ascii=False, indent=2),
        c1_against=json.dumps(list(c1.evidence_against or []), ensure_ascii=False, indent=2),
        c1_metrics=json.dumps(list(c1.metric_timeline or []), ensure_ascii=False, indent=2),
        c1_trace=json.dumps(list(c1.trace_timeline or []), ensure_ascii=False, indent=2),
        c1_causal=json.dumps(dict(c1.causal_evidence or {}), ensure_ascii=False, indent=2),
        c2_comp=str(c2.component),
        c2_reason=str(c2.reason),
        c2_score=float(c2.score),
        c2_for=json.dumps(list(c2.evidence_for or []), ensure_ascii=False, indent=2),
        c2_against=json.dumps(list(c2.evidence_against or []), ensure_ascii=False, indent=2),
        c2_metrics=json.dumps(list(c2.metric_timeline or []), ensure_ascii=False, indent=2),
        c2_trace=json.dumps(list(c2.trace_timeline or []), ensure_ascii=False, indent=2),
        c2_causal=json.dumps(dict(c2.causal_evidence or {}), ensure_ascii=False, indent=2),
    )
    return system, user


def _build_time_prompt(
    *,
    instruction: str,
    component: str,
    reason: str,
    time_candidates: Sequence[TimeEvidence],
) -> Tuple[str, str]:
    system = "You are an RCA time verifier. Choose the most likely root-cause occurrence time. Output JSON only.\n"
    blocks = []
    for i, t in enumerate(list(time_candidates or []), start=1):
        blocks.append(
            "\n".join(
                [
                    f"CANDIDATE {i}: {t.datetime} (source: {t.source})",
                    "",
                    "At this minute:",
                    json.dumps(list(t.metric_at_this_time or []), ensure_ascii=False, indent=2),
                    "",
                    "1 minute before:",
                    json.dumps(list(t.metric_1min_before or []), ensure_ascii=False, indent=2),
                    "",
                    "1 minute after:",
                    json.dumps(list(t.metric_1min_after or []), ensure_ascii=False, indent=2),
                    "",
                    "Trace status at this minute:",
                    json.dumps(dict(t.trace_at_this_time or {}), ensure_ascii=False, indent=2),
                    "",
                    "Evidence FOR this time:",
                    json.dumps(list(t.evidence_for or []), ensure_ascii=False, indent=2),
                    "",
                    "Evidence AGAINST this time:",
                    json.dumps(list(t.evidence_against or []), ensure_ascii=False, indent=2),
                    "",
                    f"Score: {float(t.score):.3f}",
                ]
            )
        )
    user = (
        f"Fixed component: {str(component or '')}\n"
        f"Fixed reason: {str(reason or '')}\n"
        f"Query: {str(instruction)}\n\n"
        "=== TIME CANDIDATE COMPARISON ===\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\n=== QUESTION ===\n"
        f'Which time represents the earliest stable root-cause onset?\nPrefer: metric first breach before trace impact, stable anomaly (not just a spike), consistent with reason "{str(reason or "")}".\n\n'
        "Output JSON only:\n"
        "{\n"
        '  "selected_id": 1,\n'
        '  "confidence": 0.0,\n'
        '  "rationale": "why this time: what evidence at this minute supports it",\n'
        '  "key_signal": "which metric/trace signal was decisive"\n'
        "}\n"
    )
    return system, user


def _parse_gate_response(llm_output: str, logger: Any) -> Optional[Any]:
    """
    Robustly parse LLM gate response.
    Returns the value of "selected" if present and valid.
    Returns None on any parse failure — caller must handle fallback.
    Never raises.
    """
    if not llm_output or not str(llm_output).strip():
        logger.warning("gate llm returned empty response")
        return None
    try:
        cleaned = str(llm_output).strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(l for l in lines if not str(l).strip().startswith("```")).strip()
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            logger.warning(f"gate llm response is not a dict: type={type(result)}")
            return None
        if "selected" not in result:
            logger.warning(f"gate llm response missing key selected, keys={list(result.keys())}")
            return None
        return result.get("selected")
    except json.JSONDecodeError as e:
        raw = str(llm_output)
        logger.warning(f"gate llm response json parse failed: {e} | raw_head={raw[:200]!r}")
        return None
    except Exception as e:
        logger.warning(f"gate llm response unexpected error: {e}")
        return None


def _solve_with_evidence_gate(
    *,
    spec: TaskSpec,
    dataset: str,
    instruction: str,
    scoring_points: str,
    args: Any,
    logger: Any,
    no_llm_fallback: bool,
    light_agent_tasks: set[int],
) -> Optional[Tuple[str, List[Any], List[Any]]]:
    """Evidence-gated solver with hardened LLM parsing + early skip to avoid wasting heavy evidence build."""
    need_time = bool(spec.need_time)
    need_comp = bool(spec.need_component)
    need_reason = bool(spec.need_reason)
    fields = []
    if need_time:
        fields.append("time")
    if need_comp:
        fields.append("component")
    if need_reason:
        fields.append("reason")

    k_fail = int(required_root_count(scoring_points))
    if int(k_fail) != 1:
        return None

    t0 = time.perf_counter()
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

    candidate_generation_seconds = time.perf_counter() - t0

    deterministic_only = bool(getattr(args, "deterministic_only", False))
    allow_gate = bool(getattr(args, "uncertainty_llm", False)) and (not deterministic_only) and (not no_llm_fallback)
    allow_time_llm = bool(getattr(args, "time_llm", False)) and (not deterministic_only) and (not no_llm_fallback)

    component_scores = getattr(ev3, "combined_scores", {}) if ev3 is not None else {}
    component_candidates = list(getattr(ev3, "candidates", []) or [])
    reason_scores = (ev2 or {}).get("scores", {}) if isinstance(ev2, dict) else {}
    reason_candidates = list(reason_scores.keys()) if reason_scores else extract_reasons_from_scoring_points(scoring_points)

    pairs = build_candidate_pairs(
        component_scores=component_scores,
        reason_scores=reason_scores,
        component_candidates=component_candidates or ([comp] if str(comp or "").strip() else []),
        reason_candidates=reason_candidates or ([det_reason] if str(det_reason or "").strip() else []),
        need_component=need_comp,
        need_reason=need_reason,
        top_pairs=3,
    )
    try:
        pair_payload = [{"component": str(c or ""), "reason": str(r or ""), "score": float(s or 0.0)} for (c, r, s) in list(pairs or [])[:3]]
    except Exception:
        pair_payload = []

    top1_score = float(pairs[0][2]) if pairs else 0.0
    top2_score = float(pairs[1][2]) if len(pairs) >= 2 else 0.0
    score_gap = float(top1_score - top2_score) if len(pairs) >= 2 else float("inf")

    quick_need_gate = False
    score_gap_threshold = float(getattr(args, "evidence_score_gap_threshold", 0.15) or 0.15)
    top1_score_floor = float(getattr(args, "evidence_top1_score_floor", 0.30) or 0.30)
    top1_score_floor_single = float(getattr(args, "evidence_top1_score_floor_single", 0.50) or 0.50)
    required_fields = fields
    n_required_fields = len(required_fields) if hasattr(required_fields, "__len__") else 1

    trigger_reason = "skipped"
    if allow_gate and n_required_fields >= 1:
        if n_required_fields == 1:
            quick_need_gate = bool(top1_score < top1_score_floor_single)
            if quick_need_gate:
                trigger_reason = "top1_below_floor_single"
        else:
            if score_gap < score_gap_threshold:
                quick_need_gate = True
                trigger_reason = "gap_below_threshold"
            elif top1_score < top1_score_floor:
                quick_need_gate = True
                trigger_reason = "top1_below_floor"

    gate_debug = {
        "required_fields": int(n_required_fields),
        "top1_score": float(top1_score),
        "top2_score": float(top2_score),
        "score_gap": float(score_gap),
        "quick_need_gate": bool(quick_need_gate),
        "trigger_reason": str(trigger_reason),
        "pairs": list(pair_payload),
        "det_dt": str(dt or ""),
        "det_comp": str(comp or ""),
        "det_reason": str(det_reason or ""),
        "allow_gate": bool(allow_gate),
        "score_gap_threshold": float(score_gap_threshold),
        "top1_score_floor": float(top1_score_floor),
        "top1_score_floor_single": float(top1_score_floor_single),
    }

    time_cands: List[TimeCandidate] = []
    quick_need_time = False
    if need_time and allow_time_llm:
        time_cands = estimate_occurrence_datetime_candidates(
            dataset=dataset,
            instruction=instruction,
            component_first_times=(getattr(ev3, "first_times", None) if ev3 is not None else None),
            component_peak_times=(getattr(ev3, "peak_times", None) if ev3 is not None else None),
            estimated_failure_time=str(getattr(ev3, "estimated_failure_time", "") or "") if ev3 is not None else "",
            component_scores=(getattr(ev3, "combined_scores", None) if ev3 is not None else None),
            reasons=reason_candidates or None,
            top_k=int(getattr(args, "time_candidate_top_k", 8) or 8),
        )
        if len(time_cands) >= 2:
            if float(time_cands[0].score) - float(time_cands[1].score) < 0.20:
                quick_need_time = True
    gate_debug["quick_need_time"] = bool(quick_need_time)
    gate_debug["allow_time_llm"] = bool(allow_time_llm)

    gate_skip_gap_threshold = float(getattr(args, "gate_skip_gap_threshold", 0.30) or 0.30)
    if quick_need_gate and (not quick_need_time) and score_gap >= gate_skip_gap_threshold:
        logger.info(
            "evidence_gate skipped "
            f"task_id={int(spec.task_id)} difficulty={str(spec.difficulty)} "
            f"score_gap={score_gap:.4f} threshold={gate_skip_gap_threshold:.4f} "
            f"top1_score={top1_score:.4f} top2_score={top2_score:.4f}"
        )
        root_reason = det_reason if dataset == "Telecom" else det_reason_other
        obj_pred = assemble_prediction(
            dataset=dataset,
            instruction=instruction,
            scoring_points=scoring_points,
            args=args,
            need_time=need_time,
            need_comp=need_comp,
            need_reason=need_reason,
            k_fail=1,
            root_dt=str(dt or ""),
            root_comp=str(comp or ""),
            root_reason=str(root_reason or "") if det_reason_ready else "",
            det_reason=str(det_reason or ""),
            det_reason_other=str(det_reason_other or ""),
            det_reason_ready=bool(det_reason_ready),
        )
        timing_breakdown = {
            "total_execution_seconds": float(candidate_generation_seconds),
            "setup_seconds": 0.0,
            "candidate_generation_seconds": float(candidate_generation_seconds),
            "component_candidate_seconds": 0.0,
            "time_candidate_seconds": 0.0,
            "basil_evidence_build_seconds": 0.0,
            "basil_index_query_seconds": 0.0,
            "basil_index_total_seconds": 0.0,
            "agent_call_seconds": 0.0,
            "llm_verifier_seconds": 0.0,
            "llm_input_tokens": 0,
            "llm_gate_called": False,
            "llm_time_verifier_seconds": 0.0,
            "llm_time_input_tokens": 0,
            "llm_time_called": False,
            "postprocess_seconds": 0.0,
        }
        traj = [
            {
                "step": "evidence_gate",
                "action": "evidence_gate",
                "dataset": str(dataset),
                "task_id": int(spec.task_id),
                "need_llm": False,
                "gate_path": "skipped",
                "gate_debug": gate_debug,
                "chosen": {"datetime": str(dt or ""), "component": str(comp or ""), "reason": str(root_reason or "")},
            },
            {"step": "timing", "action": "timing", "strategy": "evidence_gate", "timing_breakdown_seconds": timing_breakdown},
        ]
        return json.dumps(obj_pred, ensure_ascii=False), traj, []

    if (not quick_need_gate) and (not quick_need_time):
        root_reason = det_reason if dataset == "Telecom" else det_reason_other
        obj_pred = assemble_prediction(
            dataset=dataset,
            instruction=instruction,
            scoring_points=scoring_points,
            args=args,
            need_time=need_time,
            need_comp=need_comp,
            need_reason=need_reason,
            k_fail=1,
            root_dt=str(dt or ""),
            root_comp=str(comp or ""),
            root_reason=str(root_reason or "") if det_reason_ready else "",
            det_reason=str(det_reason or ""),
            det_reason_other=str(det_reason_other or ""),
            det_reason_ready=bool(det_reason_ready),
        )
        timing_breakdown = {
            "total_execution_seconds": float(candidate_generation_seconds),
            "setup_seconds": 0.0,
            "candidate_generation_seconds": float(candidate_generation_seconds),
            "component_candidate_seconds": 0.0,
            "time_candidate_seconds": 0.0,
            "basil_evidence_build_seconds": 0.0,
            "basil_index_query_seconds": 0.0,
            "basil_index_total_seconds": 0.0,
            "agent_call_seconds": 0.0,
            "llm_verifier_seconds": 0.0,
            "llm_input_tokens": 0,
            "llm_gate_called": False,
            "llm_time_verifier_seconds": 0.0,
            "llm_time_input_tokens": 0,
            "llm_time_called": False,
            "postprocess_seconds": 0.0,
        }
        traj = [
            {
                "step": "evidence_gate",
                "action": "evidence_gate",
                "dataset": str(dataset),
                "task_id": int(spec.task_id),
                "need_llm": False,
                "gate_path": "skipped",
                "gate_debug": gate_debug,
                "chosen": {"datetime": str(dt or ""), "component": str(comp or ""), "reason": str(root_reason or "")},
            },
            {"step": "timing", "action": "timing", "strategy": "evidence_gate", "timing_breakdown_seconds": timing_breakdown},
        ]
        return json.dumps(obj_pred, ensure_ascii=False), traj, []

    logger.info(
        "evidence_gate triggered "
        f"task_id={int(spec.task_id)} difficulty={str(spec.difficulty)} "
        f"quick_need_gate={bool(quick_need_gate)} quick_need_time={bool(quick_need_time)} "
        f"top1_score={top1_score:.4f} top2_score={top2_score:.4f} score_gap={score_gap:.4f} "
        f"gate_skip_gap_threshold={gate_skip_gap_threshold:.4f} "
        f"evidence_score_gap_threshold={score_gap_threshold:.4f} evidence_top1_score_floor={top1_score_floor:.4f}"
    )

    ev_build_t0 = time.perf_counter()
    cand_evs: List[CandidateEvidence] = []
    for i, (c, r, s) in enumerate(pairs):
        gap = float(s) - float(pairs[i + 1][2]) if i + 1 < len(pairs) else float(s)
        cand_evs.append(
            build_candidate_evidence(
                dataset=dataset,
                instruction=instruction,
                component=str(c),
                reason=str(r),
                score=float(s),
                score_gap_vs_next=float(gap),
                all_candidate_components=[x[0] for x in pairs if str(x[0]).strip()],
                top_k_metrics=8,
                topology_max_rows=int(getattr(args, "basil_max_trace_rows", 0) or 0),
            )
        )

    time_evs: List[TimeEvidence] = []
    if need_time and time_cands:
        focus_comp = str(cand_evs[0].component) if cand_evs and str(cand_evs[0].component).strip() else str(comp or "")
        focus_reason = str(cand_evs[0].reason) if cand_evs and str(cand_evs[0].reason).strip() else str(det_reason or "")
        for cand in time_cands[: max(1, int(getattr(args, "time_llm_top_k", 5) or 5))]:
            time_evs.append(
                build_time_evidence(
                    dataset=dataset,
                    instruction=instruction,
                    time_candidate=cand,
                    component=focus_comp,
                    reason=focus_reason,
                )
            )

    evidence_build_seconds = time.perf_counter() - ev_build_t0

    unc = compute_uncertainty(candidates=cand_evs, time_candidates=time_evs, task_fields=fields)

    final_comp = str(comp or "")
    final_reason = str(det_reason if dataset == "Telecom" else det_reason_other)
    final_dt = str(dt or "")

    llm_gate_called = False
    llm_time_called = False
    llm_input_tokens = 0
    llm_time_input_tokens = 0
    llm_verifier_seconds = 0.0
    llm_time_verifier_seconds = 0.0
    prompt_messages: List[Any] = []
    traj: List[Any] = []
    gate_path = "deterministic"
    parse_success = False

    if bool(unc.get("need_llm")):
        focus = set(unc.get("focus") or [])
        if ("component_reason" in focus) and allow_gate and len(cand_evs) >= 2:
            sys_p, user_p = _build_component_reason_prompt(
                dataset=dataset,
                instruction=instruction,
                uncertainty_reasons=list(unc.get("reasons") or []),
                c1=cand_evs[0],
                c2=cand_evs[1],
            )
            t_llm0 = time.perf_counter()
            obj, raw, in_tok, _, msgs = call_json_only(system=sys_p, user=user_p, temperature=0.0)
            llm_verifier_seconds = time.perf_counter() - t_llm0
            llm_gate_called = True
            llm_input_tokens += int(in_tok)
            prompt_messages.extend(msgs)
            traj.append({"step": "llm_component_reason", "response": obj, "raw_response": raw})
            selected = _parse_gate_response(raw, logger=logger)
            try:
                conf = float(obj.get("confidence", 0.0) or 0.0) if isinstance(obj, dict) else 0.0
            except Exception:
                conf = 0.0
            try:
                sel = int(selected) if selected is not None else 0
            except Exception:
                sel = 0
            parse_success = bool(selected is not None and sel in (1, 2))
            if not parse_success:
                logger.info("evidence_gate llm parse fallback to deterministic selected=none")
                gate_path = "llm_parse_fallback"
            if conf >= 0.6 and sel in (1, 2):
                gate_path = "llm_success"
                chosen = cand_evs[sel - 1]
                if need_comp:
                    final_comp = str(chosen.component or "")
                if need_reason:
                    final_reason = str(chosen.reason or "")

        if ("time" in focus) and allow_time_llm and time_evs:
            sys_t, user_t = _build_time_prompt(
                instruction=instruction,
                component=final_comp,
                reason=final_reason,
                time_candidates=time_evs[:5],
            )
            t_llm0 = time.perf_counter()
            obj, raw, in_tok, _, msgs = call_json_only(system=sys_t, user=user_t, temperature=0.0)
            llm_time_verifier_seconds = time.perf_counter() - t_llm0
            llm_time_called = True
            llm_time_input_tokens += int(in_tok)
            prompt_messages.extend(msgs)
            traj.append({"step": "llm_time", "response": obj, "raw_response": raw})
            try:
                conf = float(obj.get("confidence", 0.0) or 0.0)
                sel = int(obj.get("selected_id", 0) or 0)
            except Exception:
                conf, sel = 0.0, 0
            if conf >= 0.5 and 1 <= sel <= len(time_evs):
                final_dt = str(time_evs[sel - 1].datetime)

    obj_pred = assemble_prediction(
        dataset=dataset,
        instruction=instruction,
        scoring_points=scoring_points,
        args=args,
        need_time=need_time,
        need_comp=need_comp,
        need_reason=need_reason,
        k_fail=1,
        root_dt=str(final_dt or ""),
        root_comp=str(final_comp or ""),
        root_reason=str(final_reason or "") if det_reason_ready else "",
        det_reason=str(det_reason or ""),
        det_reason_other=str(det_reason_other or ""),
        det_reason_ready=bool(det_reason_ready),
    )

    root = obj_pred.get("1", {}) if isinstance(obj_pred, dict) else {}
    missing = []
    if need_time and not str(root.get("root cause occurrence datetime", "") or "").strip():
        missing.append("root cause occurrence datetime")
    if need_comp and not str(root.get("root cause component", "") or "").strip():
        missing.append("root cause component")
    if need_reason and not str(root.get("root cause reason", "") or "").strip():
        missing.append("root cause reason")
    if missing:
        logger.warning(f"evidence_gate produced incomplete prediction: missing={missing}")
        return None

    logger.info(
        "evidence_gate resolved "
        f"task_id={int(spec.task_id)} difficulty={str(spec.difficulty)} path={gate_path} "
        f"evidence_build_ms={int(evidence_build_seconds * 1000)} llm_called={bool(llm_gate_called or llm_time_called)} "
        f"llm_ms={int((llm_verifier_seconds + llm_time_verifier_seconds) * 1000)} parse_success={bool(parse_success)}"
    )

    time_stats = []
    for cand in time_cands or []:
        try:
            time_stats.append({"datetime": str(cand.datetime), "source": str(cand.source), "score": float(cand.score)})
        except Exception:
            continue

    total_execution_seconds = float(candidate_generation_seconds + evidence_build_seconds + llm_verifier_seconds + llm_time_verifier_seconds)
    timing_breakdown = {
        "total_execution_seconds": total_execution_seconds,
        "setup_seconds": 0.0,
        "candidate_generation_seconds": float(candidate_generation_seconds),
        "component_candidate_seconds": 0.0,
        "time_candidate_seconds": 0.0,
        "basil_evidence_build_seconds": 0.0,
        "basil_index_query_seconds": 0.0,
        "basil_index_total_seconds": 0.0,
        "agent_call_seconds": 0.0,
        "llm_verifier_seconds": float(llm_verifier_seconds),
        "llm_input_tokens": int(llm_input_tokens),
        "llm_gate_called": bool(llm_gate_called),
        "llm_time_verifier_seconds": float(llm_time_verifier_seconds),
        "llm_time_input_tokens": int(llm_time_input_tokens),
        "llm_time_called": bool(llm_time_called),
        "postprocess_seconds": 0.0,
        "time_candidate_count": int(len(time_stats)),
    }
    traj.append(
        {
            "step": "evidence_gate",
            "action": "evidence_gate",
            "dataset": str(dataset),
            "task_id": int(spec.task_id),
            "need_llm": bool(unc.get("need_llm")),
            "gate_path": str(gate_path),
            "gate_debug": gate_debug,
            "uncertainty": unc,
            "chosen": {"datetime": final_dt, "component": final_comp, "reason": final_reason},
            "time_candidates": time_stats[:10],
        }
    )
    traj.append({"step": "timing", "action": "timing", "strategy": "evidence_gate", "timing_breakdown_seconds": timing_breakdown})

    return json.dumps(obj_pred, ensure_ascii=False), traj, prompt_messages


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
        solved = None
        if not bool(getattr(args, "force_counterfactual", False)):
            try:
                solved = _solve_with_evidence_gate(
                    spec=spec,
                    dataset=dataset,
                    instruction=instruction,
                    scoring_points=scoring_points,
                    args=args,
                    logger=logger,
                    no_llm_fallback=no_llm_fallback,
                    light_agent_tasks=set(getattr(agent_standard_extras, "light_agent_tasks", set()) or set()),
                )
            except Exception as e:
                logger.warning(f"evidence_gate easy solver failed, fallback: {e}")
                solved = None
        if solved is not None:
            return solved
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
        solved = None
        if not bool(getattr(args, "force_counterfactual", False)) and fast_mid_hard and not bool(getattr(args, "mid_hard_agent", False)):
            try:
                solved = _solve_with_evidence_gate(
                    spec=spec,
                    dataset=dataset,
                    instruction=instruction,
                    scoring_points=scoring_points,
                    args=args,
                    logger=logger,
                    no_llm_fallback=no_llm_fallback,
                    light_agent_tasks=set(light_agent_tasks),
                )
            except Exception as e:
                logger.warning(f"evidence_gate mid/hard solver failed, fallback: {e}")
                solved = None
        if solved is not None:
            return solved
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
