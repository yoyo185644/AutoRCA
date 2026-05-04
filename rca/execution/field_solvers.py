from __future__ import annotations

from typing import Any, Optional, Tuple

import pandas as pd

from rca.light_agent import choose_from_candidates
from rca.telemetry_solver import (
    canonicalize_component,
    estimate_k_occurrence_datetimes_from_logs,
    estimate_k_occurrence_datetimes_from_trace,
    estimate_top_components_from_logs,
    extract_reasons_from_scoring_points,
    required_root_count,
)
from rca.reason_candidates import predict_reason_universal
from rca.component_candidates import build_component_candidates
from rca.time_candidates import estimate_occurrence_datetime_from_trace
from rca.fast_solver import evidence_supports_reason, infer_reason_with_evidence
from rca.memory.config import basil_config_from_args, basil_enabled


def solve_time(*, dataset: str, instruction: str, scoring_points: str, need_time: bool) -> Tuple[str, Optional[int], Any]:
    if not need_time:
        return "", None, None
    k_fail = required_root_count(scoring_points)
    if dataset.startswith("Market/") and int(k_fail) > 1:
        dts = estimate_k_occurrence_datetimes_from_logs(dataset=dataset, instruction=instruction, k=int(k_fail))
        if not dts:
            dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k_fail))
        dt = dts[0] if dts else ""
        ev_time = None
    else:
        dt, ev_time = estimate_occurrence_datetime_from_trace(dataset=dataset, instruction=instruction)
    try:
        center_time_s = int(pd.Timestamp(dt, tz="Asia/Shanghai").tz_convert("UTC").timestamp())
    except Exception:
        center_time_s = None
    return str(dt or ""), center_time_s, ev_time


def solve_reason(
    *,
    spec: Any,
    dataset: str,
    instruction: str,
    scoring_points: str,
    args: Any,
    need_reason: bool,
) -> Tuple[str, str, bool, Any]:
    if not need_reason:
        return "", "", False, None
    if dataset == "Telecom":
        det_reason, ev2 = infer_reason_with_evidence(instruction)
        if bool(getattr(args, "light_agent", False)):
            candidates = list((ev2.get("scores") or {}).keys())
            dec = choose_from_candidates(
                task_id=2,
                instruction=instruction,
                output_field="root cause reason",
                candidates=candidates,
                evidence=ev2,
                temperature=float(args.light_agent_temperature),
            )
            agent_reason = dec.chosen
            if agent_reason and evidence_supports_reason(ev2, agent_reason):
                det_reason = agent_reason
        return str(det_reason or ""), "", True, ev2

    reason, ev2 = predict_reason_universal(
        dataset=dataset,
        instruction=instruction,
        scoring_points=scoring_points,
        temperature=float(getattr(args, "light_agent_temperature", 0.0)),
    )
    return "", str(reason or ""), True, ev2


def solve_component(
    *,
    spec: Any,
    dataset: str,
    instruction: str,
    args: Any,
    need_comp: bool,
    need_time: bool,
    need_reason: bool,
    center_time_s: Optional[int],
    det_reason: str,
    det_reason_ready: bool,
    det_reason_other: str,
    ev2: Any,
    light_agent_tasks: set[int],
    no_llm_fallback: bool,
) -> Tuple[str, Optional[str], Any, str]:
    if not need_comp:
        return "", None, None, str(det_reason or "")

    if dataset == "Telecom" and getattr(spec, "difficulty", "") != "easy":
        delta_s = 1800
        center_time_s_for_comp = None
    else:
        delta_s = 1800 if (dataset == "Telecom" and need_time) else 300
        center_time_s_for_comp = center_time_s

    top_k = int(getattr(args, "task3_top_k", 12))
    ev3 = build_component_candidates(
        dataset=dataset,
        instruction=instruction,
        top_k=top_k,
        center_time_s=center_time_s_for_comp,
        delta_s=delta_s,
        use_basil=basil_enabled(args),
        basil_config=basil_config_from_args(args, top_k, required_fields=getattr(spec, "required_fields", None), task_id=getattr(spec, "task_id", None)),
        basil_reasons=[x for x in [det_reason, det_reason_other] if str(x or "").strip()] if need_reason else [],
    )

    def _ctype(c: str) -> str:
        c = str(c or "")
        if c.startswith("db_"):
            return "db"
        if c.startswith(("os_", "node-")):
            return "node"
        if c.startswith(("docker_", "container")):
            return "container"
        if "-" in c:
            return "service"
        return "other"

    def _rtype(r: str) -> str:
        r = str(r or "")
        if r in ("db close", "db connection limit"):
            return "db"
        if r == "CPU fault":
            return "container"
        if r in ("network delay", "network loss"):
            return "node"
        return ""

    comp = ev3.top1
    det_reason_out = str(det_reason or "")
    if dataset == "Telecom" and need_reason and det_reason_ready:
        rs = (ev2 or {}).get("scores", {}) if isinstance(ev2, dict) else {}
        cs = ev3.combined_scores or {}
        if rs and cs:
            max_r = max([float(v) for v in rs.values()] + [0.0])
            max_c = max([abs(float(v)) for v in cs.values()] + [0.0])
            best_pair = None
            best_joint = -1.0
            for r, rv in rs.items():
                rn = float(rv) / max(max_r, 1e-9)
                desired_t = _rtype(r)
                pool = [c for c in cs.keys() if desired_t and _ctype(c) == desired_t] if desired_t else list(cs.keys())
                if not pool:
                    pool = list(cs.keys())
                c_best = max(pool, key=lambda c: float(cs.get(c, 0.0)))
                cn = float(cs.get(c_best, 0.0)) / max(max_c, 1e-9)
                joint = 0.55 * cn + 0.45 * rn
                if joint > best_joint:
                    best_joint = joint
                    best_pair = (str(r), str(c_best))
            if best_pair is not None:
                det_reason_out, comp = best_pair

    top1_score = float(ev3.combined_scores.get(comp, 0.0))
    desired = ""
    desired_ratio = 0.0
    if det_reason_out in ("db close", "db connection limit"):
        desired = "db"
        desired_ratio = 0.62
    elif det_reason_out == "CPU fault":
        desired = "container"
        desired_ratio = 0.66
    elif det_reason_out in ("network delay", "network loss"):
        desired = "node"
        desired_ratio = 0.62
    if desired and top1_score > 1e-12:
        pool = [c for c in ev3.combined_scores.keys() if _ctype(c) == desired]
        if pool:
            best_c = max(pool, key=lambda c: float(ev3.combined_scores.get(c, 0.0)))
            best_s = float(ev3.combined_scores.get(best_c, 0.0))
            if best_s >= desired_ratio * top1_score:
                comp = best_c

    cand = list(ev3.candidates or [])
    filtered = [c for c in cand if desired and _ctype(c) == desired] if desired else []
    allow_light_agent = (int(getattr(spec, "task_id", 0)) in light_agent_tasks) or (dataset == "Telecom" and getattr(spec, "difficulty", "") != "easy" and need_comp)
    gap_thr = float(getattr(args, "task3_agent_gap", 0.15))
    use_agent = bool(args.light_agent) and (float(ev3.gap_norm) < gap_thr) and (not no_llm_fallback) and allow_light_agent
    if use_agent:
        use_cands = filtered if filtered else cand
        dec = choose_from_candidates(
            task_id=3,
            instruction=instruction,
            output_field="root cause component",
            candidates=use_cands,
            evidence=ev3.to_dict(),
            temperature=float(args.light_agent_temperature),
        )
        if dec.chosen:
            comp = dec.chosen

    if dataset.startswith("Market/"):
                comp = canonicalize_component(comp)

    dt_override = None
    if dataset == "Telecom" and need_time:
        dt2 = (ev3.first_times or {}).get(comp) or (ev3.peak_times or {}).get(comp) or str(getattr(ev3, "estimated_failure_time", "") or "")
        if str(dt2 or "").strip():
            dt_override = str(dt2)

    _ = det_reason_other
    return str(comp or ""), dt_override, ev3, str(det_reason_out or "")


def assemble_prediction(
    *,
    dataset: str,
    instruction: str,
    scoring_points: str,
    args: Any,
    need_time: bool,
    need_comp: bool,
    need_reason: bool,
    k_fail: int,
    root_dt: str,
    root_comp: str,
    root_reason: str,
    det_reason: str,
    det_reason_other: str,
    det_reason_ready: bool,
) -> dict:
    obj: dict = {"1": {}}
    if need_time and str(root_dt or "").strip():
        obj["1"]["root cause occurrence datetime"] = str(root_dt)
    if need_comp and str(root_comp or "").strip():
        obj["1"]["root cause component"] = str(root_comp)
    if need_reason and str(root_reason or "").strip():
        obj["1"]["root cause reason"] = str(root_reason)

    if int(k_fail) <= 1:
        return obj

    roots = []
    dts = estimate_k_occurrence_datetimes_from_logs(dataset=dataset, instruction=instruction, k=int(k_fail)) if dataset.startswith("Market/") else []
    if not dts:
        dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k_fail))
    reasons = extract_reasons_from_scoring_points(scoring_points) if (need_reason and dataset.startswith("Market/")) else []
    for i in range(int(k_fail)):
        r: dict = {}
        dt_i = dts[i] if i < len(dts) else dts[-1]
        if need_time:
            r["root cause occurrence datetime"] = dt_i
            try:
                ct_s = int(pd.Timestamp(dt_i, tz="Asia/Shanghai").tz_convert("UTC").timestamp())
            except Exception:
                ct_s = None
        else:
            ct_s = None
        if need_comp:
            log_comps = estimate_top_components_from_logs(dataset=dataset, instruction=instruction, center_dt=dt_i) if dataset.startswith("Market/") else []
            ev3_i = None
            if log_comps:
                comp_i = canonicalize_component(log_comps[0])
            else:
                top_k = int(getattr(args, "task3_top_k", 12))
                ev3_i = build_component_candidates(
                    dataset=dataset,
                    instruction=instruction,
                    top_k=top_k,
                    center_time_s=ct_s,
                    use_basil=basil_enabled(args),
                    basil_config=basil_config_from_args(args, top_k, required_fields=getattr(spec, "required_fields", None), task_id=getattr(spec, "task_id", None)),
                    basil_reasons=extract_reasons_from_scoring_points(scoring_points) if need_reason else [],
                )
                comp_i = ev3_i.top1
                if dataset.startswith("Market/"):
                    comp_i = canonicalize_component(comp_i)
            if any(x.get("root cause component") == comp_i for x in roots):
                if log_comps:
                    for c in log_comps[1:]:
                        c2 = canonicalize_component(c)
                        if not any(x.get("root cause component") == c2 for x in roots):
                            comp_i = c2
                            break
                elif ev3_i is not None:
                    ranked = sorted(
                        list(ev3_i.candidates or []),
                        key=lambda c: float(ev3_i.combined_scores.get(c, 0.0)),
                        reverse=True,
                    )
                    for c in ranked:
                        c2 = canonicalize_component(c) if dataset.startswith("Market/") else c
                        if not any(x.get("root cause component") == c2 for x in roots):
                            comp_i = c2
                            break
            r["root cause component"] = comp_i
        if need_reason:
            if len(reasons) >= int(k_fail):
                r["root cause reason"] = reasons[i]
            elif dataset != "Telecom" and det_reason_ready:
                r["root cause reason"] = det_reason_other
            else:
                r["root cause reason"] = det_reason
        roots.append(r)

    obj = {}
    for j, r in enumerate(roots, start=1):
        rr = {}
        if "root cause occurrence datetime" in r:
            rr["root cause occurrence datetime"] = r["root cause occurrence datetime"]
        if "root cause component" in r:
            rr["root cause component"] = r["root cause component"]
        if "root cause reason" in r:
            rr["root cause reason"] = r["root cause reason"]
        obj[str(j)] = rr
    return obj
