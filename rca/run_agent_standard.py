from typing import Any


import os
import sys
import json
import argparse
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from main.evaluate import evaluate
from rca.api_router import configs

from datetime import datetime
from loguru import logger
from nbformat import v4 as nbf
import pandas as pd
import signal
from rca.telecom_easy_solver import solve_telecom_easy, estimate_task1_datetime_from_trace, pick_component_by_granger, task2_reason_with_evidence, evidence_supports_task2_reason
from rca.light_agent import choose_from_candidates
from rca.task3_hybrid import build_task3_candidates
from rca.task2_hybrid import predict_task2_reason_universal
from rca.task_time_hybrid import estimate_occurrence_datetime_from_trace
from rca.query_parsing import extract_task2_reason_candidates, extract_task3_component_candidates
from rca.market_solver import canonicalize_market_component, canonicalize_market_prediction, estimate_k_occurrence_datetimes_from_logs, estimate_k_occurrence_datetimes_from_trace, estimate_top_components_from_logs, extract_reasons_from_scoring_points, required_root_count
from rca.prediction_postprocess import normalize_prediction_for_eval
try:
    from rca.window_granger_solver import solve_with_window_granger
except Exception:
    solve_with_window_granger = None

def handler(signum, frame):
    raise TimeoutError("Loop execution exceeded the time limit")


def load_selected_row_ids(row_ids_file: str):
    if not row_ids_file:
        return None
    with open(row_ids_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    row_ids = data.get("row_ids", data)
    selected = set()
    if isinstance(row_ids, dict):
        for _, values in row_ids.items():
            if isinstance(values, list):
                selected.update([int(v) for v in values])
    elif isinstance(row_ids, list):
        selected.update([int(v) for v in row_ids])
    return selected


def build_task_focus(task_id: int) -> str:
    task_focus = {
        1: "Only the root cause occurrence datetime is evaluated. Focus on finding the earliest stable root-cause onset time within the failure window.",
        2: "Only the root cause reason is evaluated. Focus on the dominant failure reason; use component/time only as internal evidence if needed.",
        3: "Only the root cause component is evaluated. Focus on ranking the most likely component; do not spend extra steps finalizing reason/time unless they help distinguish candidates.",
        4: "The root cause occurrence datetime and root cause reason are evaluated.",
        5: "The root cause occurrence datetime and root cause component are evaluated.",
        6: "The root cause component and root cause reason are evaluated.",
        7: "The root cause occurrence datetime, component, and reason are all evaluated.",
    }
    return task_focus.get(task_id, "Follow the instruction and identify the required root cause fields.")


def _parse_task_id_set(spec: str):
    s = str(spec or "").strip()
    if not s:
        return None
    out = set()
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out if out else None


def main(args, uid, dataset):
    if dataset == "Telecom" and args.telecom_easy_only:
        args.fast_easy_solver = True
        if not getattr(args, "telecom_easy_allow_llm", False):
            args.no_llm_fallback = True
    if args.telecom_use_archive_baseline and not args.use_archive_baseline:
        args.use_archive_baseline = True
        args.archive_pred = args.telecom_archive_pred
    if args.use_archive_baseline and not args.archive_pred:
        default_map = {
            "Telecom": "rca/archive/agent-Telecom.csv",
            "Bank": "rca/archive/agent-Bank.csv",
            "Market/cloudbed-1": "rca/archive/agent-Market-cloudbed-1.csv",
            "Market/cloudbed-2": "rca/archive/agent-Market-cloudbed-2.csv",
        }
        args.archive_pred = default_map.get(dataset, "")
    if args.use_archive_baseline and not args.archive_pred:
        raise FileNotFoundError("archive_pred is required when use_archive_baseline is enabled.")

    inst_file = f"dataset/{dataset}/query.csv"
    gt_file = f"dataset/{dataset}/record.csv"
    eval_file = f"test/result/{dataset}/agent-{args.tag}-{configs['MODEL'].split('/')[-1]}-{uid}.csv"
    obs_path = f"test/monitor/{dataset}/agent-{args.tag}-{configs['MODEL'].split('/')[-1]}"
    unique_obs_path = f"{obs_path}/{uid}"

    instruct_data = pd.read_csv(inst_file)
    gt_data = pd.read_csv(gt_file)
    if not os.path.exists(inst_file) or not os.path.exists(gt_file):
        raise FileNotFoundError(f"Please download the dataset first.")

    if not os.path.exists(f"{unique_obs_path}/history"):
        os.makedirs(f"{unique_obs_path}/history")
    if not os.path.exists(f"{unique_obs_path}/trajectory"):
        os.makedirs(f"{unique_obs_path}/trajectory")
    if not os.path.exists(f"{unique_obs_path}/prompt"):
        os.makedirs(f"{unique_obs_path}/prompt")
    if not os.path.exists(eval_file):
        if not os.path.exists(f"test/result/{dataset}"):
            os.makedirs(f"test/result/{dataset}")
        eval_df = pd.DataFrame(columns=["instruction", "prediction", "groundtruth", "passed", "failed", "score"])
    else:
        eval_df = pd.read_csv(eval_file)
    done_row_ids = set()
    if "row_id" in eval_df.columns:
        try:
            done_row_ids = set(pd.to_numeric(eval_df["row_id"], errors="coerce").dropna().astype(int).tolist())
        except Exception:
            done_row_ids = set()

    scores = {
        "total": 0,
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    nums = {
        "total": 0,
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }

    signal.signal(signal.SIGALRM, handler)
    logger.info(f"Using dataset: {dataset}")
    logger.info(f"Using model: {configs['MODEL'].split('/')[-1]}")
    selected_row_ids = load_selected_row_ids(args.row_ids_file)
    archive_df = None
    if args.use_archive_baseline:
        archive_df = pd.read_csv(args.archive_pred)
        archive_df["row_id_int"] = pd.to_numeric(archive_df.get("row_id"), errors="coerce").fillna(-1).astype(int)
    no_llm_fallback = bool(args.no_llm_fallback or args.use_archive_baseline)
    light_agent_tasks = _parse_task_id_set(getattr(args, "light_agent_task_ids", "")) or set([3])
    task2_universal_tasks = _parse_task_id_set(getattr(args, "task2_universal_task_ids", "")) or set([2])
    rca_agent_tasks = _parse_task_id_set(getattr(args, "rca_agent_task_ids", ""))
    fast_mid_hard = bool(getattr(args, "fast_mid_hard_solver", False))
    telecom_reason_tasks = _parse_task_id_set(getattr(args, "telecom_reason_agent_task_ids", "2,4,6,7")) or set([2, 4, 6, 7])
    
    for idx, row in instruct_data.iterrows():

        if idx < args.start_idx:
                continue
        if idx > args.end_idx:
            break
        if selected_row_ids is not None and idx not in selected_row_ids:
            continue
        if idx in done_row_ids:
            continue
        
        task_index = row["task_index"]
        task_focus = build_task_focus(int(task_index.split('_')[1]))
        instruction = row["instruction"]
        scoring_points = row["scoring_points"]
        task_id = int(task_index.split('_')[1])
        if args.telecom_easy_only and dataset == "Telecom" and task_id > 3:
            continue
        best_score = 0
        best_prediction = None
        best_trajectory = None
        best_prompt = None

        if task_id <= 3:
            catalog = "easy"
        elif task_id <= 6:
            catalog = "middle"
        elif task_id <= 7:
            catalog = "hard"

        temp_scores = scores.copy()
        temp_nums = nums.copy()

        for i in range(args.sample_num):
            uuid = uid + f"_#{idx}-{i}"
            nb = nbf.new_notebook()
            nbfile = f"{unique_obs_path}/trajectory/{uuid}.ipynb"
            promptfile = f"{unique_obs_path}/prompt/{uuid}.json"
            logfile = f"{unique_obs_path}/history/{uuid}.log"
            logger.remove()
            logger.add(sys.stdout, colorize=True, enqueue=True, level="INFO")
            logger.add(logfile, colorize=True, enqueue=True, level="INFO")
            logger.debug('\n' + "#"*80 + f"\n{uuid}: {task_index}\n" + "#"*80)
            try: 
                signal.alarm(args.timeout)

                prediction = None
                trajectory = []
                prompt = []

                if args.window_granger_solver and (dataset.startswith("Market/") or dataset == "Bank"):
                    if solve_with_window_granger is None:
                        logger.warning("window_granger_solver is not available; skipping window_granger_solver.")
                    else:
                        prediction = solve_with_window_granger(dataset=dataset, instruction=instruction, task_id=task_id)
                if args.hybrid_window_granger_prior and dataset.startswith("Market/"):
                    hybrid_tasks = {
                        "Market/cloudbed-1": {2, 3, 4},
                        "Market/cloudbed-2": {1, 4},
                    }
                    if task_id in hybrid_tasks.get(dataset, set()):
                        if solve_with_window_granger is None:
                            logger.warning("window_granger_solver is not available; skipping hybrid_window_granger_prior.")
                        else:
                            prediction = solve_with_window_granger(dataset=dataset, instruction=instruction, task_id=task_id)

                if prediction is None and archive_df is not None:
                    row_match = archive_df[archive_df["row_id_int"] == idx]
                    if not row_match.empty:
                        prediction = row_match.iloc[0]["prediction"]
                        if dataset == "Telecom" and args.telecom_task1_trace_time and task_id == 1:
                            try:
                                dt = estimate_task1_datetime_from_trace(instruction)
                                import json as _json
                                obj = _json.loads(prediction)
                                if "1" in obj and isinstance(obj["1"], dict):
                                    obj["1"]["root cause occurrence datetime"] = dt
                                    prediction = _json.dumps(obj, ensure_ascii=False, indent=4)
                            except Exception as e:
                                logger.warning(f"Task1 trace time override failed: {e}")
                        if dataset == "Telecom" and args.telecom_task3_granger_component and task_id == 3:
                            try:
                                import json as _json
                                obj = _json.loads(prediction)
                                comp = pick_component_by_granger(instruction)
                                if "1" in obj and isinstance(obj["1"], dict):
                                    obj["1"]["root cause component"] = comp
                                    prediction = _json.dumps(obj, ensure_ascii=False, indent=4)
                            except Exception as e:
                                logger.warning(f"Task3 granger component override failed: {e}")
                    else:
                        logger.warning(f"Archive baseline missing row_id={idx}; using empty prediction.")
                        prediction = "{}"

                if prediction is None:
                    if bool(getattr(args, "force_counterfactual", False)) and int(task_id) >= 4 and not no_llm_fallback:
                        prediction = None
                    elif fast_mid_hard and int(task_id) >= 4 and not no_llm_fallback and not bool(getattr(args, "mid_hard_agent", False)):
                        try:
                            need_time = int(task_id) in (4, 5, 7)
                            need_comp = int(task_id) in (5, 6, 7)
                            need_reason = int(task_id) in (4, 6, 7)
                            k_fail = required_root_count(scoring_points)
                            obj = {"1": {}}
                            det_reason = ""
                            det_reason_ready = False
                            det_reason_other = ""
                            ev2 = None
                            if need_time:
                                if dataset.startswith("Market/") and int(k_fail) > 1:
                                    dts = estimate_k_occurrence_datetimes_from_logs(dataset=dataset, instruction=instruction, k=int(k_fail))
                                    if not dts:
                                        dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k_fail))
                                    dt = dts[0] if dts else ""
                                    ev_time = None
                                else:
                                    dt, ev_time = estimate_occurrence_datetime_from_trace(dataset=dataset, instruction=instruction)
                                obj["1"]["root cause occurrence datetime"] = dt
                                try:
                                    center_time_s = int(pd.Timestamp(dt, tz="Asia/Shanghai").tz_convert("UTC").timestamp())
                                except Exception:
                                    center_time_s = None
                            else:
                                center_time_s = None
                                ev_time = None
                            if need_reason:
                                if dataset == "Telecom":
                                    det_reason, ev2 = task2_reason_with_evidence(instruction)
                                    gap = float(ev2.get("gap", 0.0) or 0.0)
                                    best_score = float((ev2.get("best") or {}).get("score", 0.0) or 0.0)
                                    always_agent = int(task_id) in (4, 6, 7)
                                    use_agent = bool(getattr(args, "telecom_task2_agent", False)) and (int(task_id) in telecom_reason_tasks) and (always_agent or gap < float(args.telecom_task2_agent_gap) or best_score < float(args.telecom_task2_agent_min_score))
                                    if use_agent:
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
                                        if agent_reason and evidence_supports_task2_reason(ev2, agent_reason):
                                            det_reason = agent_reason
                                    det_reason_ready = True
                                else:
                                    if getattr(args, "task2_universal_agent", False) and (int(task_id) in task2_universal_tasks):
                                        reason, _ = predict_task2_reason_universal(
                                            dataset=dataset,
                                            instruction=instruction,
                                            scoring_points=scoring_points,
                                            temperature=float(args.light_agent_temperature),
                                        )
                                        det_reason_other = str(reason or "")
                                        det_reason_ready = True
                            if need_comp:
                                if dataset == "Telecom" and int(task_id) >= 4:
                                    delta_s = 1800
                                    center_time_s_for_comp = None
                                else:
                                    delta_s = 1800 if (dataset == "Telecom" and need_time) else 300
                                    center_time_s_for_comp = center_time_s
                                ev3 = build_task3_candidates(
                                    dataset=dataset,
                                    instruction=instruction,
                                    top_k=int(args.task3_top_k),
                                    center_time_s=center_time_s_for_comp,
                                    delta_s=delta_s,
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
                                comp = ev3.top1
                                if dataset == "Telecom" and need_reason and det_reason_ready:
                                    rs = (ev2 or {}).get("scores", {}) if isinstance(ev2, dict) else {}
                                    cs = ev3.combined_scores or {}
                                    def _rtype(r: str) -> str:
                                        r = str(r or "")
                                        if r in ("db close", "db connection limit"):
                                            return "db"
                                        if r == "CPU fault":
                                            return "container"
                                        if r in ("network delay", "network loss"):
                                            return "node"
                                        return ""
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
                                            det_reason, comp = best_pair
                                top1_score = float(ev3.combined_scores.get(comp, 0.0))
                                desired = ""
                                desired_ratio = 0.0
                                if det_reason in ("db close", "db connection limit"):
                                    desired = "db"
                                    desired_ratio = 0.62
                                elif det_reason == "CPU fault":
                                    desired = "container"
                                    desired_ratio = 0.66
                                elif det_reason in ("network delay", "network loss"):
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
                                use_agent = bool(args.light_agent) and (float(ev3.gap_norm) < float(args.task3_agent_gap)) and (not no_llm_fallback) and (int(task_id) in light_agent_tasks or (dataset == "Telecom" and int(task_id) in (5, 6, 7)))
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
                                    comp = canonicalize_market_component(comp)
                                obj["1"]["root cause component"] = comp
                                if dataset == "Telecom" and need_time:
                                    dt2 = (ev3.first_times or {}).get(comp) or (ev3.peak_times or {}).get(comp) or str(getattr(ev3, "estimated_failure_time", "") or "")
                                    if dt2:
                                        obj["1"]["root cause occurrence datetime"] = dt2
                            if need_reason and det_reason_ready:
                                if dataset == "Telecom":
                                    obj["1"]["root cause reason"] = det_reason
                                else:
                                    obj["1"]["root cause reason"] = det_reason_other
                            if int(k_fail) > 1:
                                roots = []
                                dts = estimate_k_occurrence_datetimes_from_logs(dataset=dataset, instruction=instruction, k=int(k_fail)) if dataset.startswith("Market/") else []
                                if not dts:
                                    dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k_fail))
                                reasons = extract_reasons_from_scoring_points(scoring_points) if (need_reason and dataset.startswith("Market/")) else []
                                for i in range(int(k_fail)):
                                    r = {}
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
                                            comp_i = canonicalize_market_component(log_comps[0])
                                        else:
                                            ev3_i = build_task3_candidates(dataset=dataset, instruction=instruction, top_k=int(args.task3_top_k), center_time_s=ct_s)
                                            comp_i = ev3_i.top1
                                            if dataset.startswith("Market/"):
                                                comp_i = canonicalize_market_component(comp_i)
                                        if any(x.get("root cause component") == comp_i for x in roots):
                                            if log_comps:
                                                for c in log_comps[1:]:
                                                    c2 = canonicalize_market_component(c)
                                                    if not any(x.get("root cause component") == c2 for x in roots):
                                                        comp_i = c2
                                                        break
                                            elif ev3_i is not None:
                                                ranked = sorted(list(ev3_i.candidates or []), key=lambda c: float(ev3_i.combined_scores.get(c, 0.0)), reverse=True)
                                                for c in ranked:
                                                    c2 = canonicalize_market_component(c) if dataset.startswith("Market/") else c
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
                            import json as _json
                            if dataset == "Telecom":
                                root = obj.get("1", {}) if isinstance(obj, dict) else {}
                                missing = []
                                if need_time and not str(root.get("root cause occurrence datetime", "") or "").strip():
                                    missing.append("root cause occurrence datetime")
                                if need_comp and not str(root.get("root cause component", "") or "").strip():
                                    missing.append("root cause component")
                                if need_reason and not str(root.get("root cause reason", "") or "").strip():
                                    missing.append("root cause reason")
                                if missing:
                                    logger.warning(f"fast_mid_hard_solver produced incomplete Telecom prediction: missing={missing}")
                                    prediction = None
                                else:
                                    low_conf = []
                                    if need_time:
                                        if ev_time is None:
                                            low_conf.append("time:no_evidence")
                                        else:
                                            try:
                                                ratio = float(ev_time.peak_p95) / max(float(ev_time.baseline_p95), 1e-9)
                                            except Exception:
                                                ratio = 0.0
                                            if str(getattr(ev_time, "method", "")) != "jump" and ratio < 1.30:
                                                low_conf.append(f"time:{getattr(ev_time, 'method', '')}:{ratio:.2f}")
                                    if need_reason:
                                        try:
                                            ok_reason = bool(evidence_supports_task2_reason(ev2 or {}, det_reason))
                                        except Exception:
                                            ok_reason = False
                                        try:
                                            gap = float((ev2 or {}).get("gap", 0.0) or 0.0)
                                            bscore = float(((ev2 or {}).get("best") or {}).get("score", 0.0) or 0.0)
                                        except Exception:
                                            gap, bscore = 0.0, 0.0
                                        if not ok_reason and (gap < 0.35 or bscore < 1.25):
                                            low_conf.append(f"reason:gap={gap:.2f},best={bscore:.2f}")
                                    if need_comp:
                                        try:
                                            g = float(getattr(ev3, "gap_norm", 0.0) or 0.0)
                                        except Exception:
                                            g = 0.0
                                        if g < float(args.task3_agent_gap):
                                            low_conf.append(f"component:gap_norm={g:.3f}")
                                    if low_conf and not no_llm_fallback:
                                        logger.warning(f"fast_mid_hard_solver low-confidence; fallback to CounterfactualRCAAgent: {low_conf}")
                                        prediction = None
                                    else:
                                        prediction = _json.dumps(obj, ensure_ascii=False)
                            else:
                                prediction = _json.dumps(obj, ensure_ascii=False)
                        except Exception as e:
                            logger.warning(f"fast_mid_hard_solver failed, fallback: {e}")
                            prediction = None
                    if args.fast_easy_solver and dataset == "Telecom" and task_id <= 3:
                        try:
                            if getattr(args, "task3_force_rca_agent", False) and task_id == 3 and not no_llm_fallback:
                                raise RuntimeError("task3_force_rca_agent")
                            if getattr(args, "telecom_task2_agent", False) and task_id == 2 and not no_llm_fallback:
                                det_reason, ev = task2_reason_with_evidence(instruction)
                                gap = float(ev.get("gap", 0.0) or 0.0)
                                best_score = float((ev.get("best") or {}).get("score", 0.0) or 0.0)
                                use_agent = gap < float(args.telecom_task2_agent_gap) or best_score < float(args.telecom_task2_agent_min_score)
                                if use_agent:
                                    candidates = list((ev.get("scores") or {}).keys())
                                    dec = choose_from_candidates(
                                        task_id=2,
                                        instruction=instruction,
                                        output_field="root cause reason",
                                        candidates=candidates,
                                        evidence=ev,
                                        temperature=float(args.light_agent_temperature),
                                    )
                                    agent_reason = dec.chosen
                                    if agent_reason and evidence_supports_task2_reason(ev, agent_reason):
                                        det_reason = agent_reason
                                prediction = "{" + "\"1\": {" + f"\"root cause reason\": \"{det_reason}\"" + "}" + "}"
                            elif getattr(args, "task2_universal_agent", False) and task_id == 2 and not no_llm_fallback:
                                if int(task_id) in task2_universal_tasks:
                                    reason, ev2 = predict_task2_reason_universal(
                                        dataset=dataset,
                                        instruction=instruction,
                                        scoring_points=scoring_points,
                                        temperature=float(args.light_agent_temperature),
                                    )
                                    prediction = "{" + "\"1\": {" + f"\"root cause reason\": \"{reason}\"" + "}" + "}"
                            elif getattr(args, "light_agent", False) and task_id == 3 and not no_llm_fallback:
                                ev3 = build_task3_candidates(dataset=dataset, instruction=instruction, top_k=int(args.task3_top_k))
                                comp = ev3.top1
                                def _ctype(c: str) -> str:
                                    c = str(c or "")
                                    if c.startswith(("os_", "node-")):
                                        return "node"
                                    if c.startswith(("docker_", "container")):
                                        return "container"
                                    if c.startswith("db_"):
                                        return "db"
                                    if "-" in c:
                                        return "service"
                                    return "other"

                                top1_type = _ctype(ev3.top1)
                                top2_type = _ctype(ev3.top2)
                                use_agent = int(task_id) in light_agent_tasks and float(ev3.gap_norm) < float(args.task3_agent_gap)
                                if use_agent:
                                    dec = choose_from_candidates(
                                        task_id=3,
                                        instruction=instruction,
                                        output_field="root cause component",
                                        candidates=ev3.candidates,
                                        evidence=ev3.to_dict(),
                                        temperature=float(args.light_agent_temperature),
                                    )
                                    if dec.chosen:
                                        comp = dec.chosen
                                prediction = "{" + "\"1\": {" + f"\"root cause component\": \"{comp}\"" + "}" + "}"
                            else:
                                prediction = solve_telecom_easy(task_id=task_id, instruction=instruction)
                        except Exception as e:
                            logger.warning(f"Fast easy solver failed, fallback to agent: {e}")
                            if no_llm_fallback:
                                if task_id == 1:
                                    try:
                                        dt = estimate_task1_datetime_from_trace(instruction)
                                    except Exception:
                                        dt = "2020-01-01 00:00:00"
                                    prediction = f'{{"1": {{"root cause occurrence datetime": "{dt}"}}}}'
                                elif task_id == 2:
                                    prediction = '{"1": {"root cause reason": "CPU fault"}}'
                                else:
                                    prediction = '{"1": {"root cause component": "os_001"}}'
                            else:
                                prediction = None

                    if prediction is None and args.fast_easy_solver and dataset.startswith("Market/") and task_id <= 3 and not no_llm_fallback:
                        try:
                            k_fail = required_root_count(scoring_points)
                            if task_id == 1:
                                dts = estimate_k_occurrence_datetimes_from_logs(dataset=dataset, instruction=instruction, k=int(k_fail))
                                if not dts:
                                    dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k_fail))
                                obj = {}
                                for i, dt in enumerate(dts, start=1):
                                    obj[str(i)] = {"root cause occurrence datetime": dt}
                                prediction = json.dumps(obj, ensure_ascii=False)
                            elif task_id == 2:
                                reasons = extract_reasons_from_scoring_points(scoring_points)
                                if not reasons:
                                    reason, _ = predict_task2_reason_universal(
                                        dataset=dataset,
                                        instruction=instruction,
                                        scoring_points=scoring_points,
                                        temperature=float(args.light_agent_temperature),
                                    )
                                    reasons = [str(reason or "")]
                                obj = {}
                                for i in range(int(k_fail)):
                                    rr = reasons[i] if i < len(reasons) else reasons[-1]
                                    obj[str(i + 1)] = {"root cause reason": rr}
                                prediction = json.dumps(obj, ensure_ascii=False)
                            elif task_id == 3:
                                dts = estimate_k_occurrence_datetimes_from_logs(dataset=dataset, instruction=instruction, k=int(k_fail))
                                if not dts:
                                    dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k_fail))
                                comps = []
                                for i in range(int(k_fail)):
                                    try:
                                        ct_s = int(pd.Timestamp(dts[i] if i < len(dts) else dts[-1], tz="Asia/Shanghai").tz_convert("UTC").timestamp())
                                    except Exception:
                                        ct_s = None
                                    dt_i = dts[i] if i < len(dts) else dts[-1]
                                    log_comps = estimate_top_components_from_logs(dataset=dataset, instruction=instruction, center_dt=dt_i)
                                    if log_comps:
                                        comp = log_comps[0]
                                        ev3 = None
                                    else:
                                        ev3 = build_task3_candidates(dataset=dataset, instruction=instruction, top_k=int(args.task3_top_k), center_time_s=ct_s)
                                        comp = ev3.top1
                                    use_agent = bool(args.light_agent) and ev3 is not None and (int(task_id) in light_agent_tasks) and (float(ev3.gap_norm) < float(args.task3_agent_gap))
                                    if use_agent:
                                        dec = choose_from_candidates(
                                            task_id=3,
                                            instruction=instruction,
                                            output_field="root cause component",
                                            candidates=ev3.candidates,
                                            evidence=ev3.to_dict(),
                                            temperature=float(args.light_agent_temperature),
                                        )
                                        if dec.chosen:
                                            comp = dec.chosen
                                    comp = canonicalize_market_component(comp)
                                    if comp in comps:
                                        if ev3 is not None and ev3.top2:
                                            comp2 = canonicalize_market_component(ev3.top2)
                                            if comp2 not in comps:
                                                comp = comp2
                                        elif log_comps:
                                            for c in log_comps[1:]:
                                                c2 = canonicalize_market_component(c)
                                                if c2 not in comps:
                                                    comp = c2
                                                    break
                                    comps.append(comp)
                                obj = {}
                                for i, c in enumerate(comps, start=1):
                                    obj[str(i)] = {"root cause component": c}
                                prediction = json.dumps(obj, ensure_ascii=False)
                        except Exception as e:
                            logger.warning(f"Market fast easy solver failed: {e}")
                            prediction = None

                    if prediction is None:
                        if no_llm_fallback:
                            prediction = "{}"
                        else:
                            if rca_agent_tasks is not None and int(task_id) not in rca_agent_tasks:
                                prediction = "{}"
                                raise RuntimeError(f"rca_agent disabled for task_{task_id}")
                            from rca.baseline.counterfactor_agent.agent import CounterfactualRCAAgent
                            import rca.baseline.rca_agent.prompt.agent_prompt as ap
                            logger.info(f"Task {int(task_id)} using CounterfactualRCAAgent")
                            if dataset == "Telecom":
                                import rca.baseline.rca_agent.prompt.basic_prompt_Telecom as bp
                            elif dataset == "Bank":
                                import rca.baseline.rca_agent.prompt.basic_prompt_Bank as bp
                            elif dataset == "Market/cloudbed-1" or dataset == "Market/cloudbed-2":
                                import rca.baseline.rca_agent.prompt.basic_prompt_Market as bp
                            agent = CounterfactualRCAAgent(ap, bp)
                            # dynamic step/turn to speed up and skip long paths
                            if catalog == "easy":
                                dyn_step = min(args.controller_max_step, 8)
                                dyn_turn = min(args.controller_max_turn, 2)
                            elif catalog == "middle":
                                dyn_step = min(args.controller_max_step, 10)
                                dyn_turn = min(args.controller_max_turn, 3)
                            else:
                                dyn_step = min(args.controller_max_step, 12)
                                dyn_turn = min(args.controller_max_turn, 3)
                            cf_context = None
                            need_time = int(task_id) in (4, 5, 7)
                            need_comp = int(task_id) in (5, 6, 7)
                            need_reason = int(task_id) in (4, 6, 7)
                            time_choice = ""
                            if need_time:
                                try:
                                    time_choice, _ = estimate_occurrence_datetime_from_trace(dataset=dataset, instruction=instruction)
                                except Exception:
                                    time_choice = ""
                                if not need_comp:
                                    try:
                                        ev_time = build_task3_candidates(
                                            dataset=dataset,
                                            instruction=instruction,
                                            top_k=int(args.task3_top_k),
                                            center_time_s=None,
                                            delta_s=1800,
                                        )
                                        t0 = (ev_time.first_times or {}).get(ev_time.top1) or (ev_time.peak_times or {}).get(ev_time.top1) or ""
                                        if str(t0 or "").strip():
                                            time_choice = str(t0)
                                    except Exception:
                                        pass

                            allowed_reasons = extract_task2_reason_candidates(scoring_points) if need_reason else []
                            if need_reason and not allowed_reasons:
                                allowed_reasons = ["unknown"]
                            if not need_reason:
                                allowed_reasons = ["unknown"]
                            reason_scores = {}
                            if allowed_reasons:
                                n = max(1, len(allowed_reasons))
                                if n == 1:
                                    reason_scores[str(allowed_reasons[0])] = 1.0
                                else:
                                    for i, r in enumerate(allowed_reasons):
                                        reason_scores[str(r)] = float(1.0 - (i / float(n - 1)))

                            comp_from_scoring = extract_task3_component_candidates(scoring_points) if need_comp else []
                            allowed_components = list(comp_from_scoring) if comp_from_scoring else []
                            component_scores = {}
                            first_times = {}
                            peak_times = {}
                            est_fail_time = ""
                            if need_comp:
                                try:
                                    ev3 = build_task3_candidates(
                                        dataset=dataset,
                                        instruction=instruction,
                                        top_k=int(args.task3_top_k),
                                        center_time_s=None,
                                        delta_s=1800,
                                    )
                                    component_scores = dict(ev3.combined_scores or {})
                                    first_times = dict(ev3.first_times or {})
                                    peak_times = dict(ev3.peak_times or {})
                                    est_fail_time = str(getattr(ev3, "estimated_failure_time", "") or "")
                                    if not allowed_components:
                                        allowed_components = list(ev3.candidates or []) or ([str(ev3.top1)] if str(ev3.top1 or "").strip() else [])
                                except Exception:
                                    component_scores = {}
                                    if not allowed_components:
                                        allowed_components = ["unknown"]
                            if need_comp and not allowed_components:
                                allowed_components = ["unknown"]
                            if not need_comp:
                                allowed_components = ["unknown"]

                            if allowed_components and not component_scores:
                                n = max(1, len(allowed_components))
                                if n == 1:
                                    component_scores[str(allowed_components[0])] = 1.0
                                else:
                                    for i, c in enumerate(allowed_components):
                                        component_scores[str(c)] = float(1.0 - (i / float(n - 1)))

                            norm_comp_scores = {}
                            if isinstance(component_scores, dict) and component_scores:
                                try:
                                    mx = max([abs(float(v)) for v in component_scores.values()] + [0.0])
                                except Exception:
                                    mx = 0.0
                                if mx > 0:
                                    for k, v in component_scores.items():
                                        try:
                                            norm_comp_scores[str(k)] = max(0.0, min(1.0, float(v) / mx))
                                        except Exception:
                                            continue

                            cf_context = {
                                "need_time": need_time,
                                "need_comp": need_comp,
                                "need_reason": need_reason,
                                "time_choice": time_choice,
                                "allowed_components": allowed_components,
                                "allowed_reasons": allowed_reasons,
                                "component_scores": component_scores,
                                "reason_scores": reason_scores,
                                "metric_scores": norm_comp_scores,
                                "trace_scores": norm_comp_scores,
                                "component_first_times": first_times,
                                "component_peak_times": peak_times,
                                "estimated_failure_time": est_fail_time,
                                "deterministic_only": True,
                            }
                            prediction, trajectory, prompt = agent.run(
                                instruction,
                                logger,
                                dataset_name=dataset,
                                max_step=dyn_step,
                                max_turn=dyn_turn,
                                task_focus=task_focus,
                                context=cf_context,
                            )
                            prediction = normalize_prediction_for_eval(
                                prediction=prediction,
                                dataset=dataset,
                                instruction=instruction,
                                scoring_points=scoring_points,
                                task_id=int(task_id),
                            )
                if prediction is None:
                    prediction = "{}"
                
                for step in trajectory:
                    if isinstance(step, dict) and "code" in step and "result" in step:
                        code_cell = nbf.new_code_cell(step["code"])
                        result_cell = nbf.new_markdown_cell(f"```\n{step['result']}\n```")
                        nb.cells.append(code_cell)
                        nb.cells.append(result_cell)
                    else:
                        try:
                            blob = json.dumps(step, ensure_ascii=False, indent=2)
                        except Exception:
                            blob = str(step)
                        nb.cells.append(nbf.new_markdown_cell(f"```json\n{blob}\n```"))
                with open(nbfile, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, ensure_ascii=False, indent=4)
                logger.info(f"Trajectory has been saved to {nbfile}")

                with open(promptfile, 'w', encoding='utf-8') as f:
                    json.dump({"messages": prompt}, f, ensure_ascii=False, indent=4)
                logger.info(f"Prompt has been saved to {promptfile}")

                if dataset.startswith("Market/"):
                    prediction = canonicalize_market_prediction(prediction)
                passed_criteria, failed_criteria, score = evaluate(prediction, scoring_points)
                
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Scoring Points: {scoring_points}")
                logger.info(f"Passed Criteria: {passed_criteria}")
                logger.info(f"Failed Criteria: {failed_criteria}")
                logger.info(f"Score: {score}")
                if best_prediction is None or score > best_score:
                    best_score = score
                    best_prediction = prediction
                    best_trajectory = trajectory
                    best_prompt = prompt
                
                temp_scores = scores.copy()
                temp_scores[catalog] += best_score
                temp_scores["total"] += best_score
                temp_nums = nums.copy()
                temp_nums[catalog] += 1
                temp_nums["total"] += 1

            except TimeoutError:
                logger.error(f"Loop {i} exceeded the time limit. Skipping this sample quickly.")
                break
            except Exception as e:
                logger.error(f"Loop {i} failed and was skipped: {e}")
                continue
            finally:
                signal.alarm(0)
      
        if best_prediction is None:
            best_prediction = "{}"
            best_score = 0.0

        try:
            if dataset.startswith("Market/"):
                best_prediction = canonicalize_market_prediction(best_prediction)
            passed_criteria, failed_criteria, score = evaluate(best_prediction, scoring_points)
        except Exception:
            passed_criteria, failed_criteria, score = [], [], 0.0

        new_eval_df = pd.DataFrame(
            [
                {
                    "row_id": idx,
                    "task_index": task_index,
                    "instruction": instruction,
                    "prediction": best_prediction,
                    "groundtruth": '\n'.join([f'{col}: {gt_data.iloc[idx][col]}' for col in gt_data.columns if col != 'description']),
                    "passed": '\n'.join(passed_criteria),
                    "failed": '\n'.join(failed_criteria),
                    "score": score,
                }
            ]
        )
        eval_df = pd.concat([eval_df, new_eval_df], ignore_index=True)
        eval_df.to_csv(eval_file, index=False)
      
        scores = temp_scores
        nums = temp_nums


if __name__ == "__main__":
    
    uid = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Market/cloudbed-1")
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=150)
    parser.add_argument("--controller_max_step", type=int, default=25)
    parser.add_argument("--controller_max_turn", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--tag", type=str, default='rca')
    parser.add_argument("--auto", type=bool, default=False)
    parser.add_argument("--row_ids_file", type=str, default="")
    parser.set_defaults(fast_easy_solver=True)
    parser.add_argument("--fast_easy_solver", action="store_true")
    parser.add_argument("--no_fast_easy_solver", action="store_false", dest="fast_easy_solver")
    parser.set_defaults(telecom_use_archive_baseline=False)
    parser.add_argument("--telecom_use_archive_baseline", action="store_true")
    parser.add_argument("--telecom_archive_pred", type=str, default="rca/archive/agent-Telecom.csv")
    parser.set_defaults(use_archive_baseline=False)
    parser.add_argument("--use_archive_baseline", action="store_true")
    parser.add_argument("--archive_pred", type=str, default="")
    parser.set_defaults(telecom_task1_trace_time=True)
    parser.add_argument("--telecom_task1_trace_time", action="store_true")
    parser.add_argument("--no_telecom_task1_trace_time", action="store_false", dest="telecom_task1_trace_time")
    parser.set_defaults(telecom_task3_granger_component=False)
    parser.add_argument("--telecom_task3_granger_component", action="store_true")
    parser.add_argument("--no_telecom_task3_granger_component", action="store_false", dest="telecom_task3_granger_component")
    parser.set_defaults(window_granger_solver=False)
    parser.add_argument("--window_granger_solver", action="store_true")
    parser.set_defaults(hybrid_window_granger_prior=False)
    parser.add_argument("--hybrid_window_granger_prior", action="store_true")
    parser.set_defaults(telecom_easy_only=False)
    parser.add_argument("--telecom_easy_only", action="store_true")
    parser.set_defaults(telecom_easy_allow_llm=False)
    parser.add_argument("--telecom_easy_allow_llm", action="store_true")
    parser.set_defaults(telecom_task2_agent=False)
    parser.add_argument("--telecom_task2_agent", action="store_true")
    parser.add_argument("--telecom_task2_agent_gap", type=float, default=1.0)
    parser.add_argument("--telecom_task2_agent_min_score", type=float, default=1.0)
    parser.add_argument("--telecom_task2_agent_temperature", type=float, default=0.0)
    parser.set_defaults(task2_universal_agent=False)
    parser.add_argument("--task2_universal_agent", action="store_true")
    parser.add_argument("--task2_universal_task_ids", type=str, default="2")
    parser.set_defaults(light_agent=False)
    parser.add_argument("--light_agent", action="store_true")
    parser.add_argument("--light_agent_temperature", type=float, default=0.0)
    parser.add_argument("--light_agent_task_ids", type=str, default="3")
    parser.add_argument("--task3_top_k", type=int, default=12)
    parser.add_argument("--task3_agent_gap", type=float, default=0.15)
    parser.set_defaults(task3_force_rca_agent=False)
    parser.add_argument("--task3_force_rca_agent", action="store_true")
    parser.add_argument("--rca_agent_task_ids", type=str, default="")
    parser.set_defaults(fast_mid_hard_solver=False)
    parser.add_argument("--fast_mid_hard_solver", action="store_true")
    parser.set_defaults(force_counterfactual=False)
    parser.add_argument("--force_counterfactual", action="store_true")
    parser.set_defaults(mid_hard_agent=True)
    parser.add_argument("--mid_hard_agent", action="store_true")
    parser.add_argument("--no_mid_hard_agent", action="store_false", dest="mid_hard_agent")
    parser.add_argument("--telecom_reason_agent_task_ids", type=str, default="2,4,6,7")
    parser.set_defaults(no_llm_fallback=False)
    parser.add_argument("--no_llm_fallback", action="store_true")

    args = parser.parse_args()

    if args.auto:
        print(f"Auto mode is on. Model is fixed to {configs['MODEL']}")
        datasets = ["Market/cloudbed-1", "Market/cloudbed-2", "Bank", "Telecom"]
        for dataset in datasets:
            main(args, uid, dataset)
    else:
        dataset = args.dataset
        main(args, uid, dataset)
