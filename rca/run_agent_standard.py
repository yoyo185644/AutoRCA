import os
import sys
import json
import argparse
import time
from typing import Any, Dict, List, Optional
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from main.evaluate import evaluate
from rca.api_router import configs

from datetime import datetime
from loguru import logger
from nbformat import v4 as nbf
import pandas as pd
import signal
from rca.telemetry_solver import canonicalize_prediction
from rca.execution.agent_standard import AgentStandardRuntimeExtras, build_agent_standard_runner
from rca.execution.run_agent_standard_flow import get_task_spec, solve_by_mode

import re

_COMP_PATTERN = r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
_REASON_PATTERN = r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
_TIME_PATTERN = r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)"


def _split_lines(text: Any) -> List[str]:
    if isinstance(text, (list, tuple, set)):
        return [str(x).strip() for x in text if str(x).strip()]
    return [x.strip() for x in str(text or "").splitlines() if str(x).strip()]


def _extract_expected(scoring_points: str) -> Dict[str, List[str]]:
    sp = str(scoring_points or "")
    return {
        "time": [str(x).strip() for x in re.findall(_TIME_PATTERN, sp)],
        "component": [str(x).strip() for x in re.findall(_COMP_PATTERN, sp)],
        "reason": [str(x).strip() for x in re.findall(_REASON_PATTERN, sp)],
    }

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


def extract_timing_breakdown(trajectory):
    if not isinstance(trajectory, list):
        return {}
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        if step.get("action") == "timing" and isinstance(step.get("timing_breakdown_seconds"), dict):
            return dict(step.get("timing_breakdown_seconds") or {})
    return {}


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


def _parse_csv_set(spec: str):
    s = str(spec or "").strip()
    if not s:
        return None
    out = {part.strip().lower() for part in s.split(",") if part.strip()}
    return out if out else None


def main(args, uid, dataset):
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
    eval_df = eval_df.drop(columns=["time_candidates"], errors="ignore")
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
    selected_task_ids = _parse_task_id_set(getattr(args, "task_ids", ""))
    selected_difficulties = _parse_csv_set(getattr(args, "difficulty", ""))
    archive_df = None
    if args.use_archive_baseline:
        archive_df = pd.read_csv(args.archive_pred)
        archive_df["row_id_int"] = pd.to_numeric(archive_df.get("row_id"), errors="coerce").fillna(-1).astype(int)
    no_llm_fallback = bool(args.no_llm_fallback or args.use_archive_baseline)
    light_agent_tasks = set([3])
    rca_agent_tasks = None
    fast_mid_hard = False
    agent_standard_runner = build_agent_standard_runner()
    agent_standard_extras = AgentStandardRuntimeExtras(
        light_agent_tasks=set(light_agent_tasks),
    )
    
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
        spec = get_task_spec(task_id)
        catalog = spec.difficulty
        if selected_task_ids is not None and task_id not in selected_task_ids:
            continue
        if selected_difficulties is not None and str(catalog).lower() not in selected_difficulties:
            continue
        best_score = 0
        best_prediction = None
        best_trajectory = None
        best_prompt = None
        best_solve_seconds = 0.0
        best_evaluate_seconds = 0.0
        best_sample_seconds = 0.0
        best_timing_breakdown = {}
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

                sample_start = time.perf_counter()
                solve_start = time.perf_counter()
                prediction, trajectory, prompt = solve_by_mode(
                    spec=spec,
                    args=args,
                    dataset=dataset,
                    idx=int(idx),
                    instruction=instruction,
                    scoring_points=scoring_points,
                    task_focus=task_focus,
                    logger=logger,
                    archive_df=archive_df,
                    no_llm_fallback=no_llm_fallback,
                    fast_mid_hard=fast_mid_hard,
                    light_agent_tasks=set(light_agent_tasks),
                    rca_agent_tasks=rca_agent_tasks,
                    agent_standard_runner=agent_standard_runner,
                    agent_standard_extras=agent_standard_extras,
                )
                solve_seconds = time.perf_counter() - solve_start
                trajectory = list(trajectory or [])
                trajectory.append(
                    {
                        "step": "timing",
                        "action": "solve_by_mode_timing",
                        "row_id": int(idx),
                        "sample_id": int(i),
                        "task_index": str(task_index),
                        "difficulty": str(catalog),
                        "solve_seconds": float(solve_seconds),
                    }
                )
                
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
                    prediction = canonicalize_prediction(prediction)
                eval_start = time.perf_counter()
                passed_criteria, failed_criteria, score = evaluate(prediction, scoring_points)
                evaluate_seconds = time.perf_counter() - eval_start
                sample_seconds = time.perf_counter() - sample_start
                
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Scoring Points: {scoring_points}")
                logger.info(f"Passed Criteria: {passed_criteria}")
                logger.info(f"Failed Criteria: {failed_criteria}")
                logger.info(f"Score: {score}")
                logger.info(
                    f"Timing: solve_seconds={solve_seconds:.6f}, "
                    f"evaluate_seconds={evaluate_seconds:.6f}, sample_seconds={sample_seconds:.6f}"
                )
                timing_breakdown = extract_timing_breakdown(trajectory)
                if timing_breakdown:
                    logger.info(
                        "Timing breakdown: "
                        f"candidate_generation_seconds={float(timing_breakdown.get('candidate_generation_seconds', 0.0)):.6f}, "
                        f"basil_evidence_build_seconds={float(timing_breakdown.get('basil_evidence_build_seconds', 0.0)):.6f}, "
                        f"basil_index_query_seconds={float(timing_breakdown.get('basil_index_query_seconds', 0.0)):.6f}, "
                        f"basil_candidate_decision_seconds={float(timing_breakdown.get('basil_candidate_decision_seconds', 0.0)):.6f}, "
                        f"agent_call_seconds={float(timing_breakdown.get('agent_call_seconds', 0.0)):.6f}, "
                        f"llm_verifier_seconds={float(timing_breakdown.get('llm_verifier_seconds', 0.0)):.6f}, "
                        f"llm_input_tokens={int(timing_breakdown.get('llm_input_tokens', 0) or 0)}, "
                        f"llm_time_verifier_seconds={float(timing_breakdown.get('llm_time_verifier_seconds', 0.0)):.6f}, "
                        f"llm_time_input_tokens={int(timing_breakdown.get('llm_time_input_tokens', 0) or 0)}, "
                        f"agent_total_input_tokens={int(timing_breakdown.get('agent_total_input_tokens', 0) or 0)}, "
                        f"time_candidate_count={int(timing_breakdown.get('time_candidate_count', 0) or 0)}, "
                        f"time_candidate_unique_minute_count={int(timing_breakdown.get('time_candidate_unique_minute_count', 0) or 0)}, "
                        f"time_candidate_spread_minutes={float(timing_breakdown.get('time_candidate_spread_minutes', 0.0) or 0.0):.2f}, "
                        f"total_execution_seconds={float(timing_breakdown.get('total_execution_seconds', 0.0)):.6f}"
                    )
                if best_prediction is None or score > best_score:
                    best_score = score
                    best_prediction = prediction
                    best_trajectory = trajectory
                    best_prompt = prompt
                    best_solve_seconds = float(solve_seconds)
                    best_evaluate_seconds = float(evaluate_seconds)
                    best_sample_seconds = float(sample_seconds)
                    best_timing_breakdown = dict(timing_breakdown or {})
                
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
                best_prediction = canonicalize_prediction(best_prediction)
            passed_criteria, failed_criteria, score = evaluate(best_prediction, scoring_points)
        except Exception:
            passed_criteria, failed_criteria, score = [], [], 0.0

        new_eval_df = pd.DataFrame(
            [
                {
                    "row_id": idx,
                    "task_index": task_index,
                    "task_id": int(task_id),
                    "instruction": instruction,
                    "scoring_points": scoring_points,
                    "prediction": best_prediction,
                    "groundtruth": '\n'.join([f'{col}: {gt_data.iloc[idx][col]}' for col in gt_data.columns if col != 'description']),
                    "passed": '\n'.join(passed_criteria),
                    "failed": '\n'.join(failed_criteria),
                    "score": score,
                    "exact_correct": bool(float(score or 0.0) >= 0.999),
                    "solve_seconds": best_solve_seconds,
                    "evaluate_seconds": best_evaluate_seconds,
                    "sample_seconds": best_sample_seconds,
                    "candidate_generation_seconds": float(best_timing_breakdown.get("candidate_generation_seconds", 0.0) or 0.0),
                    "basil_index_total_seconds": float(best_timing_breakdown.get("basil_index_total_seconds", 0.0) or 0.0),
                    "basil_evidence_extract_seconds": float(best_timing_breakdown.get("basil_evidence_extract_seconds", 0.0) or 0.0),
                    "basil_evidence_build_seconds": float(best_timing_breakdown.get("basil_evidence_build_seconds", 0.0) or 0.0),
                    "basil_index_query_seconds": float(best_timing_breakdown.get("basil_index_query_seconds", 0.0) or 0.0),
                    "basil_candidate_fusion_seconds": float(best_timing_breakdown.get("basil_candidate_fusion_seconds", 0.0) or 0.0),
                    "basil_candidate_expansion_seconds": float(best_timing_breakdown.get("basil_candidate_expansion_seconds", 0.0) or 0.0),
                    "basil_candidate_rerank_seconds": float(best_timing_breakdown.get("basil_candidate_rerank_seconds", 0.0) or 0.0),
                    "basil_candidate_decision_seconds": float(best_timing_breakdown.get("basil_candidate_decision_seconds", 0.0) or 0.0),
                    "basil_evidence_called": bool(best_timing_breakdown.get("basil_evidence_called", False)),
                    "basil_low_confidence": bool(best_timing_breakdown.get("basil_low_confidence", False)),
                    "basil_confidence_reason": str(best_timing_breakdown.get("basil_confidence_reason", "") or ""),
                    "basil_component_coverage": int(best_timing_breakdown.get("basil_component_coverage", 0) or 0),
                    "basil_time_coverage": int(best_timing_breakdown.get("basil_time_coverage", 0) or 0),
                    "basil_reason_coverage": int(best_timing_breakdown.get("basil_reason_coverage", 0) or 0),
                    "agent_call_seconds": float(best_timing_breakdown.get("agent_call_seconds", 0.0) or 0.0),
                    "llm_verifier_seconds": float(best_timing_breakdown.get("llm_verifier_seconds", 0.0) or 0.0),
                    "llm_input_tokens": int(best_timing_breakdown.get("llm_input_tokens", 0) or 0),
                    "llm_gate_called": bool(best_timing_breakdown.get("llm_gate_called", False)),
                    "llm_time_verifier_seconds": float(best_timing_breakdown.get("llm_time_verifier_seconds", 0.0) or 0.0),
                    "llm_time_input_tokens": int(best_timing_breakdown.get("llm_time_input_tokens", 0) or 0),
                    "llm_time_called": bool(best_timing_breakdown.get("llm_time_called", False)),
                    "agent_total_input_tokens": int(best_timing_breakdown.get("agent_total_input_tokens", 0) or 0),
                    "llm_called": bool(best_timing_breakdown.get("llm_gate_called", False) or best_timing_breakdown.get("llm_time_called", False)),
                    "postprocess_seconds": float(best_timing_breakdown.get("postprocess_seconds", 0.0) or 0.0),
                    "setup_seconds": float(best_timing_breakdown.get("setup_seconds", 0.0) or 0.0),
                    "component_candidate_seconds": float(best_timing_breakdown.get("component_candidate_seconds", 0.0) or 0.0),
                    "time_candidate_seconds": float(best_timing_breakdown.get("time_candidate_seconds", 0.0) or 0.0),
                    "time_candidate_count": int(best_timing_breakdown.get("time_candidate_count", 0) or 0),
                    "time_candidate_unique_datetime_count": int(best_timing_breakdown.get("time_candidate_unique_datetime_count", 0) or 0),
                    "time_candidate_unique_minute_count": int(best_timing_breakdown.get("time_candidate_unique_minute_count", 0) or 0),
                    "time_candidate_spread_seconds": float(best_timing_breakdown.get("time_candidate_spread_seconds", 0.0) or 0.0),
                    "time_candidate_spread_minutes": float(best_timing_breakdown.get("time_candidate_spread_minutes", 0.0) or 0.0),
                    "total_execution_seconds": float(best_timing_breakdown.get("total_execution_seconds", 0.0) or 0.0),
                    "time_expected": '\n'.join(_extract_expected(scoring_points).get("time", [])),
                    "component_expected": '\n'.join(_extract_expected(scoring_points).get("component", [])),
                    "reason_expected": '\n'.join(_extract_expected(scoring_points).get("reason", [])),
                    "time_correct": bool(set(_extract_expected(scoring_points).get("time", [])).intersection(set(_split_lines(passed_criteria)))),
                    "component_correct": bool(set(_extract_expected(scoring_points).get("component", [])).intersection(set(_split_lines(passed_criteria)))),
                    "reason_correct": bool(set(_extract_expected(scoring_points).get("reason", [])).intersection(set(_split_lines(passed_criteria)))),
                }
            ]
        )
        eval_df = pd.concat([eval_df, new_eval_df], ignore_index=True)
        eval_df.to_csv(eval_file, index=False)
      
        scores = temp_scores
        nums = temp_nums

    try:
        df = eval_df.copy()
        if "row_id" in df.columns:
            df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce")
        mask = pd.Series([True] * len(df))
        if "row_id" in df.columns:
            mask = mask & df["row_id"].notna()
            mask = mask & (df["row_id"] >= int(args.start_idx)) & (df["row_id"] <= int(args.end_idx))
        df = df[mask].copy()
        if df.empty:
            return

        total = int(len(df))
        llm_ratio = float(df["llm_called"].mean()) if "llm_called" in df.columns else 0.0
        exact_acc = float((df["score"] >= 0.999).mean()) if "score" in df.columns else 0.0

        llm_df = df[df.get("llm_called", False) == True].copy() if "llm_called" in df.columns else df.iloc[0:0].copy()
        det_df = df[df.get("llm_called", False) == False].copy() if "llm_called" in df.columns else df.copy()
        llm_acc = float((llm_df["score"] >= 0.999).mean()) if not llm_df.empty and "score" in llm_df.columns else 0.0
        det_acc = float((det_df["score"] >= 0.999).mean()) if not det_df.empty and "score" in det_df.columns else 0.0

        time_mask = df["time_expected"].astype(str).str.strip().ne("") if "time_expected" in df.columns else pd.Series([False] * len(df))
        comp_mask = df["component_expected"].astype(str).str.strip().ne("") if "component_expected" in df.columns else pd.Series([False] * len(df))
        reason_mask = df["reason_expected"].astype(str).str.strip().ne("") if "reason_expected" in df.columns else pd.Series([False] * len(df))

        def _acc_on(mask_s: pd.Series, col: str, sub: Optional[pd.DataFrame] = None) -> float:
            d = df if sub is None else sub
            if col not in d.columns:
                return 0.0
            m = mask_s.reindex(d.index).fillna(False)
            dd = d[m].copy()
            if dd.empty:
                return 0.0
            return float(dd[col].astype(bool).mean())

        time_acc = _acc_on(time_mask, "time_correct")
        time_acc_llm = _acc_on(time_mask, "time_correct", llm_df)
        time_acc_det = _acc_on(time_mask, "time_correct", det_df)

        comp_acc = _acc_on(comp_mask, "component_correct")
        comp_acc_llm = _acc_on(comp_mask, "component_correct", llm_df)
        comp_acc_det = _acc_on(comp_mask, "component_correct", det_df)

        reason_acc = _acc_on(reason_mask, "reason_correct")
        reason_acc_llm = _acc_on(reason_mask, "reason_correct", llm_df)
        reason_acc_det = _acc_on(reason_mask, "reason_correct", det_df)

        solve_avg = float(df["solve_seconds"].mean()) if "solve_seconds" in df.columns else 0.0
        solve_avg_llm = float(llm_df["solve_seconds"].mean()) if not llm_df.empty and "solve_seconds" in llm_df.columns else 0.0
        solve_avg_det = float(det_df["solve_seconds"].mean()) if not det_df.empty and "solve_seconds" in det_df.columns else 0.0
        evidence_build_avg = float(df["basil_evidence_build_seconds"].mean()) if "basil_evidence_build_seconds" in df.columns else 0.0
        evidence_query_avg = float(df["basil_index_query_seconds"].mean()) if "basil_index_query_seconds" in df.columns else 0.0
        candidate_decision_avg = float(df["basil_candidate_decision_seconds"].mean()) if "basil_candidate_decision_seconds" in df.columns else 0.0
        agent_call_avg = float(df["agent_call_seconds"].mean()) if "agent_call_seconds" in df.columns else 0.0

        token_cols = []
        if "llm_input_tokens" in df.columns:
            token_cols.append("llm_input_tokens")
        if "llm_time_input_tokens" in df.columns:
            token_cols.append("llm_time_input_tokens")
        if "agent_total_input_tokens" in df.columns:
            token_avg = float(df["agent_total_input_tokens"].mean())
            token_avg_llm = float(llm_df["agent_total_input_tokens"].mean()) if not llm_df.empty else 0.0
        else:
            token_avg = float(df[token_cols].sum(axis=1).mean()) if token_cols else 0.0
            token_avg_llm = float(llm_df[token_cols].sum(axis=1).mean()) if (token_cols and not llm_df.empty) else 0.0

        logger.info(
            "Run summary: "
            f"total={total}, "
            f"llm_trigger_ratio={llm_ratio:.3f}, "
            f"exact_acc={exact_acc:.3f}, "
            f"exact_acc_llm={llm_acc:.3f}, "
            f"exact_acc_det={det_acc:.3f}, "
            f"time_acc={time_acc:.3f} (llm={time_acc_llm:.3f}, det={time_acc_det:.3f}), "
            f"component_acc={comp_acc:.3f} (llm={comp_acc_llm:.3f}, det={comp_acc_det:.3f}), "
            f"reason_acc={reason_acc:.3f} (llm={reason_acc_llm:.3f}, det={reason_acc_det:.3f}), "
            f"solve_seconds_avg={solve_avg:.3f} (llm={solve_avg_llm:.3f}, det={solve_avg_det:.3f}), "
            f"evidence_build_seconds_avg={evidence_build_avg:.3f}, "
            f"index_query_seconds_avg={evidence_query_avg:.6f}, "
            f"candidate_decision_seconds_avg={candidate_decision_avg:.6f}, "
            f"agent_call_seconds_avg={agent_call_avg:.3f}, "
            f"llm_tokens_avg={token_avg:.1f} (llm_only_avg={token_avg_llm:.1f})"
        )
    except Exception as e:
        logger.warning(f"Run summary failed: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--difficulty", type=str, default="")
    parser.add_argument("--task_ids", type=str, default="")
    parser.add_argument("--row_ids_file", type=str, default="")
    parser.add_argument("--route_mode", type=str, default="auto")
    parser.add_argument("--time_candidate_top_k", type=int, default=8)
    parser.set_defaults(fast_easy_solver=True)
    parser.add_argument("--fast_easy_solver", action="store_true")
    parser.add_argument("--no_fast_easy_solver", action="store_false", dest="fast_easy_solver")
    parser.set_defaults(use_archive_baseline=False)
    parser.add_argument("--use_archive_baseline", action="store_true")
    parser.add_argument("--archive_pred", type=str, default="")
    parser.set_defaults(window_granger_solver=False)
    parser.add_argument("--window_granger_solver", action="store_true")
    parser.set_defaults(hybrid_window_granger_prior=False)
    parser.add_argument("--hybrid_window_granger_prior", action="store_true")
    parser.set_defaults(light_agent=False)
    parser.add_argument("--light_agent", action="store_true")
    parser.add_argument("--light_agent_temperature", type=float, default=0.0)
    parser.set_defaults(force_counterfactual=False)
    parser.add_argument("--force_counterfactual", action="store_true")
    parser.set_defaults(no_llm_fallback=False)
    parser.add_argument("--no_llm_fallback", action="store_true")
    parser.set_defaults(uncertainty_llm=False)
    parser.add_argument("--uncertainty_llm", action="store_true")
    parser.add_argument("--no_uncertainty_llm", action="store_false", dest="uncertainty_llm")
    parser.set_defaults(time_llm=False)
    parser.add_argument("--time_llm", action="store_true")
    parser.add_argument("--no_time_llm", action="store_false", dest="time_llm")
    parser.add_argument("--time_llm_top_k", type=int, default=3)
    parser.add_argument("--time_llm_max_score_drop", type=float, default=0.20)
    parser.add_argument("--time_llm_gap_threshold", type=float, default=0.08)
    parser.add_argument("--time_llm_confidence_floor", type=float, default=0.85)
    parser.add_argument("--time_llm_competing_drop", type=float, default=0.12)
    parser.set_defaults(time_llm_force=False)
    parser.add_argument("--time_llm_force", action="store_true")
    parser.add_argument("--no_time_llm_force", action="store_false", dest="time_llm_force")
    parser.set_defaults(llm_force_midhard=False)
    parser.add_argument("--llm_force_midhard", action="store_true")
    parser.add_argument("--no_llm_force_midhard", action="store_false", dest="llm_force_midhard")
    parser.set_defaults(deterministic_only=False)
    parser.add_argument("--deterministic_only", action="store_true")
    parser.add_argument("--llm_uncertain_gap_threshold", type=float, default=1.0)
    parser.add_argument("--llm_confidence_floor", type=float, default=1.0)
    parser.add_argument("--llm_time_confidence_floor", type=float, default=1.0)
    parser.add_argument("--llm_verify_top_k", type=int, default=3)
    parser.add_argument("--evidence_score_gap_threshold", type=float, default=0.15)
    parser.add_argument("--evidence_top1_score_floor", type=float, default=0.30)
    parser.add_argument("--evidence_time_gap_threshold", type=float, default=0.20)
    parser.add_argument("--evidence_time_conflict_seconds", type=int, default=300)
    parser.add_argument("--gate_skip_gap_threshold", type=float, default=0.3)
    parser.set_defaults(basil_index=False)
    parser.add_argument("--basil_index", action="store_true")
    parser.add_argument("--basil_bucket_size", type=int, default=300)
    parser.add_argument("--basil_block_size", type=int, default=32)
    parser.add_argument("--basil_max_lag", type=int, default=900)
    parser.add_argument("--basil_severity_floor", type=float, default=0.0)
    parser.add_argument("--basil_w_local", type=float, default=1.0)
    parser.add_argument("--basil_w_causal", type=float, default=0.45)
    parser.add_argument("--basil_w_reason", type=float, default=0.25)
    parser.add_argument("--basil_approx_epsilon", type=float, default=0.0)
    parser.add_argument("--basil_max_trace_rows", type=int, default=0)
    parser.set_defaults(basil_persist_index=True)
    parser.add_argument("--basil_persist_index", action="store_true")
    parser.add_argument("--no_basil_persist_index", action="store_false", dest="basil_persist_index")
    parser.add_argument("--basil_index_cache_dir", type=str, default="test/cache/basil_index")
    parser.add_argument("--basil_similarity_ts_weight", type=float, default=0.45)
    parser.add_argument("--basil_similarity_topology_weight", type=float, default=0.30)
    parser.add_argument("--basil_similarity_pattern_weight", type=float, default=0.25)
    parser.add_argument("--basil_history_decay_lambda", type=float, default=0.03)
    parser.add_argument("--basil_historical_top_k", type=int, default=5)
    parser.add_argument("--basil_min_evidence_threshold", type=int, default=2)
    parser.add_argument(
        "--basil_task_type",
        type=str,
        default="AUTO",
        choices=[
            "AUTO",
            "TIME_ONLY",
            "REASON_ONLY",
            "COMPONENT_ONLY",
            "REASON_TIME",
            "COMPONENT_TIME",
            "COMPONENT_REASON",
            "COMPONENT_REASON_TIME",
        ],
    )
    parser.add_argument("--basil_rerank_top_n", type=int, default=20)
    parser.add_argument("--basil_candidate_reason_top_r", type=int, default=3)
    parser.add_argument("--basil_time_tolerance_s", type=int, default=300)
    return parser


def cli_main(argv=None) -> int:
    uid = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args = build_arg_parser().parse_args(argv)

    if args.auto:
        print(f"Auto mode is on. Model is fixed to {configs['MODEL']}")
        datasets = ["Market/cloudbed-1", "Market/cloudbed-2", "Bank", "Telecom"]
        for dataset in datasets:
            main(args, uid, dataset)
    else:
        dataset = args.dataset
        main(args, uid, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
