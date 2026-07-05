from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)

from rca.agent.agent_standard import AgentStandardRuntimeExtras, build_agent_standard_runner
from rca.execution.run_agent_standard_flow import get_task_spec, solve_by_mode


def evaluate(prediction: str, scoring_points: str):
    """
    Evaluate single JSON-like prediction with corresponding scoring points
        args:
            prediction: str, the prediction (JSON-like string)
            scoring_points: str, the scoring points string
    """

    import itertools

    predict_pattern = (
        r'{\s*'
        r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
        r'(?:"root cause component":\s*"(.*?)")?,?\s*'
        r'(?:"root cause reason":\s*"(.*?)")?\s*}'
    )

    predict_matches = re.findall(predict_pattern, prediction)

    predict_results = []

    for match in predict_matches:
        datetime_str, component, reason = match
        predict_results.append({
            "root cause occurrence datetime": datetime_str,
            "root cause component": component,
            "root cause reason": reason
        })

    prediction_length = len(predict_results)

    component_pattern = r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
    reason_pattern = r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
    time_pattern = r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)"

    components = re.findall(component_pattern, scoring_points)
    reasons = re.findall(reason_pattern, scoring_points)
    times = re.findall(time_pattern, scoring_points)

    scoringpoints_length = max(len(components), len(reasons), len(times))
    socres_num = len(components) + len(reasons) + len(times)

    def time_difference(time1_str, time2_str):
        from datetime import datetime
        time_format = "%Y-%m-%d %H:%M:%S"

        try:
            time1 = datetime.strptime(time1_str, time_format)
            time2 = datetime.strptime(time2_str, time_format)
        except ValueError:
            return False

        time_difference = abs(time1 - time2)
        if time_difference.total_seconds() <= 60:
            return True
        else:
            return False

    scores_get = 0
    passing_criteria = []
    failing_criteria = []

    if scoringpoints_length == prediction_length:
        best_sore = -1
        for perm in itertools.permutations(predict_results):
            current_score = 0
            current_passing = []
            for i in range(scoringpoints_length):
                if len(components) == scoringpoints_length:
                    if perm[i]['root cause component'] == components[i]:
                        current_score += 1
                        current_passing.append(components[i])
                if len(reasons) == scoringpoints_length:
                    if perm[i]['root cause reason'] == reasons[i]:
                        current_score += 1
                        current_passing.append(reasons[i])
                if len(times) == scoringpoints_length:
                    if time_difference(times[i], perm[i]['root cause occurrence datetime']):
                        current_score += 1
                        current_passing.append(times[i])
            if current_score > best_sore:
                best_sore = current_score
                passing_criteria = current_passing
        scores_get = best_sore

    failing_criteria = list(set(components + reasons + times) - set(passing_criteria))

    final_score = scores_get / socres_num
    bin_score = round(final_score, 2)
    return passing_criteria, failing_criteria, bin_score


def _infer_task_id(task_index: Any) -> int:
    try:
        return int(str(task_index).split("_")[1])
    except Exception:
        return 0


def _task_focus(task_id: int) -> str:
    return {
        1: "Only the root cause occurrence datetime is evaluated.",
        2: "Only the root cause reason is evaluated.",
        3: "Only the root cause component is evaluated.",
        4: "The root cause occurrence datetime and root cause reason are evaluated.",
        5: "The root cause occurrence datetime and root cause component are evaluated.",
        6: "The root cause component and root cause reason are evaluated.",
        7: "The root cause occurrence datetime, component, and reason are all evaluated.",
    }.get(int(task_id), "Follow the instruction and identify the required root cause fields.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Telecom")
    p.add_argument("--difficulty", type=str, default="")
    p.add_argument("--task_ids", type=str, default="")
    p.add_argument("--row_ids", type=str, default="")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=999999)
    p.add_argument("--sample_num", type=int, default=1)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--tag", type=str, default="agent_standard")
    p.add_argument("--route_mode", type=str, default="auto")
    p.set_defaults(fast_easy_solver=True)
    p.add_argument("--fast_easy_solver", action="store_true")
    p.add_argument("--no_fast_easy_solver", action="store_false", dest="fast_easy_solver")
    p.set_defaults(uncertainty_llm=False)
    p.add_argument("--uncertainty_llm", action="store_true")
    p.add_argument("--no_uncertainty_llm", action="store_false", dest="uncertainty_llm")
    p.set_defaults(time_llm=False)
    p.add_argument("--time_llm", action="store_true")
    p.add_argument("--no_time_llm", action="store_false", dest="time_llm")
    p.add_argument("--time_llm_top_k", type=int, default=3)
    p.add_argument("--time_candidate_top_k", type=int, default=8)
    p.add_argument("--evidence_score_gap_threshold", type=float, default=0.15)
    p.add_argument("--evidence_top1_score_floor", type=float, default=0.30)
    p.add_argument("--evidence_top1_score_floor_single", type=float, default=0.50)
    p.add_argument("--gate_skip_gap_threshold", type=float, default=0.30)
    p.add_argument("--llm_change_min_confidence", type=float, default=0.85)
    p.add_argument("--llm_max_score_drop_vs_det_top1", type=float, default=0.05)
    p.set_defaults(adaptive_profile=False)
    p.add_argument("--adaptive_profile", action="store_true")
    p.add_argument("--confidence_level", type=float, default=0.95)
    p.add_argument("--evidence_budget", type=str, default="medium", choices=["small", "medium", "large", "fast", "accurate"])
    p.add_argument("--history_weight", type=float, default=0.0)
    p.add_argument("--llm_risk_tolerance", type=str, default="conservative", choices=["conservative", "balanced", "aggressive"])
    p.add_argument("--adaptive_profile_max_files", type=int, default=80)
    p.add_argument("--adaptive_profile_max_rows_per_file", type=int, default=200000)
    p.set_defaults(basil_index=False)
    p.add_argument("--basil_index", action="store_true")
    p.add_argument("--basil_bucket_size", type=int, default=300)
    p.add_argument("--basil_block_size", type=int, default=32)
    p.add_argument("--basil_max_lag", type=int, default=900)
    p.add_argument("--basil_severity_floor", type=float, default=0.0)
    p.add_argument("--basil_w_local", type=float, default=1.0)
    p.add_argument("--basil_w_causal", type=float, default=0.45)
    p.add_argument("--basil_w_reason", type=float, default=0.25)
    p.add_argument("--basil_approx_epsilon", type=float, default=0.0)
    p.add_argument("--basil_max_trace_rows", type=int, default=0)
    p.set_defaults(basil_persist_index=True)
    p.add_argument("--basil_persist_index", action="store_true")
    p.add_argument("--no_basil_persist_index", action="store_false", dest="basil_persist_index")
    p.add_argument("--basil_index_cache_dir", type=str, default="test/cache/basil_index")
    p.add_argument("--basil_similarity_ts_weight", type=float, default=0.45)
    p.add_argument("--basil_similarity_topology_weight", type=float, default=0.30)
    p.add_argument("--basil_similarity_pattern_weight", type=float, default=0.25)
    p.add_argument("--basil_history_decay_lambda", type=float, default=0.03)
    p.add_argument("--basil_historical_top_k", type=int, default=5)
    p.add_argument("--basil_min_evidence_threshold", type=int, default=2)
    p.add_argument(
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
    p.add_argument("--basil_rerank_top_n", type=int, default=20)
    p.add_argument("--basil_candidate_reason_top_r", type=int, default=3)
    p.add_argument("--basil_time_tolerance_s", type=int, default=300)
    p.set_defaults(deterministic_only=False)
    p.add_argument("--deterministic_only", action="store_true")
    p.set_defaults(no_llm_fallback=False)
    p.add_argument("--no_llm_fallback", action="store_true")
    p.set_defaults(force_counterfactual=False)
    p.add_argument("--force_counterfactual", action="store_true")
    p.set_defaults(window_granger_solver=False)
    p.add_argument("--window_granger_solver", action="store_true")
    p.set_defaults(hybrid_window_granger_prior=False)
    p.add_argument("--hybrid_window_granger_prior", action="store_true")
    p.set_defaults(mid_hard_agent=False)
    p.add_argument("--mid_hard_agent", action="store_true")
    p.set_defaults(light_agent=False)
    p.add_argument("--light_agent", action="store_true")
    p.add_argument("--light_agent_temperature", type=float, default=0.0)
    p.add_argument("--task3_top_k", type=int, default=12)
    p.add_argument("--task3_agent_gap", type=float, default=0.15)
    p.set_defaults(enable_query_count_parser=True)
    p.add_argument("--enable_query_count_parser", action="store_true")
    p.add_argument("--disable_query_count_parser", action="store_false", dest="enable_query_count_parser")
    p.set_defaults(enable_multi_root_joint_selection=True)
    p.add_argument("--enable_multi_root_joint_selection", action="store_true")
    p.add_argument("--disable_multi_root_joint_selection", action="store_false", dest="enable_multi_root_joint_selection")
    p.add_argument("--multi_root_top_n", type=int, default=10)
    return p


def _build_args_namespace(ns: argparse.Namespace) -> Any:
    return SimpleNamespace(**vars(ns))


def _should_keep_row(spec_difficulty: str, wanted: str) -> bool:
    w = str(wanted or "").strip().lower()
    if not w:
        return True
    return str(spec_difficulty or "").strip().lower() == w


def _parse_task_ids(spec: str) -> Optional[set[int]]:
    s = str(spec or "").strip()
    if not s:
        return None
    out: set[int] = set()
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out if out else None


def _parse_row_ids(spec: str) -> Optional[set[int]]:
    s = str(spec or "").strip()
    if not s:
        return None
    out: set[int] = set()
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out if out else None


def _extract_timing_breakdown(trajectory: Any) -> Dict[str, Any]:
    if not isinstance(trajectory, list):
        return {}
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        if step.get("action") == "timing" and isinstance(step.get("timing_breakdown_seconds"), dict):
            return dict(step.get("timing_breakdown_seconds") or {})
    return {}


def cli_main(argv: Optional[List[str]] = None) -> int:
    """Run the agent over a dataset's query.csv and score every row.

    This is the single test entry point: it solves each task with the
    agent-standard runner and immediately scores the prediction with
    `evaluate()`, writing a per-row result CSV under test/result/{dataset}/.
    """
    parser = build_arg_parser()
    args, _unknown = parser.parse_known_args(argv)
    args_obj = _build_args_namespace(args)

    dataset = str(getattr(args_obj, "dataset", "Telecom") or "Telecom")
    query_path = f"dataset/{dataset}/query.csv"
    if not os.path.exists(query_path):
        raise SystemExit(f"Missing dataset file: {query_path}")

    record_path = f"dataset/{dataset}/record.csv"
    record_df = pd.read_csv(record_path) if os.path.exists(record_path) else pd.DataFrame()

    df = pd.read_csv(query_path).reset_index().rename(columns={"index": "row_id"})
    start_idx = int(getattr(args_obj, "start_idx", 0) if getattr(args_obj, "start_idx", None) is not None else 0)
    end_idx = int(getattr(args_obj, "end_idx", 999999) if getattr(args_obj, "end_idx", None) is not None else 999999)
    selected_task_ids = _parse_task_ids(getattr(args_obj, "task_ids", ""))
    selected_row_ids = _parse_row_ids(getattr(args_obj, "row_ids", ""))

    out_dir = f"test/result/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    tag = str(getattr(args_obj, "tag", "agent_standard") or "agent_standard")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"{out_dir}/{tag}_{ts}.csv"

    runner = build_agent_standard_runner()
    extras = AgentStandardRuntimeExtras(light_agent_tasks=set([3]))

    columns = [
        "instruction",
        "prediction",
        "groundtruth",
        "passed",
        "failed",
        "score",
        "row_id",
        "task_index",
        "solve_seconds",
        "evaluate_seconds",
        "sample_seconds",
        "candidate_generation_seconds",
        "basil_evidence_build_seconds",
        "basil_index_query_seconds",
        "basil_index_total_seconds",
        "agent_call_seconds",
        "llm_input_tokens",
        "llm_gate_called",
        "llm_verify_count",
        "llm_changed_count",
        "llm_veto_count",
        "llm_adopted_count",
        "postprocess_seconds",
        "setup_seconds",
        "component_candidate_seconds",
        "time_candidate_seconds",
        "total_execution_seconds",
        "candidate_pool_count",
        "candidate_component_count",
        "candidate_reason_count",
        "candidate_time_count",
        "candidate_component_family_count",
        "candidate_time_cluster_count",
        "llm_max_prompt_candidates",
        "candidate_pool_preview",
    ]
    pd.DataFrame(columns=columns).to_csv(out_path, index=False)

    rows: List[Dict[str, Any]] = []
    n_total = 0
    n_perfect = 0
    t0 = time.perf_counter()
    for _, r in df.iterrows():
        row_id = int(r["row_id"])
        if selected_row_ids is not None and row_id not in selected_row_ids:
            continue
        if row_id < start_idx or row_id > end_idx:
            continue

        task_id = _infer_task_id(r.get("task_index"))
        if selected_task_ids is not None and int(task_id) not in selected_task_ids:
            continue
        spec = get_task_spec(task_id)
        if not _should_keep_row(str(spec.difficulty), str(getattr(args_obj, "difficulty", "") or "")):
            continue

        instruction = str(r.get("instruction") or "")

        sample_t0 = time.perf_counter()
        solve_t0 = time.perf_counter()
        route_mode = str(getattr(args_obj, "route_mode", "auto") or "auto").strip().lower()
        fast_mid_hard = bool(route_mode == "fast")
        pred, traj, prompt = solve_by_mode(
            spec=spec,
            args=args_obj,
            dataset=dataset,
            idx=row_id,
            instruction=instruction,
            task_focus=_task_focus(task_id),
            logger=logger,
            archive_df=None,
            no_llm_fallback=bool(getattr(args_obj, "no_llm_fallback", False)),
            fast_mid_hard=fast_mid_hard,
            light_agent_tasks=set([3]),
            rca_agent_tasks=None,
            agent_standard_runner=runner,
            agent_standard_extras=extras,
        )
        solve_seconds = float(time.perf_counter() - solve_t0)
        timing = _extract_timing_breakdown(traj)

        eval_t0 = time.perf_counter()
        scoring_points = str(r.get("scoring_points") or "")
        passed, failed, score = evaluate(pred, scoring_points)
        evaluate_seconds = float(time.perf_counter() - eval_t0)
        sample_seconds = float(time.perf_counter() - sample_t0)

        logger.info(f"Row {int(row_id)} {str(r.get('task_index') or '')} | score={float(score):.4f}")
        logger.info(f"Prediction: {pred}")
        logger.info(f"Scoring Points: {scoring_points}")
        logger.info(f"Passed Criteria: {passed}")
        logger.info(f"Failed Criteria: {failed}")
        logger.info(
            f"Timing: solve_seconds={solve_seconds:.6f}, evaluate_seconds={evaluate_seconds:.6f}, sample_seconds={sample_seconds:.6f}"
        )

        gt = ""
        try:
            if not record_df.empty and 0 <= int(row_id) < int(len(record_df)):
                rr = record_df.iloc[int(row_id)]
                gt = "\n".join(
                    [
                        f"level: {str(rr.get('level', ''))}",
                        f"reason: {str(rr.get('reason', ''))}",
                        f"component: {str(rr.get('component', ''))}",
                        f"timestamp: {str(rr.get('timestamp', ''))}",
                        f"datetime: {str(rr.get('datetime', ''))}",
                    ]
                )
        except Exception:
            gt = ""

        basil_evidence_build_seconds = float(timing.get("basil_evidence_build_seconds", 0.0) or 0.0)
        basil_index_query_seconds = float(timing.get("basil_index_query_seconds", 0.0) or 0.0)
        basil_index_total_seconds = float(timing.get("basil_index_total_seconds", 0.0) or 0.0)

        logger.info(
            f"BASIL timing: evidence_build_seconds={basil_evidence_build_seconds:.6f}, "
            f"index_query_seconds={basil_index_query_seconds:.6f}, "
            f"index_total_seconds={basil_index_total_seconds:.6f}"
        )
        if timing.get("candidate_pool_count") is not None:
            logger.info(
                "Candidate pool: "
                f"total={int(timing.get('candidate_pool_count', 0) or 0)}, "
                f"components={int(timing.get('candidate_component_count', 0) or 0)}, "
                f"reasons={int(timing.get('candidate_reason_count', 0) or 0)}, "
                f"times={int(timing.get('candidate_time_count', 0) or 0)}, "
                f"prompt_limit={int(timing.get('llm_max_prompt_candidates', 0) or 0)}"
            )

        row_obj = {
            "instruction": instruction,
            "prediction": pred,
            "groundtruth": gt,
            "passed": json.dumps(passed, ensure_ascii=False),
            "failed": json.dumps(failed, ensure_ascii=False),
            "score": float(score),
            "row_id": int(row_id),
            "task_index": str(r.get("task_index") or ""),
            "solve_seconds": float(solve_seconds),
            "evaluate_seconds": float(evaluate_seconds),
            "sample_seconds": float(sample_seconds),
            "candidate_generation_seconds": float(timing.get("candidate_generation_seconds", 0.0) or 0.0),
            "basil_evidence_build_seconds": float(basil_evidence_build_seconds),
            "basil_index_query_seconds": float(basil_index_query_seconds),
            "basil_index_total_seconds": float(basil_index_total_seconds),
            "agent_call_seconds": float(timing.get("agent_call_seconds", 0.0) or 0.0),
            "llm_input_tokens": int(timing.get("llm_input_tokens", 0) or 0),
            "llm_gate_called": bool(timing.get("llm_gate_called", False)),
            "llm_verify_count": int(timing.get("llm_verify_count", 0) or 0),
            "llm_changed_count": int(timing.get("llm_changed_count", 0) or 0),
            "llm_veto_count": int(timing.get("llm_veto_count", 0) or 0),
            "llm_adopted_count": int(timing.get("llm_adopted_count", 0) or 0),
            "postprocess_seconds": float(timing.get("postprocess_seconds", 0.0) or 0.0),
            "setup_seconds": float(timing.get("setup_seconds", 0.0) or 0.0),
            "component_candidate_seconds": float(timing.get("component_candidate_seconds", 0.0) or 0.0),
            "time_candidate_seconds": float(timing.get("time_candidate_seconds", 0.0) or 0.0),
            "total_execution_seconds": float(timing.get("total_execution_seconds", solve_seconds) or solve_seconds),
            "candidate_pool_count": int(timing.get("candidate_pool_count", 0) or 0),
            "candidate_component_count": int(timing.get("candidate_component_count", 0) or 0),
            "candidate_reason_count": int(timing.get("candidate_reason_count", 0) or 0),
            "candidate_time_count": int(timing.get("candidate_time_count", 0) or 0),
            "candidate_component_family_count": int(timing.get("candidate_component_family_count", 0) or 0),
            "candidate_time_cluster_count": int(timing.get("candidate_time_cluster_count", 0) or 0),
            "llm_max_prompt_candidates": int(timing.get("llm_max_prompt_candidates", 0) or 0),
            "candidate_pool_preview": json.dumps(timing.get("candidate_pool_preview", []) or [], ensure_ascii=False),
        }
        rows.append(row_obj)
        n_total += 1
        if float(score) == 1.0:
            n_perfect += 1
        pd.DataFrame([row_obj], columns=columns).to_csv(out_path, mode="a", header=False, index=False)
    total_s = time.perf_counter() - t0
    acc = float(n_perfect / max(1, n_total))
    logger.info(f"saved={out_path} rows={int(n_total)} acc={acc:.4f} total_seconds={total_s:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
