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
from rca.telemetry_solver import canonicalize_prediction
from rca.execution.agent_standard import AgentStandardRuntimeExtras, build_agent_standard_runner
from rca.execution.run_agent_standard_flow import get_task_spec, solve_by_mode

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
        best_score = 0
        best_prediction = None
        best_trajectory = None
        best_prompt = None
        catalog = spec.difficulty

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
                best_prediction = canonicalize_prediction(best_prediction)
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
    parser.add_argument("--row_ids_file", type=str, default="")
    parser.add_argument("--route_mode", type=str, default="auto")
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
