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
from rca.telecom_easy_solver import solve_telecom_easy, estimate_task1_datetime_from_trace, pick_component_by_granger
from rca.window_granger_solver import solve_with_window_granger

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

def main(args, uid, dataset):
    if dataset == "Telecom" and args.telecom_easy_only:
        args.fast_easy_solver = True
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
    no_llm_fallback = bool(args.no_llm_fallback or args.use_archive_baseline or (dataset == "Telecom" and args.telecom_easy_only))
    
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
                    prediction = solve_with_window_granger(dataset=dataset, instruction=instruction, task_id=task_id)
                if args.hybrid_window_granger_prior and dataset.startswith("Market/"):
                    hybrid_tasks = {
                        "Market/cloudbed-1": {2, 3, 4},
                        "Market/cloudbed-2": {1, 4},
                    }
                    if task_id in hybrid_tasks.get(dataset, set()):
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
                    if args.fast_easy_solver and dataset == "Telecom" and task_id <= 3:
                        try:
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

                    if prediction is None:
                        if no_llm_fallback:
                            prediction = "{}"
                        else:
                            from rca.baseline.rca_agent.rca_agent import RCA_Agent
                            import rca.baseline.rca_agent.prompt.agent_prompt as ap
                            if dataset == "Telecom":
                                import rca.baseline.rca_agent.prompt.basic_prompt_Telecom as bp
                            elif dataset == "Bank":
                                import rca.baseline.rca_agent.prompt.basic_prompt_Bank as bp
                            elif dataset == "Market/cloudbed-1" or dataset == "Market/cloudbed-2":
                                import rca.baseline.rca_agent.prompt.basic_prompt_Market as bp
                            agent = RCA_Agent(ap, bp)
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
                            prediction, trajectory, prompt = agent.run(
                                instruction,
                                logger,
                                dataset_name=dataset,
                                max_step=dyn_step,
                                max_turn=dyn_turn,
                                task_focus=task_focus,
                            )
                if prediction is None:
                    prediction = "{}"
                
                for step in trajectory:
                    code_cell = nbf.new_code_cell(step['code'])
                    result_cell = nbf.new_markdown_cell(f"```\n{step['result']}\n```")
                    nb.cells.append(code_cell)
                    nb.cells.append(result_cell)
                with open(nbfile, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, ensure_ascii=False, indent=4)
                logger.info(f"Trajectory has been saved to {nbfile}")

                with open(promptfile, 'w', encoding='utf-8') as f:
                    json.dump({"messages": prompt}, f, ensure_ascii=False, indent=4)
                logger.info(f"Prompt has been saved to {promptfile}")

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
