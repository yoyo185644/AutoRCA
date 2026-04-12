import json
import os
from types import SimpleNamespace

import pandas as pd

from rca.run_agent_standard import main


def run():
    uid = "telecom_easy_eval"
    tag = "telecom_easy_task3_hybrid_r5"
    args = SimpleNamespace(
        dataset="Telecom",
        sample_num=1,
        start_idx=0,
        end_idx=9999,
        controller_max_step=25,
        controller_max_turn=5,
        timeout=600,
        tag=tag,
        auto=False,
        row_ids_file="",
        fast_easy_solver=True,
        telecom_use_archive_baseline=False,
        telecom_archive_pred="rca/archive/agent-Telecom.csv",
        use_archive_baseline=False,
        archive_pred="",
        telecom_task1_trace_time=True,
        telecom_task3_granger_component=False,
        window_granger_solver=False,
        hybrid_window_granger_prior=False,
        telecom_easy_only=True,
        telecom_easy_allow_llm=True,
        telecom_task2_agent=True,
        telecom_task2_agent_gap=0.6,
        telecom_task2_agent_min_score=1.2,
        telecom_task2_agent_temperature=0.0,
        light_agent=True,
        light_agent_temperature=0.0,
        task3_top_k=20,
        task3_agent_gap=0.15,
        no_llm_fallback=False,
        auto_mode=False,
    )
    main(args, uid=uid, dataset="Telecom")

    model = __import__("rca.api_router", fromlist=["configs"]).configs["MODEL"].split("/")[-1]
    out_csv = f"test/result/Telecom/agent-{tag}-{model}-{uid}.csv"
    df = pd.read_csv(out_csv)
    summary = {
        "csv": out_csv,
        "rows": int(len(df)),
        "mean": float(df["score"].mean()) if len(df) else 0.0,
        "acc": float((df["score"] == 1.0).mean()) if len(df) else 0.0,
        "by_task": df.groupby("task_index")["score"].agg(["count", "sum", "mean"]).to_dict() if len(df) else {},
    }
    summary_path = f"test/result/Telecom/summary-{tag}-{uid}.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run()

