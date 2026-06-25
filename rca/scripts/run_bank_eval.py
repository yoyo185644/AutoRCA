import json
import os
from types import SimpleNamespace

import pandas as pd

from rca.run_agent_standard import main


def run():
    uid = "bank_eval"
    tag = "bank_basil_refactor"
    args = SimpleNamespace(
        dataset="Bank",
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
        use_archive_baseline=False,
        archive_pred="",
        light_agent=True,
        light_agent_temperature=0.0,
        task3_top_k=20,
        task3_agent_gap=0.15,
        uncertainty_llm=True,
        no_llm_fallback=False,
        auto_mode=False,
        basil_index=True,
        basil_persist_index=True,
        basil_index_cache_dir="test/cache/basil_index",
        basil_bucket_size=300,
        basil_block_size=32,
        basil_max_lag=900,
        basil_severity_floor=0.0,
        basil_w_local=1.0,
        basil_w_causal=0.45,
        basil_w_reason=0.25,
        basil_approx_epsilon=0.0,
        basil_max_trace_rows=0,
        basil_similarity_ts_weight=0.45,
        basil_similarity_topology_weight=0.30,
        basil_similarity_pattern_weight=0.25,
        basil_history_decay_lambda=0.03,
        basil_historical_top_k=5,
        deterministic_only=False,
    )
    main(args, uid=uid, dataset="Bank")

    model = __import__("rca.api_router", fromlist=["configs"]).configs["MODEL"].split("/")[-1]
    out_csv = f"test/result/Bank/agent-{tag}-{model}-{uid}.csv"
    df = pd.read_csv(out_csv)
    summary = {
        "csv": out_csv,
        "rows": int(len(df)),
        "mean": float(df["score"].mean()) if len(df) else 0.0,
        "acc": float((df["score"] == 1.0).mean()) if len(df) else 0.0,
        "by_task": df.groupby("task_index")["score"].agg(["count", "sum", "mean"]).to_dict() if len(df) else {},
    }
    summary_path = f"test/result/Bank/summary-{tag}-{uid}.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
