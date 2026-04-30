import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rca.fast_solver import infer_reason_with_evidence
from rca.component_candidates import build_component_candidates


def main():
    ps = sorted(glob.glob("test/result/Telecom/agent-telecom_easy_full_tune_r6_ab-*.csv"))
    if not ps:
        raise SystemExit("No result csv found for tag telecom_easy_full_tune_r6_ab")
    p = ps[-1]
    df = pd.read_csv(p)
    q = pd.read_csv("dataset/Telecom/query.csv")
    print("file", p)

    t2 = df[(df.task_index == "task_2") & (df.score < 1.0)].row_id.astype(int).tolist()
    print("task2_fail_ids", t2)
    for rid in t2:
        det, ev = infer_reason_with_evidence(q.loc[rid, "instruction"])
        pred = json.loads(df[df.row_id == rid].iloc[0].prediction).get("1", {}).get("root cause reason")
        print("row", rid, "pred", pred, "best", ev.get("best"), "gap", round(float(ev.get("gap", 0.0)), 3))
        print(" trace_spread", ev.get("trace_spread"))
        print(" keyword_hits", ev.get("keyword_hits"))

    print("task3")
    for rid in [26, 30, 39, 47]:
        ev = build_component_candidates("Telecom", q.loc[rid, "instruction"], top_k=80)
        pred = json.loads(df[df.row_id == rid].iloc[0].prediction).get("1", {}).get("root cause component")
        top_db = sorted(
            [(c, s) for c, s in ev.combined_scores.items() if str(c).startswith("db_")],
            key=lambda kv: kv[1],
            reverse=True,
        )[:2]
        top_os = sorted(
            [(c, s) for c, s in ev.combined_scores.items() if str(c).startswith("os_")],
            key=lambda kv: kv[1],
            reverse=True,
        )[:2]
        print("row", rid, "pred", pred, "top1", ev.top1, "top2", ev.top2, "gap", round(float(ev.gap_norm), 3))
        print(" top_os", [(c, round(s, 3)) for c, s in top_os], "top_db", [(c, round(s, 3)) for c, s in top_db])


if __name__ == "__main__":
    main()
