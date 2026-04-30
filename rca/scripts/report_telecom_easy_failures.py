import json
import re
from typing import Any, Dict, List

import pandas as pd

from rca.fast_solver import infer_reason_with_evidence
from rca.component_candidates import build_component_candidates


def _extract_reason_from_pred(pred: str) -> str:
    try:
        obj = json.loads(pred)
        return str(obj.get("1", {}).get("root cause reason", ""))
    except Exception:
        return ""


def _extract_comp_from_pred(pred: str) -> str:
    try:
        obj = json.loads(pred)
        return str(obj.get("1", {}).get("root cause component", ""))
    except Exception:
        return ""


def main():
    csv_path = "test/result/Telecom/agent-telecom_easy_full_tune_r8_ab-gpt-4o-2024-05-13-2026-04-05_18-46-30.csv"
    df = pd.read_csv(csv_path)
    q = pd.read_csv("dataset/Telecom/query.csv")

    report: Dict[str, Any] = {"csv": csv_path, "overall": {}, "task1": {}, "task2": {}, "task3": {}}
    report["overall"] = {
        "rows": int(len(df)),
        "acc": float((df.score == 1.0).mean()),
        "mean": float(df.score.mean()),
        "sum": float(df.score.sum()),
    }

    def add_fail_task(task: str) -> List[int]:
        sub = df[(df.task_index == task) & (df.score < 1.0)].copy()
        return [int(x) for x in sub.row_id.tolist()]

    t1_f = add_fail_task("task_1")
    t2_f = add_fail_task("task_2")
    t3_f = add_fail_task("task_3")
    report["task1"]["fail_ids"] = t1_f
    report["task2"]["fail_ids"] = t2_f
    report["task3"]["fail_ids"] = t3_f

    t2_details = []
    for rid in t2_f:
        inst = q.loc[rid, "instruction"]
        best, ev = infer_reason_with_evidence(inst)
        pred = _extract_reason_from_pred(df[df.row_id == rid].iloc[0].prediction)
        gt = re.findall(r"root cause reason is ([^\n\r]+)", str(q.loc[rid, "scoring_points"]), flags=re.IGNORECASE)
        t2_details.append(
            {
                "row_id": rid,
                "pred": pred,
                "gt": gt[0].strip() if gt else "",
                "best": ev.get("best"),
                "gap": ev.get("gap"),
                "scores": ev.get("scores"),
                "keyword_hits": ev.get("keyword_hits"),
                "trace_spread": ev.get("trace_spread"),
                "top_metrics": {k: (v[:6] if isinstance(v, list) else v) for k, v in (ev.get("top_metrics") or {}).items()},
            }
        )
    report["task2"]["fails"] = t2_details

    t3_details = []
    for rid in t3_f:
        inst = q.loc[rid, "instruction"]
        ev = build_component_candidates("Telecom", inst, top_k=80)
        pred = _extract_comp_from_pred(df[df.row_id == rid].iloc[0].prediction)
        gt = re.findall(r"root cause component is ([^\n\r]+)", str(q.loc[rid, "scoring_points"]), flags=re.IGNORECASE)
        t3_details.append(
            {
                "row_id": rid,
                "pred": pred,
                "gt": gt[0].strip() if gt else "",
                "t_star": ev.estimated_failure_time,
                "top1": ev.top1,
                "top2": ev.top2,
                "gap_norm": ev.gap_norm,
                "trace_top5": sorted(ev.trace_scores.items(), key=lambda kv: kv[1], reverse=True)[:5],
                "os_top5": [(c, float(s)) for c, s in sorted(((c, s) for c, s in ev.combined_scores.items() if str(c).startswith("os_")), key=lambda kv: kv[1], reverse=True)[:5]],
                "db_top5": [(c, float(s)) for c, s in sorted(((c, s) for c, s in ev.combined_scores.items() if str(c).startswith("db_")), key=lambda kv: kv[1], reverse=True)[:5]],
            }
        )
    report["task3"]["fails"] = t3_details

    out_path = "test/result/Telecom/report-telecom_easy_full_tune_r8_ab.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
