import json
import re
import sys
from typing import Any, Dict, List

import pandas as pd


DT_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def dt_to_s(value: str):
    try:
        return int(pd.Timestamp(str(value), tz="Asia/Shanghai").tz_convert("UTC").timestamp())
    except Exception:
        return None


def load_candidates(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    try:
        obj = json.loads(str(value or "[]"))
    except Exception:
        return []
    return [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []


def gt_time_from_failed(value: Any) -> str:
    hits = DT_RE.findall(str(value or ""))
    return hits[0] if hits else ""


def coverage(gt: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_s = dt_to_s(gt)
    best_err = None
    best_cand = None
    for cand in candidates:
        cand_s = dt_to_s(str(cand.get("datetime", "")))
        if gt_s is None or cand_s is None:
            continue
        err = abs(int(cand_s) - int(gt_s))
        if best_err is None or err < best_err:
            best_err = err
            best_cand = cand
    return {
        "hit": bool(best_err is not None and best_err <= 60),
        "min_abs_error_s": best_err,
        "nearest": best_cand,
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: report_time_candidate_coverage.py result.csv [result2.csv ...]")
    df = pd.concat([pd.read_csv(path) for path in sys.argv[1:]], ignore_index=True)
    rows = []
    for _, row in df.iterrows():
        gt = str(row.get("time_gt") or "") or gt_time_from_failed(row.get("failed"))
        if not gt:
            continue
        candidates = load_candidates(row.get("time_candidates"))
        cov = coverage(gt, candidates)
        rows.append(
            {
                "row_id": row.get("row_id"),
                "task_index": row.get("task_index"),
                "score": row.get("score"),
                "gt_time": gt,
                "gt_in_topk": cov["hit"],
                "min_abs_error_s": cov["min_abs_error_s"],
                "nearest_candidate": json.dumps(cov["nearest"], ensure_ascii=False),
                "top_candidates": json.dumps(candidates[:8], ensure_ascii=False),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        print("No time-failed rows found.")
        return
    print(out.to_string(index=False))
    print()
    print(
        json.dumps(
            {
                "time_failed_rows": int(len(out)),
                "gt_in_topk_count": int(out["gt_in_topk"].sum()),
                "gt_in_topk_rate": float(out["gt_in_topk"].mean()),
                "median_min_abs_error_s": float(pd.to_numeric(out["min_abs_error_s"], errors="coerce").median()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
