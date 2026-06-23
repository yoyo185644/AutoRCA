import argparse
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd


def _task_id(task_index: str):
    try:
        return int(str(task_index).split("_")[1])
    except Exception:
        return None


def _difficulty(tid):
    if tid is None:
        return "unknown"
    t = int(tid)
    if 1 <= t <= 3:
        return "easy"
    if 4 <= t <= 6:
        return "middle"
    return "hard"


def _required_fields(task_id: int) -> Tuple[bool, bool, bool]:
    t = int(task_id)
    need_time = t in (1, 4, 5, 7)
    need_reason = t in (2, 4, 6, 7)
    need_comp = t in (3, 5, 6, 7)
    return need_time, need_comp, need_reason


def _parse_pred(pred: str) -> Dict[str, Any]:
    try:
        return json.loads(str(pred))
    except Exception:
        return {}


def _count_expected_roots(scoring_points: str) -> int:
    sp = str(scoring_points or "")
    xs = re.findall(r"\\b(\\d+)-th root cause occurrence time\\b", sp)
    if xs:
        try:
            return max(int(x) for x in xs)
        except Exception:
            return len(xs)
    if "The only root cause occurrence time" in sp:
        return 1
    return 1


def _extract_expected_components(scoring_points: str) -> List[str]:
    sp = str(scoring_points or "")
    xs = re.findall(r"predicted root cause component is\\s+([^\\n]+)", sp, flags=re.I)
    out = []
    for x in xs:
        v = str(x).strip()
        if v and v not in out:
            out.append(v)
    return out


def _extract_expected_reasons(scoring_points: str) -> List[str]:
    sp = str(scoring_points or "")
    xs = re.findall(r"predicted root cause reason is\\s+([^\\n]+)", sp, flags=re.I)
    out = []
    for x in xs:
        v = str(x).strip()
        if v and v not in out:
            out.append(v)
    return out


def _extract_expected_times(scoring_points: str) -> List[str]:
    sp = str(scoring_points or "")
    xs = re.findall(r"<=1min\)\s+of\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", sp)
    out = []
    for x in xs:
        v = str(x).strip()
        if v and v not in out:
            out.append(v)
    return out


def _dt_seconds(s: str) -> int:
    try:
        return int(datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S").timestamp())
    except Exception:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--head", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(str(args.csv))
    df["task_id"] = df["task_index"].astype(str).map(_task_id)
    df["difficulty"] = df["task_id"].map(_difficulty)

    rows = []
    for _, r in df.iterrows():
        tid = int(r["task_id"])
        need_time, need_comp, need_reason = _required_fields(tid)
        sp = str(r.get("scoring_points", "") or "")
        pred_raw = str(r.get("prediction", "") or "")
        pred = _parse_pred(pred_raw)
        expected_k = _count_expected_roots(sp)
        predicted_k = len([k for k in pred.keys() if str(k).strip().isdigit()]) if isinstance(pred, dict) else 0

        exp_times = _extract_expected_times(sp) if need_time else []
        exp_comps = _extract_expected_components(sp) if need_comp else []
        exp_reas = _extract_expected_reasons(sp) if need_reason else []

        pred_times, pred_comps, pred_reas = [], [], []
        if isinstance(pred, dict):
            for i in range(1, max(1, predicted_k) + 1):
                item = pred.get(str(i), {}) if isinstance(pred.get(str(i)), dict) else {}
                pred_times.append(str(item.get("root cause occurrence datetime", "") or "").strip())
                pred_comps.append(str(item.get("root cause component", "") or "").strip())
                pred_reas.append(str(item.get("root cause reason", "") or "").strip())

        time_hit = True
        if need_time:
            time_hit = False
            for gt in exp_times:
                for pt in pred_times:
                    if _dt_seconds(gt) >= 0 and _dt_seconds(pt) >= 0 and abs(_dt_seconds(gt) - _dt_seconds(pt)) <= 60:
                        time_hit = True
                        break
                if time_hit:
                    break

        comp_hit = True
        if need_comp:
            comp_hit = any((pc == gc) for gc in exp_comps for pc in pred_comps) if exp_comps else False

        reason_hit = True
        if need_reason:
            reason_hit = any((pr == gr) for gr in exp_reas for pr in pred_reas) if exp_reas else False

        miss = []
        if predicted_k < expected_k:
            miss.append("root_count")
        if need_time and not time_hit:
            miss.append("time")
        if need_comp and not comp_hit:
            miss.append("component")
        if need_reason and not reason_hit:
            miss.append("reason")
        if not pred:
            miss.append("format")

        rows.append(
            {
                "task_index": str(r["task_index"]),
                "difficulty": str(r["difficulty"]),
                "score": float(r.get("score", 0.0) or 0.0),
                "expected_k": int(expected_k),
                "predicted_k": int(predicted_k),
                "miss": ",".join(miss) if miss else "",
            }
        )

    mdf = pd.DataFrame(rows)
    print("Miss breakdown (all):")
    print(mdf["miss"].value_counts(dropna=False).head(30).to_string())

    print("\nMiss breakdown by task_id:")
    for t in sorted(mdf["task_index"].unique()):
        g = mdf[mdf["task_index"] == t]
        vc = g["miss"].value_counts(dropna=False).head(20)
        print(f"\n{t} n={len(g)} acc={(g['score']==1.0).mean():.3f}")
        print(vc.to_string())

    print("\nRoot-count mismatch by task_id:")
    rc = mdf[mdf["predicted_k"] < mdf["expected_k"]]
    if rc.empty:
        print("none")
    else:
        print(rc.groupby("task_index")["task_index"].count().to_string())

    if int(args.head) > 0:
        print("\nSample mismatches:")
        print(mdf[mdf["miss"] != ""].head(int(args.head)).to_string(index=False))


if __name__ == "__main__":
    main()
