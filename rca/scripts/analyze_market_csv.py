import json
import re
import sys

import pandas as pd


COMP_PATTERN = r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
REASON_PATTERN = r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
TIME_PATTERN = r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)"


def required_k(scoring_points: str) -> int:
    text = str(scoring_points or "")
    k = max(len(re.findall(COMP_PATTERN, text)), len(re.findall(REASON_PATTERN, text)), len(re.findall(TIME_PATTERN, text)))
    return int(k) if k > 0 else 1


def predicted_k(prediction: str) -> int:
    s = str(prediction or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            ks = []
            for k in obj.keys():
                try:
                    ks.append(int(str(k)))
                except Exception:
                    pass
            return max(ks) if ks else 0
    except Exception:
        pass
    m = re.findall(r"\"(\d+)\"\s*:\s*\{", s)
    if m:
        return max(int(x) for x in m)
    return 0


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: analyze_market_csv.py result.csv out.json")
    csv_path, out_path = sys.argv[1:]
    df = pd.read_csv(csv_path)
    df["task_id"] = df["task_index"].astype(str).str.replace("task_", "", regex=False).astype(int)
    df["difficulty"] = df["task_id"].apply(lambda t: "easy" if t <= 3 else ("middle" if t <= 6 else "hard"))
    df["k_req"] = df["groundtruth"].map(required_k)
    df["k_pred"] = df["prediction"].map(predicted_k)

    def acc(sub):
        return float((sub["score"] == 1.0).mean()) if len(sub) else 0.0

    by_diff = {}
    for d in ["easy", "middle", "hard"]:
        sub = df[df["difficulty"] == d]
        by_diff[d] = {
            "n": int(len(sub)),
            "correct": int((sub["score"] == 1.0).sum()),
            "acc": acc(sub),
            "mean": float(sub["score"].mean()) if len(sub) else 0.0,
        }

    multi = df[df["k_req"] > 1]
    mismatch = multi[multi["k_pred"] < multi["k_req"]]
    node_dot = df[df["prediction"].astype(str).str.contains(r"node-\d+\.", regex=True, na=False)]

    out = {
        "file": csv_path,
        "rows": int(len(df)),
        "overall": {"acc": acc(df), "mean": float(df["score"].mean()) if len(df) else 0.0},
        "by_difficulty": by_diff,
        "required_k_dist": df["k_req"].value_counts().to_dict(),
        "predicted_k_dist": df["k_pred"].value_counts().to_dict(),
        "multi_required_rows": int(len(multi)),
        "multi_mismatch_rows": int(len(mismatch)),
        "multi_mismatch_by_task": mismatch["task_index"].value_counts().to_dict(),
        "node_dot_rows": int(len(node_dot)),
        "node_dot_by_task": node_dot["task_index"].value_counts().to_dict(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

