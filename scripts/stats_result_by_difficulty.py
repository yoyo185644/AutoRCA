import argparse
import os
import sys

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


def _acc(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    return float((series == 1.0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    path = str(args.csv)
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")

    df = pd.read_csv(path)
    if "task_index" not in df.columns:
        raise SystemExit("missing column: task_index")
    if "score" not in df.columns:
        raise SystemExit("missing column: score")

    df["task_id"] = df["task_index"].astype(str).map(_task_id)
    df["difficulty"] = df["task_id"].map(_difficulty)

    time_cols = [c for c in ["total_execution_seconds", "sample_seconds", "solve_seconds"] if c in df.columns]

    rows = []
    for diff, g in df.groupby("difficulty", dropna=False):
        row = {
            "difficulty": str(diff),
            "n": int(len(g)),
            "hit": int((g["score"] == 1.0).sum()),
            "acc": _acc(g["score"]),
            "score_mean": float(g["score"].mean()),
            "score_median": float(g["score"].median()),
        }
        for c in time_cols:
            row[c + "_mean"] = float(g[c].mean())
            row[c + "_p50"] = float(g[c].median())
            row[c + "_p90"] = float(g[c].quantile(0.9))
        rows.append(row)

    all_row = {
        "difficulty": "Total",
        "n": int(len(df)),
        "hit": int((df["score"] == 1.0).sum()),
        "acc": _acc(df["score"]),
        "score_mean": float(df["score"].mean()),
        "score_median": float(df["score"].median()),
    }
    for c in time_cols:
        all_row[c + "_mean"] = float(df[c].mean())
        all_row[c + "_p50"] = float(df[c].median())
        all_row[c + "_p90"] = float(df[c].quantile(0.9))
    rows.append(all_row)

    res = pd.DataFrame(rows)
    order = ["easy", "middle", "hard", "unknown", "Total"]
    res["difficulty"] = pd.Categorical(res["difficulty"], categories=order, ordered=True)
    res = res.sort_values("difficulty")

    pd.set_option("display.max_columns", 200)
    print("file:", path)
    print(res.to_string(index=False))

    pt = (
        df.groupby(["difficulty", "task_id"], dropna=False)
        .agg(
            n=("score", "size"),
            hit=("score", lambda s: int((s == 1.0).sum())),
            acc=("score", lambda s: float((s == 1.0).mean())),
            score_mean=("score", "mean"),
            solve_mean=("solve_seconds", "mean") if "solve_seconds" in df.columns else ("score", "mean"),
        )
        .reset_index()
        .sort_values(["difficulty", "task_id"])
    )
    print("\nPer-task_id:")
    print(pt.to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()

