import sys
from collections import Counter, defaultdict

import pandas as pd


def _split_items(s: object):
    if s is None:
        return []
    if isinstance(s, float) and pd.isna(s):
        return []
    text = str(s).strip()
    if not text:
        return []
    return [x.strip() for x in text.split("\n") if x.strip()]


def _task_id(task_index: str) -> int:
    return int(str(task_index).split("_")[1])


def _difficulty(task_id: int) -> str:
    return "easy" if task_id <= 3 else ("middle" if task_id <= 6 else "hard")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_result_failures.py result.csv")
    path = sys.argv[1]
    df = pd.read_csv(path)
    df["task_id"] = df["task_index"].map(_task_id)
    df["difficulty"] = df["task_id"].map(_difficulty)

    def acc(sub):
        return float((sub["score"] == 1.0).mean()) if len(sub) else 0.0

    print("file", path)
    print("rows", len(df), "overall_acc", acc(df), "mean_score", float(df["score"].mean()) if len(df) else 0.0)
    print("by_difficulty")
    for d in ["easy", "middle", "hard"]:
        sub = df[df["difficulty"] == d]
        print(" ", d, "n", len(sub), "acc", acc(sub), "mean", float(sub["score"].mean()) if len(sub) else 0.0)

    print("by_task")
    g = df.groupby("task_index")["score"].agg(["count", "sum", "mean"])
    ga = df.groupby("task_index")["score"].apply(lambda s: float((s == 1.0).mean()))
    for k in sorted(g.index.tolist(), key=lambda x: int(str(x).split("_")[1])):
        row = g.loc[k]
        print(" ", k, "n", int(row["count"]), "acc", float(ga.loc[k]), "mean", float(row["mean"]))

    print("score_dist", df["score"].value_counts().sort_index().to_dict())

    fail_counter = Counter()
    pass_counter = Counter()
    for s in df["failed"]:
        fail_counter.update(_split_items(s))
    for s in df["passed"]:
        pass_counter.update(_split_items(s))

    print("top_failed", fail_counter.most_common(30))
    print("top_passed", pass_counter.most_common(30))

    # quick diagnosis: for 2-criteria tasks (4/5/6) score==0.5 means 1 of 2 correct; for task7 score==0.33/0.67 indicates 1/2 correct out of 3.
    print("partial_breakdown")
    frac = defaultdict(int)
    for t, s in zip(df["task_id"], df["score"]):
        frac[(int(t), float(s))] += 1
    for (t, s), c in sorted(frac.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(" ", f"task_{t}", "score", s, "count", c)

    # per-task most common failed item
    print("top_failed_by_task")
    per = {}
    for k, sub in df.groupby("task_index"):
        c = Counter()
        for s in sub["failed"]:
            c.update(_split_items(s))
        per[k] = c.most_common(10)
    for k in sorted(per.keys(), key=lambda x: int(str(x).split("_")[1])):
        print(" ", k, per[k])


if __name__ == "__main__":
    main()

