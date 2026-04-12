import ast
import sys
from collections import Counter

import pandas as pd


def parse_list(x):
    if isinstance(x, list):
        return x
    s = str(x)
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: analyze_mid_hard_results.py in1.csv [in2.csv ...]")
    paths = sys.argv[1:]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["row_id_int"] = pd.to_numeric(df.get("row_id"), errors="coerce").astype("Int64")
    if "task_index" in df.columns:
        df = df.sort_values(["row_id_int", "task_index"]).drop_duplicates(subset=["row_id_int", "task_index"], keep="last")
    else:
        df = df.sort_values(["row_id_int"]).drop_duplicates(subset=["row_id_int"], keep="last")
    df["passed_l"] = df["passed"].map(parse_list)
    df["failed_l"] = df["failed"].map(parse_list)
    df["task_id"] = df["task_index"].astype(str).str.replace("task_", "", regex=False).astype(int)
    df["mode"] = df["task_id"].apply(lambda t: "easy" if t <= 3 else ("middle" if t <= 6 else "hard"))

    for m in ["middle", "hard"]:
        sub = df[df["mode"] == m].copy()
        print("==", m, "==")
        print("n", len(sub), "acc", float((sub["score"] == 1.0).mean()) if len(sub) else 0.0, "mean", float(sub["score"].mean()) if len(sub) else 0.0)
        print(sub.groupby("task_index")["score"].agg(["count", "sum", "mean"]).to_string())
        if len(sub) == 0:
            continue
        pc = Counter()
        fc = Counter()
        for ps, fs in zip(sub["passed_l"], sub["failed_l"]):
            pc.update(ps)
            fc.update(fs)
        print("pass_top", pc.most_common(10))
        print("fail_top", fc.most_common(10))
        part = sub[sub["score"] > 0][["row_id_int", "task_index", "score", "passed", "failed"]]
        print("partial_rows", len(part))
        if len(part):
            print(part.to_string(index=False))
        perf = sub[sub["score"] == 1.0][["row_id_int", "task_index", "score"]]
        print("perfect_rows", len(perf))
        if len(perf):
            print(perf.to_string(index=False))


if __name__ == "__main__":
    main()

