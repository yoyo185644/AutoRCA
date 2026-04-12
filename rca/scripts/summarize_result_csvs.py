import json
import sys

import pandas as pd


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: summarize_result_csvs.py out.json in1.csv [in2.csv ...]")
    out_path = sys.argv[1]
    in_paths = sys.argv[2:]
    frames = [pd.read_csv(p) for p in in_paths]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "row_id" in df.columns:
        df["row_id_int"] = pd.to_numeric(df["row_id"], errors="coerce").astype("Int64")
    else:
        df["row_id_int"] = pd.NA
    key_cols = ["row_id_int", "task_index"] if "task_index" in df.columns else ["row_id_int"]
    df = df.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last")
    out = {
        "files": in_paths,
        "rows": int(len(df)),
        "acc": float((df["score"] == 1.0).mean()) if len(df) else 0.0,
        "mean": float(df["score"].mean()) if len(df) else 0.0,
        "sum": float(df["score"].sum()) if len(df) else 0.0,
        "by_task": {},
        "by_task_acc": {},
        "by_mode": {},
        "missing_row_ids": [],
    }
    if len(df) and df["row_id_int"].notna().any():
        present = set(int(x) for x in df["row_id_int"].dropna().astype(int).tolist())
        mx = max(present) if present else -1
        if mx >= 0:
            out["missing_row_ids"] = [i for i in range(mx + 1) if i not in present]
    if len(df) and "task_index" in df.columns:
        g = df.groupby("task_index")["score"].agg(["count", "sum", "mean"])
        out["by_task"] = {k: {"count": int(v["count"]), "sum": float(v["sum"]), "mean": float(v["mean"])} for k, v in g.to_dict(orient="index").items()}
        ga = df.groupby("task_index")["score"].apply(lambda s: float((s == 1.0).mean()))
        out["by_task_acc"] = {k: float(v) for k, v in ga.to_dict().items()}
        task_id = df["task_index"].astype(str).str.replace("task_", "", regex=False)
        task_id = pd.to_numeric(task_id, errors="coerce")
        mode = task_id.apply(lambda t: "easy" if t <= 3 else ("middle" if t <= 6 else "hard"))
        df2 = df.copy()
        df2["mode"] = mode
        gm = df2.groupby("mode")["score"].agg(["count", "sum", "mean"])
        out["by_mode"] = {
            k: {
                "count": int(v["count"]),
                "sum": float(v["sum"]),
                "mean": float(v["mean"]),
                "acc": float((df2[df2["mode"] == k]["score"] == 1.0).mean()),
            }
            for k, v in gm.to_dict(orient="index").items()
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
