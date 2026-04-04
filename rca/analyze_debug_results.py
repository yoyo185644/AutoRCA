import argparse
import glob
import os
import re

import pandas as pd


def infer_difficulty(task_index: str) -> str:
    task_num = int(str(task_index).split("_")[-1])
    if task_num <= 3:
        return "easy"
    if task_num <= 6:
        return "middle"
    return "hard"


def extract_field(text: str, key: str) -> str:
    patterns = [
        rf'{re.escape(key)}"\s*:\s*"([^"]+)"',
        rf"{re.escape(key)}\s*:\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text))
        if match:
            return match.group(1).strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", action="append", required=True)
    parser.add_argument("--query", type=str, default="dataset/Telecom/query.csv")
    args = parser.parse_args()

    paths = []
    for pattern in args.pattern:
        paths.extend(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError("No result files matched the given patterns.")

    dfs = []
    for path in sorted(paths):
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)

    query = pd.read_csv(args.query).reset_index().rename(columns={"index": "row_id"})
    query["difficulty"] = query["task_index"].apply(infer_difficulty)
    full["row_id"] = full["row_id"].astype(int)
    full = full.merge(query[["row_id", "difficulty"]], on="row_id", how="left")
    full = full.sort_values("row_id")

    print("Matched files:")
    for path in sorted(paths):
        print(path)
    print()
    print("Overall metrics:")
    print(f"rows={len(full)}, score_sum={full['score'].sum():.3f}, avg={full['score'].mean():.3f}")
    print(f"perfect={int((full['score'] == 1.0).sum())}, accuracy={(full['score'] == 1.0).mean():.3%}")
    print()
    print("By difficulty:")
    print(full.groupby("difficulty")["score"].agg(["count", "sum", "mean"]).to_string())
    print()
    full["pred_component"] = full["prediction"].apply(lambda x: extract_field(x, "root cause component"))
    full["pred_reason"] = full["prediction"].apply(lambda x: extract_field(x, "root cause reason"))
    full["pred_datetime"] = full["prediction"].apply(lambda x: extract_field(x, "root cause occurrence datetime"))
    full["gt_component"] = full["groundtruth"].apply(lambda x: extract_field(x, "component"))
    full["gt_reason"] = full["groundtruth"].apply(lambda x: extract_field(x, "reason"))
    full["gt_datetime"] = full["groundtruth"].apply(lambda x: extract_field(x, "datetime"))
    print("Field match summary:")
    print(f"component_match={(full['pred_component'] == full['gt_component']).sum()}/{len(full)}")
    print(f"reason_match={(full['pred_reason'] == full['gt_reason']).sum()}/{len(full)}")
    print(f"datetime_exact_match={(full['pred_datetime'] == full['gt_datetime']).sum()}/{len(full)}")
    print()
    print("Failed cases:")
    failed = full.loc[
        full["score"] < 1,
        [
            "row_id",
            "task_index",
            "difficulty",
            "score",
            "pred_component",
            "gt_component",
            "pred_reason",
            "gt_reason",
            "pred_datetime",
            "gt_datetime",
            "instruction",
        ],
    ]
    print(failed.to_string(index=False, max_colwidth=120))


if __name__ == "__main__":
    main()
