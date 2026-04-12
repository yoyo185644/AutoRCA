import json
import sys

import pandas as pd


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: find_missing_rows_in_result.py rows.json result.csv out.json")
    rows_path, result_path, out_path = sys.argv[1:]
    with open(rows_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    target = set(int(x) for x in (rows.get("row_ids") or []))
    df = pd.read_csv(result_path)
    present = set(int(x) for x in pd.to_numeric(df["row_id"], errors="coerce").dropna().astype(int).tolist())
    missing = sorted(list(target - present))
    out = {"target_count": int(len(target)), "present_count": int(len(present & target)), "missing": missing}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

