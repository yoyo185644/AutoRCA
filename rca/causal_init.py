import os
import json
from typing import Dict, Any, List, Tuple

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def _dataset_key(dataset_name: str) -> str:
    return (dataset_name or "unknown").replace("/", "__")


def _gy_path(dataset_name: str) -> str:
    return os.path.join("rca", "skill_kb", f"{_dataset_key(dataset_name)}__Gy.json")


def _init_gy(dataset_name: str) -> Dict[str, Any]:
    return {"dataset": dataset_name, "nodes": [], "edges": [], "metadata": {"created_utc": "init"}}


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _telemetry_metric_path(dataset_name: str, date_str: str, level: str = "node") -> str:
    fname = f"metric_{level}.csv"
    return os.path.join("dataset", dataset_name, "telemetry", date_str, "metric", fname)


def _load_metric_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(lambda x: int(x // 1000) if len(str(x)) == 13 else x)
    return df


def _build_series(df: pd.DataFrame, max_per_component: int = 8) -> Dict[Tuple[str, str], pd.Series]:
    df = df[["cmdb_id", "name", "timestamp", "value"]].dropna()
    df = df.sort_values("timestamp")
    series_map: Dict[Tuple[str, str], pd.Series] = {}
    grouped = df.groupby(["cmdb_id", "name"])
    for (comp, metric), g in grouped:
        s = g.set_index("timestamp")["value"].astype(float)
        if len(s) >= 20:
            series_map[(comp, metric)] = s
    by_comp: Dict[str, int] = {}
    filtered: Dict[Tuple[str, str], pd.Series] = {}
    for (comp, metric), s in series_map.items():
        c = by_comp.get(comp, 0)
        if c < max_per_component:
            filtered[(comp, metric)] = s
            by_comp[comp] = c + 1
    return filtered


def _granger_edges(series_map: Dict[Tuple[str, str], pd.Series], max_pairs: int = 200, max_lag: int = 2, p_thresh: float = 0.05) -> List[Dict[str, Any]]:
    keys = list(series_map.keys())
    edges: List[Dict[str, Any]] = []
    pairs = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if pairs >= max_pairs:
                break
            k1, k2 = keys[i], keys[j]
            s1, s2 = series_map[k1], series_map[k2]
            idx = s1.index.intersection(s2.index)
            if len(idx) < 30:
                continue
            x = s1.loc[idx].values
            y = s2.loc[idx].values
            df_xy = pd.DataFrame({"x": x, "y": y})
            try:
                res_xy = grangercausalitytests(df_xy[["y", "x"]], maxlag=max_lag, verbose=False)
                res_yx = grangercausalitytests(df_xy[["x", "y"]], maxlag=max_lag, verbose=False)
            except Exception:
                continue
            best_xy_p = min([res_xy[lag][0]["ssr_ftest"][1] for lag in res_xy])
            best_yx_p = min([res_yx[lag][0]["ssr_ftest"][1] for lag in res_yx])
            if best_xy_p < p_thresh:
                edges.append({
                    "source": f"{k1[0]}::{k1[1]}",
                    "target": f"{k2[0]}::{k2[1]}",
                    "lag_min": 60,
                    "lag_max": 300,
                    "duration_min": 60,
                    "duration_max": 1800,
                    "condition": "granger_init",
                    "confidence": float(max(0.1, 1.0 - best_xy_p)),
                })
            if best_yx_p < p_thresh:
                edges.append({
                    "source": f"{k2[0]}::{k2[1]}",
                    "target": f"{k1[0]}::{k1[1]}",
                    "lag_min": 60,
                    "lag_max": 300,
                    "duration_min": 60,
                    "duration_max": 1800,
                    "condition": "granger_init",
                    "confidence": float(max(0.1, 1.0 - best_yx_p)),
                })
            pairs += 1
        if pairs >= max_pairs:
            break
    return edges


def init_gy_with_granger(dataset_name: str, date_str: str, level: str = "node", max_pairs: int = 200, max_lag: int = 2) -> str:
    metric_path = _telemetry_metric_path(dataset_name, date_str, level=level)
    df = _load_metric_df(metric_path)
    series_map = _build_series(df, max_per_component=8)
    edges = _granger_edges(series_map, max_pairs=max_pairs, max_lag=max_lag)
    gy_path = _gy_path(dataset_name)
    gy = _safe_read_json(gy_path) or _init_gy(dataset_name)
    if not isinstance(gy.get("edges"), list):
        gy["edges"] = []
    gy["edges"].extend(edges)
    _safe_write_json(gy_path, gy)
    return gy_path


def _main(argv: List[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--level", type=str, default="node")
    parser.add_argument("--max_pairs", type=int, default=200)
    parser.add_argument("--max_lag", type=int, default=2)
    args = parser.parse_args(argv)
    path = init_gy_with_granger(args.dataset, args.date, level=args.level, max_pairs=args.max_pairs, max_lag=args.max_lag)
    print(path)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
