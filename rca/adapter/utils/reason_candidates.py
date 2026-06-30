import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from rca.common.io import resolve_date_dir
from rca.adapter.utils.component_candidates import parse_time_window_generic


def _norm_name(s: str) -> str:
    s = str(s or "").lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def _ts_to_seconds(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    v = float(s.dropna().iloc[0])
    if v > 10**12:
        return (s // 1000).astype("Int64")
    return s.astype("Int64")


def _normalize_metric_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=["ts_s", "component", "metric", "value"])
    cols = set(df.columns)
    ts_col = "timestamp" if "timestamp" in cols else ("startTime" if "startTime" in cols else None)
    if ts_col is None:
        return pd.DataFrame(columns=["ts_s", "component", "metric", "value"])
    comp_col = None
    for c in ["cmdb_id", "cmdbId", "tc", "serviceName"]:
        if c in cols:
            comp_col = c
            break
    if comp_col is None:
        return pd.DataFrame(columns=["ts_s", "component", "metric", "value"])
    if "value" in cols:
        metric_col = "name" if "name" in cols else ("kpi_name" if "kpi_name" in cols else None)
        if metric_col is None:
            return pd.DataFrame(columns=["ts_s", "component", "metric", "value"])
        out = df[[ts_col, comp_col, metric_col, "value"]].copy()
        out = out.rename(columns={ts_col: "ts", comp_col: "component", metric_col: "metric"})
        out["ts_s"] = _ts_to_seconds(out["ts"])
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["ts_s", "component", "metric", "value"])
        out["component"] = out["component"].astype(str)
        out["metric"] = out["metric"].astype(str)
        return out[["ts_s", "component", "metric", "value"]]
    value_cols = [c for c in df.columns if c not in [ts_col, comp_col] and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        value_cols = [c for c in df.columns if c not in [ts_col, comp_col] and c not in ["rr", "sr", "cnt", "mrt"]]
    if not value_cols:
        return pd.DataFrame(columns=["ts_s", "component", "metric", "value"])
    out = df[[ts_col, comp_col] + value_cols].copy()
    out = out.rename(columns={ts_col: "ts", comp_col: "component"})
    out["ts_s"] = _ts_to_seconds(out["ts"])
    out = out.dropna(subset=["ts_s", "component"])
    out["component"] = out["component"].astype(str)
    melted = out.melt(id_vars=["ts_s", "component"], value_vars=value_cols, var_name="metric", value_name="value")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["value"])
    melted["metric"] = melted["metric"].astype(str)
    return melted[["ts_s", "component", "metric", "value"]]


def _trace_spread_stats(trace_path: str, start_s: int, end_s: int) -> Dict[str, Any]:
    if not os.path.exists(trace_path):
        return {"total_components": 0, "affected_components": 0, "affected_ratio": 0.0}
    df = pd.read_csv(trace_path, low_memory=False)
    if df.empty or "cmdb_id" not in df.columns:
        return {"total_components": 0, "affected_components": 0, "affected_ratio": 0.0}
    ts_col = "timestamp" if "timestamp" in df.columns else ("startTime" if "startTime" in df.columns else None)
    dur_col = "duration" if "duration" in df.columns else ("elapsedTime" if "elapsedTime" in df.columns else None)
    if ts_col is None or dur_col is None:
        return {"total_components": 0, "affected_components": 0, "affected_ratio": 0.0}
    df = df[[ts_col, "cmdb_id", dur_col]].copy()
    df["ts_s"] = _ts_to_seconds(df[ts_col])
    df["dur"] = pd.to_numeric(df[dur_col], errors="coerce")
    df["cmdb_id"] = df["cmdb_id"].astype(str)
    df = df.dropna(subset=["ts_s", "dur"])
    df = df[(df["ts_s"] >= start_s) & (df["ts_s"] <= end_s)].copy()
    if df.empty:
        return {"total_components": 0, "affected_components": 0, "affected_ratio": 0.0}
    p95 = df.groupby("cmdb_id")["dur"].quantile(0.95)
    if p95.empty:
        return {"total_components": 0, "affected_components": 0, "affected_ratio": 0.0}
    baseline = float(p95.median())
    if baseline <= 0:
        return {"total_components": int(p95.shape[0]), "affected_components": 0, "affected_ratio": 0.0}
    ratios = (p95 / baseline).fillna(0.0)
    affected = int((ratios >= 1.5).sum())
    total = int(ratios.shape[0])
    return {"total_components": total, "affected_components": affected, "affected_ratio": float(affected / total) if total else 0.0}


def _top_anomalous_metrics(metric_dir: str, start_s: int, end_s: int, top_k: int = 30) -> List[str]:
    if not os.path.isdir(metric_dir):
        return []
    paths = [os.path.join(metric_dir, p) for p in os.listdir(metric_dir) if p.endswith(".csv")]
    agg: Dict[str, float] = {}
    for p in paths:
        try:
            long = _normalize_metric_file(p)
        except Exception:
            continue
        if long.empty:
            continue
        p95 = long.groupby(["component", "metric"], as_index=False)["value"].quantile(0.95).rename(columns={"value": "p95"})
        win = long[(long["ts_s"] >= start_s) & (long["ts_s"] <= end_s)].copy()
        if win.empty:
            continue
        win = win.merge(p95, on=["component", "metric"], how="left").dropna(subset=["p95"]).copy()
        denom = win["p95"].abs().where(win["p95"].abs() > 1e-6, 1.0)
        win["breach_ratio"] = (win["value"] - win["p95"]) / denom
        win["breach_ratio"] = win["breach_ratio"].clip(lower=0.0, upper=10.0)
        win = win[win["breach_ratio"] > 0].copy()
        if win.empty:
            continue
        by = win.groupby("metric", as_index=False)["breach_ratio"].sum().sort_values("breach_ratio", ascending=False)
        for _, r in by.head(top_k).iterrows():
            m = str(r["metric"])
            agg[m] = agg.get(m, 0.0) + float(r["breach_ratio"])
    return [m for m, _ in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]


def _keyword_hits(metrics: Sequence[str]) -> Dict[str, int]:
    blob = " ".join([_norm_name(m) for m in metrics])
    def c(tokens: List[str]) -> int:
        return int(sum(1 for t in tokens if t and _norm_name(t) in blob))
    return {
        "cpu": c(["cpu", "load", "iowait", "util", "proc"]),
        "memory": c(["mem", "memory", "rss", "heap", "gc"]),
        "disk": c(["disk", "io", "iops", "await", "read", "write"]),
        "net_loss": c(["loss", "drop", "retrans", "timeout", "reset", "error", "packetloss", "corruption"]),
        "net_delay": c(["rtt", "latency", "delay", "ping", "tnsping"]),
        "db": c(["tps", "login", "session", "conn", "connection", "sql", "query", "deadlock", "lock"]),
    }


def predict_reason_universal(
    dataset: str,
    instruction: str,
    adapter: Any = None,
) -> Tuple[str, Dict[str, Any]]:
    date_dir, start, end = parse_time_window_generic(instruction)
    date_dir = resolve_date_dir(dataset, date_dir)
    start_s = int(start.tz_convert("UTC").timestamp())
    end_s = int(end.tz_convert("UTC").timestamp())
    metric_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "metric")
    trace_path = os.path.join("dataset", dataset, "telemetry", date_dir, "trace", "trace_span.csv")

    top_metrics = _top_anomalous_metrics(metric_dir, start_s, end_s, top_k=30)
    hits = _keyword_hits(top_metrics)
    spread = _trace_spread_stats(trace_path, start_s, end_s)
    cands: List[str] = []
    try:
        from rca.adapter.registry import get_dataset_adapter

        adapter = get_dataset_adapter(dataset)
        if adapter is not None:
            cands = [str(x) for x in list(adapter.basil_reason_taxonomy() or []) if str(x).strip()]
    except Exception:
        cands = []
    if cands:
        hit_blob = " ".join([str(m).lower() for m in list(top_metrics or [])])

        def reason_score(reason: str) -> float:
            text = str(reason or "").lower()
            score = 0.0
            if "cpu" in text:
                score += float(hits.get("cpu", 0))
            if "mem" in text:
                score += float(hits.get("memory", 0))
            if "disk" in text or "i/o" in text or "io" in text:
                score += float(hits.get("disk", 0))
            if "loss" in text or "packet" in text:
                score += float(hits.get("net_loss", 0))
            if "delay" in text or "latency" in text:
                score += float(hits.get("net_delay", 0))
            if "db" in text or "connection" in text or "close" in text:
                score += float(hits.get("db", 0))
            # Weak exact-token support from metric names.
            for token in re.findall(r"[a-z0-9]+", text):
                if len(token) >= 3 and token in hit_blob:
                    score += 0.25
            return float(score)

        cands = sorted(dict.fromkeys(cands), key=reason_score, reverse=True)
    evidence = {
        "dataset": dataset,
        "date_dir": date_dir,
        "start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end": end.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_reasons": cands,
        "top_metrics": top_metrics[:20],
        "keyword_hits": hits,
        "trace_spread": spread,
    }
    # LLM call removed: evidence is returned as-is; the caller (field_solvers.solve_reason)
    # is responsible for the unified LLM rerank step.
    return (cands[0] if cands else ""), evidence
