from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rca.index.telemetry_cache import TelemetryCache
from rca.index.schemas import AnomalySegment, TraceTopologyEdge, component_type


_METRIC_BASELINE_CACHE: Dict[str, pd.DataFrame] = {}
_LISTDIR_CACHE: Dict[str, List[str]] = {}
_HEADER_CACHE: Dict[str, List[str]] = {}


def _cached_listdir(directory: str) -> List[str]:
    """In-process cache for os.listdir."""
    key = str(directory)
    if key not in _LISTDIR_CACHE:
        _LISTDIR_CACHE[key] = os.listdir(key)
    return list(_LISTDIR_CACHE[key])


def _cached_header(path: str) -> List[str]:
    """In-process cache for reading CSV headers."""
    key = str(path)
    if key not in _HEADER_CACHE:
        try:
            _HEADER_CACHE[key] = pd.read_csv(key, nrows=0).columns.tolist()
        except Exception:
            _HEADER_CACHE[key] = []
    return list(_HEADER_CACHE[key])


def _compute_p95_from_dir(metric_dir: str) -> pd.DataFrame:
    """Compute p95 baselines by file; optimized I/O via header+usecols caching."""

    if not os.path.isdir(metric_dir):
        return pd.DataFrame(columns=["_file", "component", "metric", "p95"])
    paths = [os.path.join(metric_dir, p) for p in _cached_listdir(metric_dir) if str(p).endswith(".csv")]
    frames = []
    for p in paths:
        try:
            long = _normalize_metric_file(p)
        except Exception:
            continue
        if long.empty:
            continue
        base = long.groupby(["component", "metric"], as_index=False)["value"].quantile(0.95).rename(columns={"value": "p95"})
        base["_file"] = os.path.basename(p)
        frames.append(base[["_file", "component", "metric", "p95"]])
    if not frames:
        return pd.DataFrame(columns=["_file", "component", "metric", "p95"])
    out = pd.concat(frames, ignore_index=True)
    out["_file"] = out["_file"].astype(str)
    out["component"] = out["component"].astype(str)
    out["metric"] = out["metric"].astype(str)
    out["p95"] = pd.to_numeric(out["p95"], errors="coerce")
    return out.dropna(subset=["p95"]).reset_index(drop=True)


def _get_metric_baseline(metric_dir: str) -> pd.DataFrame:
    """
    Return p95 baseline for metric_dir.
    Computes once per process, caches in-process for subsequent queries.
    """
    key = str(metric_dir)
    if key not in _METRIC_BASELINE_CACHE:
        _METRIC_BASELINE_CACHE[key] = _compute_p95_from_dir(key)
    return _METRIC_BASELINE_CACHE[key]


def get_cached_metric_baseline(metric_dir: str) -> pd.DataFrame:
    """
    Public accessor for the module-level metric baseline cache.
    Computes p95 baseline on first call, serves from cache on subsequent calls.
    Safe to call from any module in the same process.
    """
    return _get_metric_baseline(metric_dir)


def get_cached_listdir(directory: str) -> List[str]:
    """Public accessor for the module-level listdir cache."""
    return _cached_listdir(directory)


def get_cached_header(path: str) -> List[str]:
    """Public accessor for the module-level CSV header cache."""
    return _cached_header(path)


def _normalize_metric_file(path: str) -> pd.DataFrame:
    """Load metric CSV into (ts_s, component, metric, value) using minimal columns when possible."""
    from rca.adapter.utils.component_candidates import _normalize_metric_file as _normalize_metric_file_full

    header = _cached_header(path)
    if not header:
        return _normalize_metric_file_full(path)

    header_l = [str(c) for c in header]
    lower = [c.lower() for c in header_l]

    ts_col = ""
    for c, cl in zip(header_l, lower):
        if "timestamp" in cl or "time" in cl:
            ts_col = c
            break
    comp_col = ""
    for cand in ["cmdb_id", "component", "node", "instance"]:
        if cand in header_l:
            comp_col = cand
            break

    metric_col = ""
    for cand in ["metric", "kpi_name", "kpiName", "kpi", "kpiName", "name"]:
        if cand in header_l:
            metric_col = cand
            break
    value_col = ""
    for cand in ["value", "val", "metric_value"]:
        if cand in header_l:
            value_col = cand
            break

    if not ts_col or not comp_col:
        return _normalize_metric_file_full(path)

    is_long = bool(metric_col and value_col)
    usecols: Optional[List[str]] = None
    if is_long:
        usecols = [ts_col, comp_col, metric_col, value_col]
    else:
        exclude = {ts_col, comp_col}
        value_cols = [c for c in header_l if c not in exclude and c not in ["rr", "sr", "cnt", "mrt"]]
        usecols = [ts_col, comp_col] + value_cols if value_cols else None

    try:
        df = pd.read_csv(path, usecols=usecols, low_memory=False) if usecols else pd.read_csv(path, low_memory=False)
    except Exception:
        return _normalize_metric_file_full(path)

    if df.empty:
        return pd.DataFrame(columns=["ts_s", "component", "metric", "value"])

    cols = set(df.columns)
    if ts_col not in cols or comp_col not in cols:
        return _normalize_metric_file_full(path)

    if is_long and metric_col in cols and value_col in cols:
        out = df[[ts_col, comp_col, metric_col, value_col]].copy()
        out = out.rename(columns={ts_col: "ts", comp_col: "component", metric_col: "metric", value_col: "value"})
        out["ts_s"] = _timestamp_to_s(out["ts"])
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["ts_s", "component", "metric", "value"])
        out["component"] = out["component"].astype(str)
        out["metric"] = out["metric"].astype(str)
        res = out[["ts_s", "component", "metric", "value"]]
        if res.empty:
            return _normalize_metric_file_full(path)
        return res

    value_cols = [c for c in df.columns if c not in [ts_col, comp_col] and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        value_cols = [c for c in df.columns if c not in [ts_col, comp_col] and c not in ["rr", "sr", "cnt", "mrt"]]
    if not value_cols:
        return pd.DataFrame(columns=["ts_s", "component", "metric", "value"])

    out = df[[ts_col, comp_col] + list(value_cols)].copy()
    out = out.rename(columns={ts_col: "ts", comp_col: "component"})
    out["ts_s"] = _timestamp_to_s(out["ts"])
    out = out.dropna(subset=["ts_s", "component"])
    out["component"] = out["component"].astype(str)
    melted = out.melt(id_vars=["ts_s", "component"], value_vars=list(value_cols), var_name="metric", value_name="value")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["value"])
    melted["metric"] = melted["metric"].astype(str)
    res = melted[["ts_s", "component", "metric", "value"]]
    if res.empty:
        return _normalize_metric_file_full(path)
    return res


def _to_local_string(ts_s: int) -> str:
    return pd.to_datetime(int(ts_s), unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")


def _timestamp_to_s(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    v = float(s.dropna().iloc[0])
    if v > 10**12:
        return (s // 1000).astype("Int64")
    return s.astype("Int64")


def _parse_local_time_s(value: str, fallback: int) -> int:
    if not str(value or "").strip():
        return int(fallback)
    try:
        return int(pd.Timestamp(value, tz="Asia/Shanghai").tz_convert("UTC").timestamp())
    except Exception:
        return int(fallback)


def _component_channel(dataset: str, component: str) -> str:
    """Return a channel key grouping components of the same infrastructure layer.

    Telecom naming:
        os_*        → node
        docker_*    → container
        db_*        → db
        *           → other

    Market naming:
        node-N      (no dot)  → node   e.g. "node-1"
        node-N.svc  (dot)     → container  e.g. "node-6.adservice2-0"
        istio-* / *.source.* / *.destination.* → mesh
        *:port                → runtime
        *                     → service
    """
    ds = str(dataset or "").lower()
    c = str(component or "")
    if "telecom" in ds:
        if c.startswith("os_") or c.startswith("node-") and "." not in c:
            return "node"
        if c.startswith("docker_") or c.startswith("container"):
            return "container"
        if c.startswith("db_"):
            return "db"
        return "other"
    if "market" in ds:
        if ":" in c:
            return "runtime"
        if c.startswith("istio-") or ".source." in c or ".destination." in c:
            return "mesh"
        if c.startswith("node-"):
            return "node" if "." not in c else "container"
        if "-" in c:
            return "service"
        return "other"
    # Default: treat all as same channel (no cross-type bias correction)
    return "default"


def _normalize_scores_by_channel(
    scores: Dict[str, float], dataset: str
) -> Dict[str, float]:
    """Normalise component scores within each infrastructure channel to [0, 1].

    This prevents components with many monitoring metrics (e.g. OS nodes with
    51 KPIs) from always outscoring components with fewer metrics (e.g. docker
    containers with 9 KPIs) purely due to metric count, regardless of anomaly
    severity.  Only applied for Telecom and Market which have separate metric
    files per component type; Bank uses a single file and needs no correction.
    """
    ds = str(dataset or "").lower()
    if "telecom" not in ds and "market" not in ds:
        return dict(scores)

    # Group by channel
    channels: Dict[str, Dict[str, float]] = {}
    for comp, score in scores.items():
        ch = _component_channel(ds, comp)
        channels.setdefault(ch, {})[comp] = float(score)

    out: Dict[str, float] = {}
    for ch, ch_scores in channels.items():
        mx = max(ch_scores.values()) if ch_scores else 0.0
        denom = mx if mx > 1e-9 else 1.0
        for comp, score in ch_scores.items():
            out[comp] = score / denom
    return out


def _reason_hints_from_metrics(metrics: List[str]) -> List[str]:
    blob = " ".join([str(m).lower() for m in metrics])
    hints: List[str] = []
    if any(k in blob for k in ["loss", "drop", "retrans", "timeout", "reset", "error"]):
        hints.append("network loss")
    if any(k in blob for k in ["tnsping", "ping", "rtt", "latency", "delay"]):
        hints.append("network delay")
    if any(k in blob for k in ["login", "session", "sess", "tps", "exec", "connection", "conn"]):
        hints.append("db connection limit")
    if any(k in blob for k in ["close", "disconnect", "logout", "on_off", "shutdown", "restart"]):
        hints.append("db close")
    if any(k in blob for k in ["cpu", "load", "iowait", "proc", "util"]):
        hints.append("CPU fault")
    if any(k in blob for k in ["mem", "memory", "rss", "heap", "oom"]):
        hints.extend(["high memory usage", "container memory load", "node memory consumption"])
    if any(k in blob for k in ["disk", "iops", "await", "read", "write", "io_"]):
        hints.append("disk fault")
    out: List[str] = []
    for h in hints:
        if h not in out:
            out.append(h)
    return out


def extract_anomaly_segments(
    dataset: str,
    instruction: str,
    center_time_s: Optional[int] = None,
    delta_s: int = 300,
    severity_floor: float = 0.0,
    top_k_metrics: int = 8,
    enable_trace_segments: bool = True,
    metric_baseline_mode: str = "global_p95",
    ignore_metric_prefixes: Optional[List[str]] = None,
    priority_metric_keywords: Optional[List[str]] = None,
) -> Tuple[List[AnomalySegment], Dict[str, Any]]:
    """Extract BASIL anomaly segments from the current query window.

    The implementation reuses the existing telemetry scanners so BASIL remains
    consistent with the current solver while exposing indexable anomaly objects.
    """

    from rca.adapter.utils.component_candidates import (
        _estimate_failure_time_from_trace,
        _to_epoch_seconds,
        _trace_component_scores,
        parse_time_window_generic,
    )
    from rca.common.io import resolve_date_dir

    date_dir, start, end = parse_time_window_generic(instruction)
    date_dir = resolve_date_dir(dataset, date_dir)
    metric_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "metric")
    trace_path = os.path.join("dataset", dataset, "telemetry", date_dir, "trace", "trace_span.csv")
    start_s, end_s = _to_epoch_seconds(start), _to_epoch_seconds(end)
    if bool(enable_trace_segments):
        t_star = (
            int(center_time_s)
            if center_time_s is not None
            else _estimate_failure_time_from_trace(trace_path, start_s, end_s, dataset=dataset, date_dir=date_dir)
        )
    else:
        if center_time_s is not None:
            t_star = int(center_time_s)
        else:
            try:
                t_star = int((int(start_s) + int(end_s)) // 2)
            except Exception:
                t_star = int(start_s)
    delta_s = int(delta_s) if int(delta_s) > 0 else 300
    local_start_s = max(int(start_s), int(t_star) - delta_s)
    local_end_s = min(int(end_s), int(t_star) + delta_s)
    if local_end_s <= local_start_s:
        local_start_s, local_end_s = int(start_s), int(end_s)

    metric_scores: Dict[str, float] = {}
    peak_times: Dict[str, str] = {}
    first_times: Dict[str, str] = {}
    top_metrics: Dict[str, List[str]] = {}

    if os.path.isdir(metric_dir):
        paths = [os.path.join(metric_dir, p) for p in _cached_listdir(metric_dir) if str(p).endswith(".csv")]
    else:
        paths = []

    mode = str(metric_baseline_mode or "global_p95").strip().lower()
    if mode not in {"global_p95", "window_pre_p95", "auto"}:
        mode = "global_p95"

    baseline_global = _get_metric_baseline(metric_dir) if mode in {"global_p95", "auto"} else pd.DataFrame()
    prefixes = [str(x).lower() for x in (ignore_metric_prefixes or []) if str(x).strip()]
    prefix_tuple = tuple(prefixes) if prefixes else ()
    prio_kws = [str(x).lower() for x in (priority_metric_keywords or []) if str(x).strip()]
    prio_tuple = tuple(prio_kws) if prio_kws else ()

    def scan_metrics(baseline_mode: str) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
        mscores: Dict[str, float] = {}
        p_times: Dict[str, str] = {}
        f_times: Dict[str, str] = {}
        t_metrics: Dict[str, List[str]] = {}
        bmode = str(baseline_mode or "global_p95").strip().lower()

        for p in paths:
            try:
                long = _normalize_metric_file(p)
            except Exception:
                continue
            if long.empty:
                continue
            if prefix_tuple:
                try:
                    long = long[~long["metric"].astype(str).str.lower().str.startswith(prefix_tuple)].copy()
                except Exception:
                    pass
            win = long[(long["ts_s"] >= local_start_s) & (long["ts_s"] <= local_end_s)].copy()
            if win.empty:
                continue
            win["_file"] = os.path.basename(p)

            if bmode == "window_pre_p95":
                pre = win[win["ts_s"] < int(t_star)].copy()
                if pre.empty:
                    pre = win
                base_hi = (
                    pre.groupby(["component", "metric"], as_index=False)["value"]
                    .quantile(0.95)
                    .rename(columns={"value": "p95"})
                )
                base_lo = (
                    pre.groupby(["component", "metric"], as_index=False)["value"]
                    .quantile(0.05)
                    .rename(columns={"value": "p05"})
                )
                base = base_hi.merge(base_lo, on=["component", "metric"], how="left")
                base["_file"] = os.path.basename(p)
                win = win.merge(base, on=["_file", "component", "metric"], how="left").dropna(subset=["p95"]).copy()
            else:
                win = win.merge(baseline_global, on=["_file", "component", "metric"], how="left").dropna(subset=["p95"]).copy()

            if win.empty:
                continue
            denom_hi = win["p95"].abs().where(win["p95"].abs() > 1e-6, 1.0)
            above = (win["value"] - win["p95"]) / denom_hi
            below = pd.Series(0.0, index=win.index)
            if "p05" in win.columns:
                denom_lo = win["p05"].abs().where(win["p05"].abs() > 1e-6, 1.0)
                below = (win["p05"] - win["value"]) / denom_lo
            win["breach_ratio"] = pd.concat([above, below], axis=1).max(axis=1)
            win["breach_ratio"] = pd.to_numeric(win["breach_ratio"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=10.0)
            win = win[win["breach_ratio"] > 0].copy()
            if win.empty:
                continue

            by_metric = (
                win.groupby(["component", "metric"], as_index=False)["breach_ratio"]
                .sum()
                .sort_values(["component", "breach_ratio"], ascending=[True, False])
            )
            topm = by_metric.groupby("component", as_index=False).head(int(top_k_metrics))
            by_comp = topm.groupby("component", as_index=False)["breach_ratio"].sum()
            # Per-file normalisation: scale this file's component scores to [0, 1]
            # before accumulating into mscores.  Without this, metric_node.csv (51
            # metrics) always outscores metric_container.csv (9 metrics) regardless
            # of anomaly severity, which causes os_ nodes to dominate the candidate
            # pool even when the root cause is a docker/service component.
            file_max = float(by_comp["breach_ratio"].max()) if not by_comp.empty else 0.0
            norm_denom = file_max if file_max > 1e-9 else 1.0
            for _, r in by_comp.iterrows():
                c = str(r["component"])
                base = float(r["breach_ratio"]) / norm_denom  # normalised to [0, 1]
                bonus = 0.0
                if prio_tuple:
                    try:
                        mlist = topm[topm["component"].astype(str) == c]["metric"].astype(str).str.lower()
                        hit = sum(int(any(k in m for k in prio_tuple)) for m in list(mlist))
                        if hit > 0:
                            bonus = 0.12 * base
                    except Exception:
                        bonus = 0.0
                mscores[c] = mscores.get(c, 0.0) + base + bonus

            win["minute"] = (win["ts_s"] // 60) * 60
            for c, g in win.groupby("component"):
                m = g.groupby("minute", as_index=False)["breach_ratio"].sum().sort_values("breach_ratio", ascending=False)
                if not m.empty and str(c) not in p_times:
                    peak_s = int(m.iloc[0]["minute"])
                    p_times[str(c)] = (
                        pd.to_datetime(peak_s, unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
                    )
                if not m.empty and str(c) not in f_times:
                    ms = g.groupby("minute", as_index=False)["breach_ratio"].sum().sort_values("minute")
                    mx = float(ms["breach_ratio"].max()) if not ms.empty else 0.0
                    if mx > 0:
                        thr = 0.30 * mx
                        cand = ms[ms["breach_ratio"] >= thr]
                        if not cand.empty:
                            first_s = int(cand.iloc[0]["minute"])
                            f_times[str(c)] = (
                                pd.to_datetime(first_s, unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
                            )
                mt = (
                    g.groupby("metric", as_index=False)["breach_ratio"]
                    .sum()
                    .sort_values("breach_ratio", ascending=False)
                    .head(top_k_metrics)["metric"]
                    .astype(str)
                    .tolist()
                )
                if mt and str(c) not in t_metrics:
                    t_metrics[str(c)] = mt

        return mscores, p_times, f_times, t_metrics

    if paths:
        metric_scores, peak_times, first_times, top_metrics = scan_metrics("global_p95" if mode != "window_pre_p95" else "window_pre_p95")
        if mode == "auto" and not metric_scores:
            metric_scores, peak_times, first_times, top_metrics = scan_metrics("window_pre_p95")
        # Per-channel normalisation: within each infrastructure layer (node /
        # container / db / mesh / runtime) scores are re-scaled to [0, 1] so
        # that a component type with fewer monitored KPIs can still compete
        # against one with many KPIs.  Applied only for Telecom and Market.
        metric_scores = _normalize_scores_by_channel(metric_scores, dataset)

    trace_scores: Dict[str, float] = {}
    if bool(enable_trace_segments):
        trace_scores = _trace_component_scores(trace_path, local_start_s, local_end_s, dataset=dataset, date_dir=date_dir)

    segments: List[AnomalySegment] = []
    seen_ids = set()

    for comp, sev in sorted(metric_scores.items(), key=lambda kv: kv[1], reverse=True):
        if float(sev or 0.0) <= float(severity_floor):
            continue
        metrics = [str(x) for x in top_metrics.get(comp, [])]
        first_s = _parse_local_time_s(first_times.get(comp, ""), local_start_s)
        peak_s = _parse_local_time_s(peak_times.get(comp, ""), int(t_star))
        sig = metrics[0] if metrics else "metric"
        sid = f"{dataset}:{date_dir}:metric:{comp}:{first_s}:{peak_s}:{sig}"
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        segments.append(
            AnomalySegment(
                segment_id=sid,
                dataset=dataset,
                date_dir=date_dir,
                component=str(comp),
                component_type=component_type(comp),
                signal_type="metric",
                signal_name=sig,
                start_s=int(first_s),
                peak_s=int(peak_s),
                end_s=int(local_end_s),
                severity=float(sev or 0.0),
                reason_hints=_reason_hints_from_metrics(metrics),
                top_metrics=metrics,
                metadata={"local_start_s": local_start_s, "local_end_s": local_end_s},
            )
        )

    if bool(enable_trace_segments):
        for comp, sev in sorted(trace_scores.items(), key=lambda kv: kv[1], reverse=True):
            if float(sev or 0.0) <= float(severity_floor):
                continue
            sid = f"{dataset}:{date_dir}:trace:{comp}:{local_start_s}:{t_star}"
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            segments.append(
                AnomalySegment(
                    segment_id=sid,
                    dataset=dataset,
                    date_dir=date_dir,
                    component=str(comp),
                    component_type=component_type(comp),
                    signal_type="trace",
                    signal_name="trace_duration_p95",
                    start_s=int(local_start_s),
                    peak_s=int(t_star),
                    end_s=int(local_end_s),
                    severity=float(sev or 0.0),
                    reason_hints=[],
                    top_metrics=[],
                    metadata={"local_start_s": local_start_s, "local_end_s": local_end_s},
                )
            )

    context = {
        "dataset": dataset,
        "date_dir": date_dir,
        "start_local": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_local": end.strftime("%Y-%m-%d %H:%M:%S"),
        "start_s": int(start_s),
        "end_s": int(end_s),
        "local_start_s": int(local_start_s),
        "local_end_s": int(local_end_s),
        "t_star_s": int(t_star),
        "t_star_local": _to_local_string(int(t_star)),
        "metric_scores": dict(metric_scores),
        "trace_scores": dict(trace_scores),
        "peak_times": dict(peak_times),
        "first_times": dict(first_times),
        "top_metrics": dict(top_metrics),
        "trace_path": trace_path,
    }
    return segments, context


def _trace_columns(path: str) -> Dict[str, Optional[str]]:
    try:
        cols = list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        cols = []
    colset = set(cols)
    return {
        "ts": "timestamp" if "timestamp" in colset else ("startTime" if "startTime" in colset else None),
        "trace": "trace_id" if "trace_id" in colset else ("traceId" if "traceId" in colset else None),
        "component": "cmdb_id" if "cmdb_id" in colset else ("serviceName" if "serviceName" in colset else None),
        "span": "span_id" if "span_id" in colset else ("id" if "id" in colset else None),
        "parent": "parent_span" if "parent_span" in colset else ("parent_id" if "parent_id" in colset else ("pid" if "pid" in colset else None)),
        "ds": "dsName" if "dsName" in colset else None,
        "service": "serviceName" if "serviceName" in colset else None,
    }


def extract_trace_topology_edges(
    dataset: str,
    date_dir: str,
    start_s: int,
    end_s: int,
    max_rows: Optional[int] = None,
) -> List[TraceTopologyEdge]:
    """Extract typed topology edges from trace spans.

    Edges are propagation candidates. For parent-child spans, latency anomalies
    are modeled as propagating from child/callee to parent/caller. For Telecom
    JDBC spans, db service names in dsName are connected to the owning component.
    """

    trace_path = os.path.join("dataset", dataset, "telemetry", date_dir, "trace", "trace_span.csv")
    if not os.path.exists(trace_path):
        return []
    try:
        edge_df = TelemetryCache.get_topology_edges(dataset, date_dir, trace_path, max_rows=max_rows)
    except Exception:
        return []
    if edge_df.empty:
        return []
    start_minute = int((int(start_s) // 60) * 60)
    end_minute = int((int(end_s) // 60) * 60)
    edge_df = edge_df[(edge_df["minute"] >= start_minute) & (edge_df["minute"] <= end_minute)].copy()
    if edge_df.empty:
        return []

    edge_counts: Counter[Tuple[str, str, str]] = Counter()
    for _, row in edge_df.iterrows():
        src = str(row.get("src") or "").strip()
        dst = str(row.get("dst") or "").strip()
        edge_type = str(row.get("edge_type") or "").strip()
        count = int(row.get("support_count") or 0)
        if src and dst and edge_type and src != dst and count > 0:
            edge_counts[(src, dst, edge_type)] += count

    return [
        TraceTopologyEdge(src=s, dst=d, edge_type=t, support_count=int(n))
        for (s, d, t), n in edge_counts.most_common()
    ]
