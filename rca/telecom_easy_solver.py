import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


_MONTH = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _telecom_candidates() -> Tuple[List[str], List[str], List[str]]:
    nodes = [f"os_{i:03d}" for i in range(1, 23)]
    pods = [f"docker_{i:03d}" for i in range(1, 9)]
    svcs = [f"db_{i:03d}" for i in range(1, 14)]
    return nodes, pods, svcs


def _parse_time_window(instruction: str) -> Tuple[str, pd.Timestamp, pd.Timestamp]:
    text = str(instruction)
    date_m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", text)
    if date_m:
        month_name, day, year = date_m.groups()
        month = _MONTH[month_name.lower()]
        day = int(day)
        year = int(year)
        date_dir = f"{year:04d}_{month:02d}_{day:02d}"
        hm_m = re.search(r"from\s*(\d{2}:\d{2})\s*to\s*(\d{2}:\d{2})", text)
        if not hm_m:
            hm_m = re.search(r"between\s*(\d{2}:\d{2})\s*and\s*(\d{2}:\d{2})", text)
        if hm_m:
            start_hm, end_hm = hm_m.groups()
            start = pd.Timestamp(f"{year:04d}-{month:02d}-{day:02d} {start_hm}:00", tz="Asia/Shanghai")
            end = pd.Timestamp(f"{year:04d}-{month:02d}-{day:02d} {end_hm}:00", tz="Asia/Shanghai")
            return date_dir, start, end

    m = re.search(r"(\d{4}-\d{2}-\d{2}).*from\s*(\d{2}:\d{2})\s*to\s*(\d{2}:\d{2})", text)
    if m:
        date_str, start_hm, end_hm = m.groups()
        dt = pd.Timestamp(date_str)
        date_dir = dt.strftime("%Y_%m_%d")
        start = pd.Timestamp(f"{date_str} {start_hm}:00", tz="Asia/Shanghai")
        end = pd.Timestamp(f"{date_str} {end_hm}:00", tz="Asia/Shanghai")
        return date_dir, start, end

    raise ValueError("Cannot parse time window from instruction.")


def _to_tz_timestamp_ms(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")


def _load_metric(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = _to_tz_timestamp_ms(df["timestamp"])
    return df


def _to_seconds_timestamp(series: pd.Series) -> pd.Series:
    v = series.astype(int)
    return v.where(v.astype(str).str.len() != 13, v // 1000)


def _metric_window_series_map(metric_dir: str, start: pd.Timestamp, end: pd.Timestamp, max_per_component: int = 3) -> Dict[Tuple[str, str], pd.Series]:
    frames = []
    for filename in ["metric_node.csv", "metric_container.csv", "metric_service.csv"]:
        path = os.path.join(metric_dir, filename)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df[["cmdb_id", "name", "timestamp", "value"]].dropna()
        df["timestamp"] = _to_seconds_timestamp(df["timestamp"])
        df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
        df = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()
        if not df.empty:
            frames.append(df[["cmdb_id", "name", "timestamp", "value"]])
    if not frames:
        return {}

    df_all = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    grouped = df_all.groupby(["cmdb_id", "name"])
    series_map: Dict[Tuple[str, str], pd.Series] = {}
    for (comp, metric), g in grouped:
        s = g.set_index("timestamp")["value"].astype(float)
        if len(s) >= 20:
            series_map[(str(comp), str(metric))] = s

    by_comp: Dict[str, int] = {}
    filtered: Dict[Tuple[str, str], pd.Series] = {}
    for (comp, metric), s in sorted(series_map.items(), key=lambda kv: kv[1].var(), reverse=True):
        c = by_comp.get(comp, 0)
        if c < max_per_component:
            filtered[(comp, metric)] = s
            by_comp[comp] = c + 1
    return filtered


def _granger_edges(series_map: Dict[Tuple[str, str], pd.Series], max_pairs: int = 200, max_lag: int = 2, p_thresh: float = 0.05):
    keys = list(series_map.keys())
    edges = []
    pairs = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if pairs >= max_pairs:
                break
            k1, k2 = keys[i], keys[j]
            s1, s2 = series_map[k1], series_map[k2]
            idx = s1.index.intersection(s2.index)
            if len(idx) < 25:
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
                edges.append((k1[0], k2[0], float(max(0.1, 1.0 - best_xy_p))))
            if best_yx_p < p_thresh:
                edges.append((k2[0], k1[0], float(max(0.1, 1.0 - best_yx_p))))
            pairs += 1
        if pairs >= max_pairs:
            break
    return edges


def pick_component_by_granger(instruction: str) -> str:
    date_dir, start, end = _parse_time_window(instruction)
    metric_dir = os.path.join("dataset", "Telecom", "telemetry", date_dir, "metric")
    series_map = _metric_window_series_map(metric_dir, start, end, max_per_component=3)
    if not series_map:
        return "os_001"
    edges = _granger_edges(series_map, max_pairs=250, max_lag=2, p_thresh=0.05)
    if not edges:
        ranked = _rank_components(metric_dir, start, end)
        return str(ranked.iloc[0]["component"]) if not ranked.empty else "os_001"
    out_score: Dict[str, float] = {}
    in_score: Dict[str, float] = {}
    for src, tgt, conf in edges:
        out_score[src] = out_score.get(src, 0.0) + conf
        in_score[tgt] = in_score.get(tgt, 0.0) + conf
    comps = set(list(out_score.keys()) + list(in_score.keys()))
    best = None
    best_score = None
    for comp in comps:
        score = out_score.get(comp, 0.0) - 0.4 * in_score.get(comp, 0.0)
        if best_score is None or score > best_score:
            best_score = score
            best = comp
    return str(best) if best else "os_001"


def _find_faults(metric_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame(columns=["cmdb_id", "name", "first_fault_time", "event_count", "max_breach_ratio"])

    thresholds = (
        metric_df.groupby(["cmdb_id", "name"], as_index=False)["value"]
        .quantile(0.95)
        .rename(columns={"value": "global_p95"})
    )
    window = metric_df[(metric_df["timestamp"] >= start) & (metric_df["timestamp"] <= end)].copy()
    if window.empty:
        return pd.DataFrame(columns=["cmdb_id", "name", "first_fault_time", "event_count", "max_breach_ratio"])
    window = window.merge(thresholds, on=["cmdb_id", "name"], how="left")
    window = window[window["value"] > window["global_p95"]].copy()
    if window.empty:
        return pd.DataFrame(columns=["cmdb_id", "name", "first_fault_time", "event_count", "max_breach_ratio"])

    window["breach_ratio"] = 1.0
    mask = window["global_p95"].abs() > 1e-6
    window.loc[mask, "breach_ratio"] = (
        (window.loc[mask, "value"] - window.loc[mask, "global_p95"]) / window.loc[mask, "global_p95"].abs()
    )

    window = window.sort_values(["cmdb_id", "name", "timestamp"])
    window["prev_ts"] = window.groupby(["cmdb_id", "name"])["timestamp"].shift(1)
    window["gap_sec"] = (window["timestamp"] - window["prev_ts"]).dt.total_seconds()
    window["new_group"] = window["gap_sec"].isna() | (window["gap_sec"] > 90)
    window["group_id"] = window.groupby(["cmdb_id", "name"])["new_group"].cumsum()
    runs = (
        window.groupby(["cmdb_id", "name", "group_id"], as_index=False)
        .agg(
            first_fault_time=("timestamp", "min"),
            event_count=("timestamp", "size"),
            max_breach_ratio=("breach_ratio", "max"),
        )
    )
    runs = runs[runs["event_count"] >= 2].copy()
    if runs.empty:
        return pd.DataFrame(columns=["cmdb_id", "name", "first_fault_time", "event_count", "max_breach_ratio"])
    return runs


def _rank_components(metric_dir: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    nodes, pods, svcs = _telecom_candidates()
    allowed = set(nodes + pods + svcs)

    parts = []
    for filename, level in [("metric_container.csv", "pod"), ("metric_service.csv", "service"), ("metric_node.csv", "node")]:
        path = os.path.join(metric_dir, filename)
        if not os.path.exists(path):
            continue
        df = _load_metric(path)
        faults = _find_faults(df, start, end)
        if faults.empty:
            continue
        faults = faults[faults["cmdb_id"].isin(allowed)].copy()
        if faults.empty:
            continue
        faults["level"] = level
        parts.append(faults)

    if not parts:
        return pd.DataFrame(columns=["component", "level", "first_fault_time", "fault_group_count", "event_count", "max_breach_ratio", "score"])

    all_faults = pd.concat(parts, ignore_index=True)
    comp = (
        all_faults.groupby(["cmdb_id", "level"], as_index=False)
        .agg(
            first_fault_time=("first_fault_time", "min"),
            fault_group_count=("group_id", "nunique"),
            event_count=("event_count", "sum"),
            max_breach_ratio=("max_breach_ratio", "max"),
        )
        .rename(columns={"cmdb_id": "component"})
    )

    level_bonus = comp["level"].map({"pod": 2.0, "service": 1.5, "node": 0.0}).fillna(0.0)
    node_component_count = comp.groupby("level")["component"].transform("count")
    node_noise_penalty = ((comp["level"] == "node") & (node_component_count >= 6)).astype(float) * 2.0
    onset_penalty = ((comp["first_fault_time"] - start).dt.total_seconds() / 60.0).clip(lower=0)
    comp["score"] = (
        comp["fault_group_count"] * 6.0
        + comp["event_count"] * 0.3
        + comp["max_breach_ratio"].clip(lower=0, upper=10) * 2.0
        + level_bonus
        - onset_penalty * 0.15
        - node_noise_penalty
    )
    comp = comp.sort_values(["score", "first_fault_time"], ascending=[False, True]).reset_index(drop=True)
    return comp


def _infer_reason(component: str, kpi_names: List[str]) -> str:
    comp = str(component)
    names = " ".join([str(x).lower() for x in kpi_names])
    if comp.startswith("docker_"):
        return "CPU fault"
    if comp.startswith("db_"):
        if any(k in names for k in ["connect", "session", "client", "thread", "queue"]):
            return "db connection limit"
        if any(k in names for k in ["close", "disconnect", "reset"]):
            return "db close"
        return "db connection limit"
    if comp.startswith("os_"):
        if any(k in names for k in ["error", "errors", "drop", "loss", "retrans"]):
            return "network loss"
        if any(k in names for k in ["ping", "rtt", "latency", "delay"]):
            return "network delay"
        if any(k in names for k in ["traffic", "recv", "send", "network"]):
            return "network delay"
        if any(k in names for k in ["cpu", "load", "util"]):
            return "CPU fault"
        return "network delay"
    return "CPU fault"


def _reason_from_window(metric_dir: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    indicators = {
        "network_loss": ["error", "errors", "drop", "loss", "retrans", "sent_errors_packets", "recv_errors_packets"],
        "network_delay": ["ping", "rtt", "latency", "delay", "tnsping"],
        "db_conn": ["sess_connect", "session", "login_per_sec", "call_per_sec"],
        "db_close": ["on_off_state", "hang", "shutdown", "restart", "close", "disconnect"],
        "cpu": ["cpu_util", "cpu_used", "cpu_idle", "processor_load"],
    }

    def scan_file(filename: str) -> List[str]:
        path = os.path.join(metric_dir, filename)
        if not os.path.exists(path):
            return []
        df = _load_metric(path)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
        if df.empty:
            return []
        thresholds = (
            df.groupby(["cmdb_id", "name"], as_index=False)["value"]
            .quantile(0.95)
            .rename(columns={"value": "p95"})
        )
        df = df.merge(thresholds, on=["cmdb_id", "name"], how="left")
        df = df[df["value"] > df["p95"]].copy()
        if df.empty:
            return []
        return df["name"].dropna().astype(str).tolist()

    names = []
    names += scan_file("metric_node.csv")
    names += scan_file("metric_container.csv")
    names += scan_file("metric_service.csv")
    text = " ".join([n.lower() for n in names])

    if any(k in text for k in indicators["network_loss"]):
        return "network loss"
    if any(k in text for k in indicators["network_delay"]):
        return "network delay"
    if any(k in text for k in indicators["db_close"]):
        return "db close"
    if any(k in text for k in indicators["db_conn"]):
        return "db connection limit"
    if any(k in text for k in indicators["cpu"]):
        return "CPU fault"
    return "CPU fault"


def _pick_component_for_reason(metric_dir: str, start: pd.Timestamp, end: pd.Timestamp, reason: str) -> str:
    nodes, pods, svcs = _telecom_candidates()
    reason = str(reason)
    if reason in ("network delay", "network loss"):
        preferred = nodes
        filename = "metric_node.csv"
        keywords = ["ping", "traffic", "send", "recv", "network", "errors", "drop", "loss"]
    elif reason in ("db connection limit", "db close"):
        preferred = svcs
        filename = "metric_service.csv"
        keywords = ["sess", "login", "call", "on_off", "hang", "dbtime", "tnsping"]
    else:
        preferred = pods
        filename = "metric_container.csv"
        keywords = ["cpu", "load", "util"]

    path = os.path.join(metric_dir, filename)
    if not os.path.exists(path):
        return preferred[0]
    df = _load_metric(path)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end) & (df["cmdb_id"].isin(preferred))].copy()
    if df.empty:
        return preferred[0]
    df["name_l"] = df["name"].astype(str).str.lower()
    mask = False
    for kw in keywords:
        mask = mask | df["name_l"].str.contains(kw, na=False)
    df = df[mask].copy()
    if df.empty:
        return preferred[0]
    counts = df.groupby("cmdb_id", as_index=False).size().sort_values("size", ascending=False)
    return str(counts.iloc[0]["cmdb_id"])


def _pick_datetime(metric_dir: str, start: pd.Timestamp, end: pd.Timestamp, component: str, reason: str) -> pd.Timestamp:
    reason = str(reason)
    if reason in ("network delay", "network loss"):
        filename = "metric_node.csv"
        keywords = ["ping", "traffic", "send", "recv", "network", "errors", "drop", "loss"]
    elif reason in ("db connection limit", "db close"):
        filename = "metric_service.csv"
        keywords = ["sess", "login", "call", "on_off", "hang", "dbtime", "tnsping"]
    else:
        filename = "metric_container.csv"
        keywords = ["cpu", "load", "util"]
    path = os.path.join(metric_dir, filename)
    if not os.path.exists(path):
        return start.floor("min")
    df = _load_metric(path)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end) & (df["cmdb_id"] == component)].copy()
    if df.empty:
        return start.floor("min")
    df["name_l"] = df["name"].astype(str).str.lower()
    mask = False
    for kw in keywords:
        mask = mask | df["name_l"].str.contains(kw, na=False)
    df = df[mask].copy()
    if df.empty:
        return start.floor("min")
    thresholds = (
        df.groupby(["cmdb_id", "name"], as_index=False)["value"]
        .quantile(0.95)
        .rename(columns={"value": "p95"})
    )
    df = df.merge(thresholds, on=["cmdb_id", "name"], how="left")
    df = df[df["value"] > df["p95"]].copy()
    if df.empty:
        return start.floor("min")
    df["breach"] = df["value"] - df["p95"]
    peak_ts = df.sort_values("breach", ascending=False).iloc[0]["timestamp"]
    return pd.Timestamp(peak_ts).floor("min")


def solve_telecom_easy(task_id: int, instruction: str) -> str:
    date_dir, start, end = _parse_time_window(instruction)
    metric_dir = os.path.join("dataset", "Telecom", "telemetry", date_dir, "metric")
    reason = _reason_from_window(metric_dir, start, end)
    if task_id == 3:
        ranked = _rank_components(metric_dir, start, end)
        component = str(ranked.iloc[0]["component"]) if not ranked.empty else "os_001"
        onset = start.floor("min")
    else:
        component = _pick_component_for_reason(metric_dir, start, end, reason)
        onset = _pick_datetime(metric_dir, start, end, component, reason)

    if task_id == 1:
        return (
            "{"
            "\"1\": {"
            f"\"root cause occurrence datetime\": \"{onset.strftime('%Y-%m-%d %H:%M:%S')}\""
            "}"
            "}"
        )
    if task_id == 2:
        return (
            "{"
            "\"1\": {"
            f"\"root cause reason\": \"{reason}\""
            "}"
            "}"
        )
    if task_id == 3:
        return (
            "{"
            "\"1\": {"
            f"\"root cause component\": \"{component}\""
            "}"
            "}"
        )

    raise ValueError("solve_telecom_easy only supports task_id 1-3.")


def estimate_task1_datetime_from_trace(instruction: str) -> str:
    date_dir, start, end = _parse_time_window(instruction)
    trace_path = os.path.join("dataset", "Telecom", "telemetry", date_dir, "trace", "trace_span.csv")
    if not os.path.exists(trace_path):
        return start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(trace_path, low_memory=False)
    if "startTime" not in df.columns or "elapsedTime" not in df.columns:
        return start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["startTime"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    df = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()
    if df.empty:
        return start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
    df["minute"] = df["ts"].dt.floor("min")
    p95 = df.groupby("minute")["elapsedTime"].quantile(0.95)
    if p95.empty:
        return start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
    p95 = p95.sort_index()
    baseline = float(p95.median())
    threshold = baseline * 1.08
    candidates = p95[p95 >= threshold]
    if not candidates.empty:
        chosen = candidates.index[0]
        return pd.Timestamp(chosen).strftime("%Y-%m-%d %H:%M:%S")
    chosen = p95.sort_values(ascending=False).index[0]
    return pd.Timestamp(chosen).strftime("%Y-%m-%d %H:%M:%S")
