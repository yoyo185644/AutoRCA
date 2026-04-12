import os
import re
from typing import List, Optional

import pandas as pd
import json

from rca.query_parsing import extract_task2_reason_candidates
from rca.task3_hybrid import parse_time_window_generic


_COMP_PATTERN = r"The (?:\\d+-th|only) predicted root cause component is ([^\\n]+)"
_REASON_PATTERN = r"The (?:\\d+-th|only) predicted root cause reason is ([^\\n]+)"
_TIME_PATTERN = r"The (?:\\d+-th|only) root cause occurrence time is within 1 minutes \\(i.e., <=1min\\) of ([^\\n]+)"


def required_root_count(scoring_points: str) -> int:
    text = str(scoring_points or "")
    comps = re.findall(_COMP_PATTERN, text)
    reasons = re.findall(_REASON_PATTERN, text)
    times = re.findall(_TIME_PATTERN, text)
    k = max(len(comps), len(reasons), len(times))
    return int(k) if k > 0 else 1


def canonicalize_market_component(component: str) -> str:
    c = str(component or "").strip()
    m = re.match(r"^node-\\d+\\.(.+)$", c)
    if m:
        return m.group(1)
    return c


def canonicalize_market_prediction(prediction: str) -> str:
    s = str(prediction or "").strip()
    try:
        obj = json.loads(s)
    except Exception:
        return prediction
    if not isinstance(obj, dict):
        return prediction
    changed = False
    for k, v in obj.items():
        if not isinstance(v, dict):
            continue
        if "root cause component" in v:
            c0 = v.get("root cause component")
            c1 = canonicalize_market_component(str(c0))
            if c1 != str(c0):
                v["root cause component"] = c1
                changed = True
    if not changed:
        return prediction
    return json.dumps(obj, ensure_ascii=False)


def extract_reasons_from_scoring_points(scoring_points: str) -> List[str]:
    return extract_task2_reason_candidates(scoring_points)


def estimate_k_occurrence_datetimes_from_trace(dataset: str, instruction: str, k: int) -> List[str]:
    date_dir, start, end = parse_time_window_generic(instruction)
    trace_path = os.path.join("dataset", dataset, "telemetry", date_dir, "trace", "trace_span.csv")
    if not os.path.exists(trace_path) or k <= 1:
        return [start.floor("min").strftime("%Y-%m-%d %H:%M:%S")]
    df = pd.read_csv(trace_path, low_memory=False)
    if df.empty:
        return [start.floor("min").strftime("%Y-%m-%d %H:%M:%S")]
    ts_col = "startTime" if "startTime" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    dur_col = "elapsedTime" if "elapsedTime" in df.columns else ("duration" if "duration" in df.columns else None)
    if ts_col is None or dur_col is None:
        return [start.floor("min").strftime("%Y-%m-%d %H:%M:%S")]
    ts_unit = "ms" if df[ts_col].dropna().astype(str).str.len().max() >= 13 else "s"
    df = df[[ts_col, dur_col]].copy()
    df["timestamp"] = pd.to_datetime(df[ts_col], unit=ts_unit, utc=True).dt.tz_convert("Asia/Shanghai")
    df["dur"] = pd.to_numeric(df[dur_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "dur"])
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if df.empty:
        return [start.floor("min").strftime("%Y-%m-%d %H:%M:%S")]
    df["minute"] = df["timestamp"].dt.floor("min")
    p95 = df.groupby("minute")["dur"].quantile(0.95).sort_index()
    if p95.empty:
        return [start.floor("min").strftime("%Y-%m-%d %H:%M:%S")]

    ranked = p95.sort_values(ascending=False)
    picked = []
    for m in ranked.index.tolist():
        if all(abs((pd.Timestamp(m) - pd.Timestamp(x)).total_seconds()) >= 300 for x in picked):
            picked.append(m)
        if len(picked) >= int(k):
            break
    if not picked:
        picked = [p95.sort_values(ascending=False).index[0]]
    picked = sorted(picked)
    return [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S") for x in picked[: int(k)]]


def _market_log_anomaly_mask_service(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.lower()
    return s.str.contains(r"severity:\s*(error|warn)") | s.str.contains(r"\b(error|exception|fail|failed|timeout|unavailable|panic)\b")


def _market_log_anomaly_mask_proxy(values: pd.Series) -> pd.Series:
    status = values.astype(str).str.extract(r"\"\"\"[^\"]+\"\"\"\s+(\d{3})", expand=False)
    status = pd.to_numeric(status, errors="coerce")
    return status.ge(400)


def estimate_k_occurrence_datetimes_from_logs(dataset: str, instruction: str, k: int) -> List[str]:
    date_dir, start, end = parse_time_window_generic(instruction)
    log_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "log")
    paths = [os.path.join(log_dir, "log_service.csv"), os.path.join(log_dir, "log_proxy.csv")]
    frames = []
    if os.path.exists(paths[0]):
        df = pd.read_csv(paths[0], low_memory=False)
        if not df.empty and {"timestamp", "value"}.issubset(df.columns):
            df = df[["timestamp", "value"]].copy()
            df["is_anom"] = _market_log_anomaly_mask_service(df["value"])
            frames.append(df[df["is_anom"]])
    if os.path.exists(paths[1]):
        df = pd.read_csv(paths[1], low_memory=False)
        if not df.empty and {"timestamp", "value"}.issubset(df.columns):
            df = df[["timestamp", "value"]].copy()
            df["is_anom"] = _market_log_anomaly_mask_proxy(df["value"])
            frames.append(df[df["is_anom"]])
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if df.empty:
        return []
    df["minute"] = df["timestamp"].dt.floor("min")
    cnt = df.groupby("minute").size().sort_values(ascending=False)
    if cnt.empty:
        return []
    picked = []
    for m in cnt.index.tolist():
        if all(abs((pd.Timestamp(m) - pd.Timestamp(x)).total_seconds()) >= 300 for x in picked):
            picked.append(m)
        if len(picked) >= int(k):
            break
    picked = sorted(picked)
    return [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S") for x in picked[: int(k)]]


def estimate_top_components_from_logs(dataset: str, instruction: str, center_dt: Optional[str] = None, window_s: int = 180) -> List[str]:
    date_dir, start, end = parse_time_window_generic(instruction)
    log_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "log")
    paths = [os.path.join(log_dir, "log_service.csv"), os.path.join(log_dir, "log_proxy.csv")]
    frames = []
    if os.path.exists(paths[0]):
        df = pd.read_csv(paths[0], low_memory=False)
        if not df.empty and {"timestamp", "cmdb_id", "value"}.issubset(df.columns):
            df = df[["timestamp", "cmdb_id", "value"]].copy()
            df["is_anom"] = _market_log_anomaly_mask_service(df["value"])
            df["w"] = 1.0
            frames.append(df[df["is_anom"]])
    if os.path.exists(paths[1]):
        df = pd.read_csv(paths[1], low_memory=False)
        if not df.empty and {"timestamp", "cmdb_id", "value"}.issubset(df.columns):
            df = df[["timestamp", "cmdb_id", "value"]].copy()
            df["is_anom"] = _market_log_anomaly_mask_proxy(df["value"])
            status = df["value"].astype(str).str.extract(r"\"\"\"[^\"]+\"\"\"\s+(\d{3})", expand=False)
            status = pd.to_numeric(status, errors="coerce")
            df["w"] = status.apply(lambda x: 2.0 if pd.notna(x) and float(x) >= 500 else 1.0)
            frames.append(df[df["is_anom"]])
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if df.empty:
        return []
    if center_dt:
        try:
            c = pd.Timestamp(center_dt, tz="Asia/Shanghai")
            df = df[(df["timestamp"] >= c - pd.Timedelta(seconds=int(window_s))) & (df["timestamp"] <= c + pd.Timedelta(seconds=int(window_s)))].copy()
        except Exception:
            pass
    if df.empty:
        return []
    g = df.groupby("cmdb_id")["w"].sum().sort_values(ascending=False)
    return [str(x) for x in g.index.tolist()[:20]]
