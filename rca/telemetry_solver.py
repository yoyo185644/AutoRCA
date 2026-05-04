import os
import re
from typing import List, Optional

import pandas as pd
import json

from rca.query_parsing import extract_reason_candidates
from rca.component_candidates import parse_time_window_generic


_COMP_PATTERN = r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
_REASON_PATTERN = r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
_TIME_PATTERN = r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)"


def _read_csv_with_python_fallback(path: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        msg = str(exc)
        if "Try engine='python'" not in msg and "Error tokenizing data" not in msg:
            raise
        kwargs = dict(kwargs)
        kwargs["engine"] = "python"
        kwargs.pop("low_memory", None)
        return pd.read_csv(path, **kwargs)


def required_root_count(scoring_points: str) -> int:
    text = str(scoring_points or "")
    comps = re.findall(_COMP_PATTERN, text)
    reasons = re.findall(_REASON_PATTERN, text)
    times = re.findall(_TIME_PATTERN, text)
    k = max(len(comps), len(reasons), len(times))
    return int(k) if k > 0 else 1


def canonicalize_component(component: str) -> str:
    c = str(component or "").strip()
    m = re.match(r"^node-\\d+\\.(.+)$", c)
    if m:
        return m.group(1)
    return c


def canonicalize_prediction(prediction: str) -> str:
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
            c1 = canonicalize_component(str(c0))
            if c1 != str(c0):
                v["root cause component"] = c1
                changed = True
    if not changed:
        return prediction
    return json.dumps(obj, ensure_ascii=False)


def extract_reasons_from_scoring_points(scoring_points: str) -> List[str]:
    return extract_reason_candidates(scoring_points)


def estimate_k_occurrence_datetimes_from_trace(dataset: str, instruction: str, k: int) -> List[str]:
    date_dir, start, end = parse_time_window_generic(instruction)
    trace_path = os.path.join("dataset", dataset, "telemetry", date_dir, "trace", "trace_span.csv")
    if not os.path.exists(trace_path) or k <= 1:
        return [start.floor("min").strftime("%Y-%m-%d %H:%M:%S")]
    columns = set(_read_csv_with_python_fallback(trace_path, nrows=0).columns)
    ts_col = "startTime" if "startTime" in columns else ("timestamp" if "timestamp" in columns else None)
    dur_col = "elapsedTime" if "elapsedTime" in columns else ("duration" if "duration" in columns else None)
    if ts_col is None or dur_col is None:
        return [start.floor("min").strftime("%Y-%m-%d %H:%M:%S")]
    df = _read_csv_with_python_fallback(trace_path, low_memory=False, usecols=[ts_col, dur_col])
    if df.empty:
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


def _log_anomaly_mask_service(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.lower()
    return s.str.contains("error", regex=False) | s.str.contains("fail", regex=False) | s.str.contains("exception", regex=False) | s.str.contains("timeout", regex=False)

def _log_anomaly_mask_proxy(values: pd.Series) -> pd.Series:
    s = values.astype(str)
    return s.str.contains(" 500 ", regex=False) | s.str.contains(" 502 ", regex=False) | s.str.contains(" 503 ", regex=False) | s.str.contains(" 504 ", regex=False) | s.str.contains(" 400 ", regex=False) | s.str.contains(" 404 ", regex=False)


def estimate_k_occurrence_datetimes_from_logs(dataset: str, instruction: str, k: int) -> List[str]:
    date_dir, start, end = parse_time_window_generic(instruction)
    log_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "log")
    paths = [os.path.join(log_dir, "log_service.csv"), os.path.join(log_dir, "log_proxy.csv")]
    frames = []

    def _load_and_filter_time(p: str) -> pd.DataFrame:
        df = _read_csv_with_python_fallback(p, low_memory=False, usecols=["timestamp", "value"] if "log_proxy" in p or "log_service" in p else None)
        if df.empty or not {"timestamp", "value"}.issubset(df.columns):
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
        return df[(df["ts"] >= start) & (df["ts"] <= end)].copy()

    if os.path.exists(paths[0]):
        df = _load_and_filter_time(paths[0])
        if not df.empty:
            df["is_anom"] = _log_anomaly_mask_service(df["value"])
            frames.append(df[df["is_anom"]])
    if os.path.exists(paths[1]):
        df = _load_and_filter_time(paths[1])
        if not df.empty:
            df["is_anom"] = _log_anomaly_mask_proxy(df["value"])
            frames.append(df[df["is_anom"]])
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return []
    df["minute"] = df["ts"].dt.floor("min")
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
    
    # 提前做时间过滤，避免在几 GB 的全量日志上做 contains 字符串匹配
    if center_dt:
        try:
            c = pd.Timestamp(center_dt, tz="Asia/Shanghai")
            filter_start = c - pd.Timedelta(seconds=int(window_s))
            filter_end = c + pd.Timedelta(seconds=int(window_s))
        except Exception:
            filter_start, filter_end = start, end
    else:
        filter_start, filter_end = start, end
        
    def _load_and_filter_time(p: str) -> pd.DataFrame:
        df = _read_csv_with_python_fallback(p, low_memory=False, usecols=["timestamp", "cmdb_id", "value"] if "log_proxy" in p or "log_service" in p else None)
        if df.empty or not {"timestamp", "cmdb_id", "value"}.issubset(df.columns):
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
        return df[(df["ts"] >= filter_start) & (df["ts"] <= filter_end)].copy()

    if os.path.exists(paths[0]):
        df = _load_and_filter_time(paths[0])
        if not df.empty:
            df["is_anom"] = _log_anomaly_mask_service(df["value"])
            df["w"] = 1.0
            frames.append(df[df["is_anom"]])
            
    if os.path.exists(paths[1]):
        df = _load_and_filter_time(paths[1])
        if not df.empty:
            df["is_anom"] = _log_anomaly_mask_proxy(df["value"])
            s_val = df["value"].astype(str)
            is_5xx = s_val.str.contains(" 500 ", regex=False) | s_val.str.contains(" 502 ", regex=False) | s_val.str.contains(" 503 ", regex=False) | s_val.str.contains(" 504 ", regex=False)
            df["w"] = 1.0
            df.loc[is_5xx, "w"] = 2.0
            frames.append(df[df["is_anom"]])
            
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return []
    g = df.groupby("cmdb_id")["w"].sum().sort_values(ascending=False)
    return [str(x) for x in g.index.tolist()[:20]]
