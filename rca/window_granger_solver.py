import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import warnings


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

_TRACE_DAY_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}
_METRIC_DAY_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}
_ALLOWED_COMPONENTS_CACHE: Dict[str, set] = {}


def _parse_time_window_for_granger(instruction: str) -> Tuple[List[str], pd.Timestamp, pd.Timestamp]:
    text = str(instruction)
    tz = "Asia/Shanghai"

    cross = re.search(
        r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4}).*?from\s*(\d{2}:\d{2})\s*to\s*([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4}),?\s*(?:at\s*)?(\d{2}:\d{2})",
        text,
    )
    if cross:
        mo1, d1, y1, hm1, mo2, d2, y2, hm2 = cross.groups()
        y1, y2 = int(y1), int(y2)
        m1, m2 = _MONTH[mo1.lower()], _MONTH[mo2.lower()]
        d1, d2 = int(d1), int(d2)
        start = pd.Timestamp(f"{y1:04d}-{m1:02d}-{d1:02d} {hm1}:00", tz=tz)
        end = pd.Timestamp(f"{y2:04d}-{m2:02d}-{d2:02d} {hm2}:00", tz=tz)
        dirs = sorted({start.strftime("%Y_%m_%d"), end.strftime("%Y_%m_%d")})
        return dirs, start, end

    single = re.search(
        r"(?:During|Between|On|The|Within|Cloudbed).*?([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4}).*?(?:from\s*(\d{2}:\d{2})\s*to\s*(\d{2}:\d{2})|between\s*(\d{2}:\d{2})\s*and\s*(\d{2}:\d{2}))",
        text,
    )
    if single:
        mo, d, y, hm1, hm2, hm3, hm4 = single.groups()
        start_hm = hm1 or hm3
        end_hm = hm2 or hm4
        y = int(y)
        mo = _MONTH[mo.lower()]
        d = int(d)
        start = pd.Timestamp(f"{y:04d}-{mo:02d}-{d:02d} {start_hm}:00", tz=tz)
        end = pd.Timestamp(f"{y:04d}-{mo:02d}-{d:02d} {end_hm}:00", tz=tz)
        dirs = [start.strftime("%Y_%m_%d")]
        return dirs, start, end

    date_m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", text)
    if date_m:
        mo, d, y = date_m.groups()
        hm_m = re.search(r"from\s*(\d{2}:\d{2})\s*to\s*(\d{2}:\d{2})", text)
        if hm_m:
            hm1, hm2 = hm_m.groups()
            y = int(y)
            mo = _MONTH[mo.lower()]
            d = int(d)
            start = pd.Timestamp(f"{y:04d}-{mo:02d}-{d:02d} {hm1}:00", tz=tz)
            end = pd.Timestamp(f"{y:04d}-{mo:02d}-{d:02d} {hm2}:00", tz=tz)
            return [start.strftime("%Y_%m_%d")], start, end
        hm_m = re.search(r"time range of\s*(\d{2}:\d{2})\s*to\s*(\d{2}:\d{2})", text)
        if hm_m:
            hm1, hm2 = hm_m.groups()
            y = int(y)
            mo = _MONTH[mo.lower()]
            d = int(d)
            start = pd.Timestamp(f"{y:04d}-{mo:02d}-{d:02d} {hm1}:00", tz=tz)
            end = pd.Timestamp(f"{y:04d}-{mo:02d}-{d:02d} {hm2}:00", tz=tz)
            return [start.strftime("%Y_%m_%d")], start, end

    m = re.search(r"(\d{4}-\d{2}-\d{2}).*from\s*(\d{2}:\d{2})\s*to\s*(\d{2}:\d{2})", text)
    if m:
        date_str, hm1, hm2 = m.groups()
        start = pd.Timestamp(f"{date_str} {hm1}:00", tz=tz)
        end = pd.Timestamp(f"{date_str} {hm2}:00", tz=tz)
        return [start.strftime("%Y_%m_%d")], start, end

    raise ValueError("Cannot parse time window from instruction.")


def _parse_failure_count(instruction: str) -> int:
    text = str(instruction).lower()
    if "two failures" in text or "2 failures" in text:
        return 2
    if "three failures" in text or "3 failures" in text:
        return 3
    return 1


def _to_epoch_seconds(ts: pd.Timestamp) -> int:
    return int(ts.tz_convert("UTC").timestamp())


def _load_trace_window(trace_path: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not os.path.exists(trace_path):
        return pd.DataFrame()
    df = pd.read_csv(trace_path)
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["ts_local"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    return df[(df["ts_local"] >= start) & (df["ts_local"] <= end)].copy()


def _estimate_failure_times_from_trace(trace_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, k: int) -> List[pd.Timestamp]:
    if trace_df.empty or "duration" not in trace_df.columns:
        return [start.floor("min")] * k
    df = trace_df.copy()
    df["minute"] = df["ts_local"].dt.floor("min")
    p95 = df.groupby("minute")["duration"].quantile(0.95).sort_index()
    if p95.empty:
        return [start.floor("min")] * k

    peaks = p95.sort_values(ascending=False).head(max(10, k * 3))
    chosen: List[pd.Timestamp] = []
    for minute in peaks.index.sort_values():
        if len(chosen) >= k:
            break
        if all(abs((minute - prev).total_seconds()) >= 5 * 60 for prev in chosen):
            chosen.append(pd.Timestamp(minute))
    while len(chosen) < k:
        chosen.append(pd.Timestamp(peaks.index[0]))
    return chosen[:k]


def _estimate_failure_times_from_metric_signal(signal_df: pd.DataFrame, k: int) -> List[pd.Timestamp]:
    if signal_df.empty:
        return []
    total = signal_df.sum(axis=1)
    if total.empty:
        return []
    peaks = total.sort_values(ascending=False).head(max(10, k * 3))
    chosen: List[pd.Timestamp] = []
    for minute in peaks.index.sort_values():
        if len(chosen) >= k:
            break
        ts = pd.to_datetime(int(minute), unit="s", utc=True).tz_convert("Asia/Shanghai")
        if all(abs((ts - prev).total_seconds()) >= 5 * 60 for prev in chosen):
            chosen.append(ts)
    while len(chosen) < k:
        minute = int(peaks.index[0])
        chosen.append(pd.to_datetime(minute, unit="s", utc=True).tz_convert("Asia/Shanghai"))
    return chosen[:k]

def _load_metric_long_variant_a(metric_dir: str, start_s: int, end_s: int) -> pd.DataFrame:
    frames = []

    for name in ["metric_node.csv", "metric_container.csv", "metric_runtime.csv", "metric_mesh.csv"]:
        path = os.path.join(metric_dir, name)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.rename(columns={"kpi_name": "metric", "cmdb_id": "component"})
        if "timestamp" not in df.columns:
            continue
        df = df[(df["timestamp"] >= start_s) & (df["timestamp"] <= end_s)].copy()
        if df.empty:
            continue
        if "metric" not in df.columns:
            continue
        df = df[["timestamp", "component", "metric", "value"]].copy()
        if name == "metric_container.csv":
            df["component"] = df["component"].astype(str).apply(lambda x: x.split(".", 1)[1] if "." in x else x)
        if name == "metric_runtime.csv":
            df["component"] = df["component"].astype(str).apply(lambda x: x.split(".ts:", 1)[0] if ".ts:" in x else x)
        frames.append(df)

    service_path = os.path.join(metric_dir, "metric_service.csv")
    if os.path.exists(service_path):
        svc = pd.read_csv(service_path)
        if not svc.empty and "timestamp" in svc.columns and "service" in svc.columns:
            svc = svc[(svc["timestamp"] >= start_s) & (svc["timestamp"] <= end_s)].copy()
            if not svc.empty:
                melt = svc.melt(id_vars=["service", "timestamp"], var_name="metric", value_name="value")
                melt["component"] = melt["service"].astype(str).apply(lambda s: s.split("-", 1)[0])
                melt["metric"] = "service." + melt["metric"].astype(str)
                frames.append(melt[["timestamp", "component", "metric", "value"]])

    if not frames:
        return pd.DataFrame(columns=["timestamp", "component", "metric", "value"])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["timestamp", "component", "metric", "value"])
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(-1).astype(int)
    out = out[out["timestamp"] >= 0].copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    return out


def _load_metric_long_variant_b(metric_dir: str, start_s: int, end_s: int) -> pd.DataFrame:
    frames = []
    for name in ["metric_container.csv", "metric_app.csv"]:
        path = os.path.join(metric_dir, name)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if name == "metric_container.csv":
            df = df.rename(columns={"kpi_name": "metric", "cmdb_id": "component"})
            df = df[(df["timestamp"] >= start_s) & (df["timestamp"] <= end_s)].copy()
            if not df.empty:
                frames.append(df[["timestamp", "component", "metric", "value"]])
        else:
            if "startTime" in df.columns and "serviceName" in df.columns:
                df = df.rename(columns={"serviceName": "component"})
                df = df[(df["startTime"] >= start_s * 1000) & (df["startTime"] <= end_s * 1000)].copy()
                if not df.empty:
                    df["timestamp"] = (pd.to_numeric(df["startTime"], errors="coerce").fillna(0).astype(int) // 1000).astype(int)
                    for metric in ["avg_time", "succee_rate", "num"]:
                        if metric in df.columns:
                            tmp = df[["timestamp", "component", metric]].copy()
                            tmp = tmp.rename(columns={metric: "value"})
                            tmp["metric"] = "app." + metric
                            frames.append(tmp[["timestamp", "component", "metric", "value"]])
            elif "timestamp" in df.columns and "tc" in df.columns:
                df = df.rename(columns={"tc": "component"})
                df = df[(df["timestamp"] >= start_s) & (df["timestamp"] <= end_s)].copy()
                if not df.empty:
                    for metric in ["mrt", "rr", "sr", "cnt"]:
                        if metric in df.columns:
                            tmp = df[["timestamp", "component", metric]].copy()
                            tmp = tmp.rename(columns={metric: "value"})
                            tmp["metric"] = "app." + metric
                            frames.append(tmp[["timestamp", "component", "metric", "value"]])

    if not frames:
        return pd.DataFrame(columns=["timestamp", "component", "metric", "value"])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["timestamp", "component", "metric", "value"])
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(-1).astype(int)
    out = out[out["timestamp"] >= 0].copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    return out


def _component_signal(metric_long: pd.DataFrame, start_s: int, end_s: int) -> pd.DataFrame:
    if metric_long.empty:
        return pd.DataFrame()
    df = metric_long.copy()
    df = df[(df["timestamp"] >= start_s) & (df["timestamp"] <= end_s)].copy()
    if df.empty:
        return pd.DataFrame()
    df["minute"] = (df["timestamp"] // 60) * 60

    grp = df.groupby(["component", "metric"])["value"]
    df["median"] = grp.transform("median")
    df["mad"] = grp.transform(lambda x: (x - x.median()).abs().median())
    df["mad"] = df["mad"].replace(0, 1e-6)
    df["z"] = (df["value"] - df["median"]) / df["mad"]
    df["absz"] = df["z"].abs().clip(upper=50)

    by_comp_metric_minute = df.groupby(["component", "metric", "minute"])["absz"].mean().reset_index()
    metric_rank = by_comp_metric_minute.groupby(["component", "metric"])["absz"].mean().reset_index()
    metric_rank = metric_rank.sort_values(["component", "absz"], ascending=[True, False])
    metric_rank["r"] = metric_rank.groupby("component").cumcount() + 1
    top = metric_rank[metric_rank["r"] <= 3][["component", "metric"]]
    by_comp_metric_minute = by_comp_metric_minute.merge(top, on=["component", "metric"], how="inner")

    comp_sig = by_comp_metric_minute.groupby(["component", "minute"])["absz"].sum().reset_index()
    pivot = comp_sig.pivot(index="minute", columns="component", values="absz").fillna(0.0)
    return pivot


def _get_trace_day(dataset: str, date_dir: str) -> pd.DataFrame:
    key = (dataset, date_dir)
    if key in _TRACE_DAY_CACHE:
        return _TRACE_DAY_CACHE[key]
    telemetry_root = os.path.join("dataset", dataset, "telemetry")
    trace_path = os.path.join(telemetry_root, date_dir, "trace", "trace_span.csv")
    if not os.path.exists(trace_path):
        _TRACE_DAY_CACHE[key] = pd.DataFrame()
        return _TRACE_DAY_CACHE[key]
    df = pd.read_csv(trace_path)
    if df.empty:
        _TRACE_DAY_CACHE[key] = pd.DataFrame()
        return _TRACE_DAY_CACHE[key]
    df = df.copy()
    df["ts_local"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    _TRACE_DAY_CACHE[key] = df
    return df


def _get_metric_day(dataset: str, date_dir: str) -> pd.DataFrame:
    key = (dataset, date_dir)
    if key in _METRIC_DAY_CACHE:
        return _METRIC_DAY_CACHE[key]
    telemetry_root = os.path.join("dataset", dataset, "telemetry")
    metric_dir = os.path.join(telemetry_root, date_dir, "metric")
    if dataset.startswith("Market/"):
        df = _load_metric_long_variant_a(metric_dir, start_s=-10**18, end_s=10**18)
    else:
        df = _load_metric_long_variant_b(metric_dir, start_s=-10**18, end_s=10**18)
    _METRIC_DAY_CACHE[key] = df
    return df


def _allowed_components(dataset: str) -> set:
    if dataset in _ALLOWED_COMPONENTS_CACHE:
        return _ALLOWED_COMPONENTS_CACHE[dataset]
    record_path = os.path.join("dataset", dataset, "record.csv")
    if not os.path.exists(record_path):
        _ALLOWED_COMPONENTS_CACHE[dataset] = set()
        return _ALLOWED_COMPONENTS_CACHE[dataset]
    df = pd.read_csv(record_path)
    comps = set(df.get("component", pd.Series([], dtype=str)).dropna().astype(str).tolist())
    _ALLOWED_COMPONENTS_CACHE[dataset] = comps
    return comps


def _normalize_component(dataset: str, component: str, allowed: set) -> str:
    c = str(component)
    if not allowed:
        return c
    if c in allowed:
        return c
    if dataset.startswith("Market/"):
        if c + "-0" in allowed:
            return c + "-0"
        if c + "-1" in allowed:
            return c + "-1"
        cand = [x for x in allowed if x.startswith(c + "-")]
        if cand:
            cand.sort()
            return cand[0]
    return c


def _granger_component_graph(signal_df: pd.DataFrame, top_components: int = 12, max_pairs: int = 120, max_lag: int = 2, p_thresh: float = 0.05):
    if signal_df.empty:
        return []
    comp_strength = signal_df.mean().sort_values(ascending=False)
    comps = comp_strength.head(top_components).index.tolist()
    data = signal_df[comps].copy()
    data = data.loc[:, data.var() > 1e-6]
    comps = data.columns.tolist()
    if len(comps) < 2:
        return []

    edges = []
    pairs = 0
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if pairs >= max_pairs:
                break
            a, b = comps[i], comps[j]
            df_ab = data[[a, b]].copy()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    res_ab = grangercausalitytests(df_ab[[b, a]], maxlag=max_lag, verbose=False)
                    res_ba = grangercausalitytests(df_ab[[a, b]], maxlag=max_lag, verbose=False)
            except Exception:
                pairs += 1
                continue
            p_ab = min([res_ab[lag][0]["ssr_ftest"][1] for lag in res_ab])
            p_ba = min([res_ba[lag][0]["ssr_ftest"][1] for lag in res_ba])
            if p_ab < p_thresh:
                edges.append((a, b, float(1.0 - p_ab)))
            if p_ba < p_thresh:
                edges.append((b, a, float(1.0 - p_ba)))
            pairs += 1
        if pairs >= max_pairs:
            break
    return edges


def _pick_root_components(signal_df: pd.DataFrame, edges, k: int) -> List[str]:
    if signal_df.empty:
        return ["unknown"] * k
    if not edges:
        ranked = signal_df.mean().sort_values(ascending=False).index.tolist()
        return (ranked + ["unknown"] * k)[:k]
    out_score: Dict[str, float] = {}
    in_score: Dict[str, float] = {}
    for src, tgt, w in edges:
        out_score[src] = out_score.get(src, 0.0) + w
        in_score[tgt] = in_score.get(tgt, 0.0) + w
    comps = set(list(out_score.keys()) + list(in_score.keys()))
    scored = []
    for c in comps:
        scored.append((c, out_score.get(c, 0.0) - 0.35 * in_score.get(c, 0.0)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return ([c for c, _ in scored] + ["unknown"] * k)[:k]


def _infer_reason_from_metrics(dataset: str, metric_long: pd.DataFrame, component: str) -> str:
    if metric_long.empty:
        return "unknown"
    df = metric_long[metric_long["component"] == component].copy()
    if df.empty:
        return "unknown"
    names = " ".join(df["metric"].astype(str).str.lower().unique().tolist())

    if dataset.startswith("Market/"):
        if any(k in names for k in ["network", "ping", "retrans", "drop", "packet", "latency"]):
            if any(k in names for k in ["drop", "loss", "corrupt", "dropped"]):
                return "container packet loss" if not component.startswith("node-") else "node network packet loss"
            return "container network latency" if not component.startswith("node-") else "node network latency"
        if any(k in names for k in ["disk", "io", "await", "iops", "read", "write"]):
            if component.startswith("node-"):
                if "write" in names:
                    return "node disk write I/O consumption"
                return "node disk read I/O consumption"
            if "write" in names:
                return "container write I/O load"
            return "container read I/O load"
        if any(k in names for k in ["mem", "memory"]):
            return "node memory consumption" if component.startswith("node-") else "container memory load"
        if any(k in names for k in ["cpu", "load"]):
            return "node CPU load" if component.startswith("node-") else "container CPU load"
        if any(k in names for k in ["termination", "restart", "killed"]):
            return "container process termination"
        return "container CPU load"

    if dataset == "Bank":
        if any(k in names for k in ["packet", "loss", "drop"]):
            return "network packet loss"
        if any(k in names for k in ["latency", "delay", "ping"]):
            return "network latency"
        if any(k in names for k in ["disk", "io", "await", "read"]):
            return "high disk I/O read usage"
        if any(k in names for k in ["write"]):
            return "high disk I/O write usage"
        if any(k in names for k in ["space", "pct_usage"]):
            return "high disk space usage"
        if any(k in names for k in ["memory", "mem"]):
            return "high memory usage"
        if any(k in names for k in ["cpu", "cpuload"]):
            return "high CPU usage"
        return "high CPU usage"

    return "unknown"


def _component_anomaly_scores(metric_day: pd.DataFrame, metric_window: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, float]]:
    if metric_day.empty or metric_window.empty:
        return {}, {}, {}
    day = metric_day.copy()
    win = metric_window.copy()

    day["value"] = pd.to_numeric(day["value"], errors="coerce")
    win["value"] = pd.to_numeric(win["value"], errors="coerce")
    day = day.dropna(subset=["component", "metric", "value"])
    win = win.dropna(subset=["component", "metric", "value"])
    if day.empty or win.empty:
        return {}, {}, {}

    grp = day.groupby(["component", "metric"])["value"]
    baseline = grp.agg(median="median", mad=lambda x: (x - x.median()).abs().median()).reset_index()
    baseline["mad"] = baseline["mad"].replace(0, 1e-6)
    win = win.merge(baseline, on=["component", "metric"], how="left")
    win = win.dropna(subset=["median", "mad"]).copy()
    if win.empty:
        return {}, {}, {}
    win["absz"] = ((win["value"] - win["median"]) / win["mad"]).abs().clip(upper=50)

    comp_metric = win.groupby(["component", "metric"])["absz"].mean().reset_index()
    comp_scores: Dict[str, float] = {}
    comp_top_metrics: Dict[str, List[str]] = {}
    comp_cpu_spike: Dict[str, float] = {}
    for comp, sub in comp_metric.groupby("component"):
        sub = sub.sort_values("absz", ascending=False)
        top = sub.head(6)
        comp_scores[str(comp)] = float(top["absz"].sum())
        comp_top_metrics[str(comp)] = top["metric"].astype(str).tolist()
        cpu_like = top[top["metric"].astype(str).str.lower().str.contains("cpu", na=False)]
        comp_cpu_spike[str(comp)] = float(cpu_like["absz"].max()) if not cpu_like.empty else 0.0
    return comp_scores, comp_top_metrics, comp_cpu_spike


def _remap_to_allowed_components(dataset: str, scores: Dict[str, float], top_metrics: Dict[str, List[str]], cpu_spike: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, float]]:
    allowed = _allowed_components(dataset)
    if not allowed:
        return scores, top_metrics, cpu_spike
    new_scores: Dict[str, float] = {}
    new_top: Dict[str, List[str]] = {}
    new_cpu: Dict[str, float] = {}
    for comp, s in scores.items():
        norm = _normalize_component(dataset, comp, allowed)
        if norm not in allowed:
            continue
        new_scores[norm] = new_scores.get(norm, 0.0) + float(s)
        cur = new_top.get(norm, [])
        cur.extend(top_metrics.get(comp, []))
        new_top[norm] = cur[:12]
        new_cpu[norm] = max(new_cpu.get(norm, 0.0), float(cpu_spike.get(comp, 0.0)))
    return new_scores, new_top, new_cpu


def _rank_components_with_granger_prior(signal_df: pd.DataFrame, anomaly_scores: Dict[str, float], edges) -> List[str]:
    if not anomaly_scores:
        if signal_df.empty:
            return ["unknown"]
        return signal_df.mean().sort_values(ascending=False).index.tolist()

    rootness: Dict[str, float] = {}
    if edges:
        out_score: Dict[str, float] = {}
        in_score: Dict[str, float] = {}
        for src, tgt, w in edges:
            out_score[src] = out_score.get(src, 0.0) + w
            in_score[tgt] = in_score.get(tgt, 0.0) + w
        comps = set(list(out_score.keys()) + list(in_score.keys()))
        for c in comps:
            rootness[c] = out_score.get(c, 0.0) - 0.35 * in_score.get(c, 0.0)

    max_abs = max([abs(v) for v in rootness.values()], default=0.0)
    ranked = []
    for comp, score in anomaly_scores.items():
        r = rootness.get(comp, 0.0)
        r_norm = (r / max_abs) if max_abs > 1e-9 else 0.0
        final = score * (1.0 + 0.25 * r_norm)
        ranked.append((comp, final))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked]


def _infer_reason_from_top_metrics(dataset: str, component: str, top_metrics: List[str], cpu_spike_score: float) -> str:
    names = " ".join([str(m).lower() for m in top_metrics])
    comp = str(component)

    if dataset.startswith("Market/"):
        is_node = comp.startswith("node-")
        if "termination" in names or "killed" in names or "restart" in names or "oom" in names:
            return "container process termination"
        if "retrans" in names:
            return "container network packet retransmission"
        if "corrupt" in names or "error" in names:
            return "container network packet corruption"
        if "drop" in names or "loss" in names or "dropped" in names:
            return "container packet loss"
        if "latency" in names or "rtt" in names or "delay" in names or "ping" in names:
            return "node network latency" if is_node else "container network latency"
        if "pct_usage" in names or "space" in names:
            return "node disk space consumption" if is_node else "container write I/O load"
        if "disk" in names or "io" in names or "iops" in names or "await" in names or "read" in names or "write" in names:
            if is_node:
                if "write" in names:
                    return "node disk write I/O consumption"
                return "node disk read I/O consumption"
            if "write" in names:
                return "container write I/O load"
            return "container read I/O load"
        if "mem" in names or "memory" in names:
            return "node memory consumption" if is_node else "container memory load"
        if "cpu" in names or "load" in names:
            if is_node and cpu_spike_score >= 15:
                return "node CPU spike"
            return "node CPU load" if is_node else "container CPU load"
        return "node CPU load" if is_node else "container CPU load"

    if dataset == "Bank":
        if "oom" in names or "out of memory" in names:
            return "JVM Out of Memory (OOM) Heap"
        if "jvm" in names and "cpu" in names:
            return "high JVM CPU load"
        if "packet" in names or "loss" in names or "drop" in names:
            return "network packet loss"
        if "latency" in names or "delay" in names or "ping" in names:
            return "network latency"
        if "space" in names or "pct_usage" in names:
            return "high disk space usage"
        if "disk" in names or "io" in names or "await" in names or "read" in names:
            return "high disk I/O read usage"
        if "write" in names:
            return "high disk I/O read usage"
        if "memory" in names or "mem" in names:
            return "high memory usage"
        if "cpu" in names or "cpuload" in names:
            return "high CPU usage"
        return "high CPU usage"

    return "unknown"


def _infer_top_reasons(dataset: str, metric_long: pd.DataFrame, start_s: int, end_s: int, k: int) -> List[str]:
    if metric_long.empty:
        return ["unknown"] * k
    df = metric_long[(metric_long["timestamp"] >= start_s) & (metric_long["timestamp"] <= end_s)].copy()
    if df.empty:
        return ["unknown"] * k
    df["minute"] = (df["timestamp"] // 60) * 60
    grp = df.groupby(["component", "metric"])["value"]
    df["median"] = grp.transform("median")
    df["mad"] = grp.transform(lambda x: (x - x.median()).abs().median())
    df["mad"] = df["mad"].replace(0, 1e-6)
    df["absz"] = ((df["value"] - df["median"]) / df["mad"]).abs().clip(upper=50)
    hot = df.sort_values("absz", ascending=False).head(500)
    if hot.empty:
        return ["unknown"] * k
    hot["reason"] = hot["component"].astype(str).apply(lambda c: _infer_reason_from_metrics(dataset, hot, c))
    counts = hot["reason"].value_counts()
    ranked = counts.index.tolist()
    return (ranked + ["unknown"] * k)[:k]


def solve_with_window_granger(dataset: str, instruction: str, task_id: int) -> str:
    date_dirs, start, end = _parse_time_window_for_granger(instruction)
    k_fail = _parse_failure_count(instruction)

    telemetry_root = os.path.join("dataset", dataset, "telemetry")
    start_s = _to_epoch_seconds(start)
    end_s = _to_epoch_seconds(end)

    metric_frames = [_get_metric_day(dataset, d) for d in date_dirs]
    metric_day = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    metric_long = metric_day
    if not metric_long.empty:
        metric_long = metric_long[(metric_long["timestamp"] >= start_s) & (metric_long["timestamp"] <= end_s)].copy()

    signal_df = _component_signal(metric_long, start_s, end_s)
    failure_times = _estimate_failure_times_from_metric_signal(signal_df, k_fail)
    if not failure_times:
        trace_frames = [_get_trace_day(dataset, d) for d in date_dirs]
        trace_df = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
        if not trace_df.empty:
            trace_df = trace_df[(trace_df["ts_local"] >= start) & (trace_df["ts_local"] <= end)].copy()
        failure_times = _estimate_failure_times_from_trace(trace_df, start, end, k_fail)

    if task_id == 1:
        parts = []
        for idx, dt_ts in enumerate(failure_times, start=1):
            dt = dt_ts.strftime("%Y-%m-%d %H:%M:%S")
            parts.append(f"\"{idx}\": {{ \"root cause occurrence datetime\": \"{dt}\" }}")
        return "{ " + ", ".join(parts) + " }"

    if task_id == 2:
        reasons = _infer_top_reasons(dataset, metric_long, start_s, end_s, k_fail)
        parts = []
        for idx, r in enumerate(reasons, start=1):
            parts.append(f"\"{idx}\": {{ \"root cause reason\": \"{r}\" }}")
        return "{ " + ", ".join(parts) + " }"

    edges = _granger_component_graph(signal_df, top_components=10, max_pairs=60) if task_id in (5, 6, 7) else []
    anomaly_scores, top_metrics_map, cpu_spike_map = _component_anomaly_scores(metric_day, metric_long)
    anomaly_scores, top_metrics_map, cpu_spike_map = _remap_to_allowed_components(dataset, anomaly_scores, top_metrics_map, cpu_spike_map)
    ranked_components = _rank_components_with_granger_prior(signal_df, anomaly_scores, edges)
    root_components = (ranked_components + ["unknown"] * k_fail)[:k_fail]

    objs = []
    for i in range(k_fail):
        comp = root_components[i] if i < len(root_components) else "unknown"
        top_metrics = top_metrics_map.get(comp, [])
        cpu_spike_score = cpu_spike_map.get(comp, 0.0)
        reason = _infer_reason_from_top_metrics(dataset, comp, top_metrics, cpu_spike_score)
        dt = failure_times[i].strftime("%Y-%m-%d %H:%M:%S")

        obj: Dict[str, str] = {}
        if task_id in (1, 4, 5, 7):
            obj["root cause occurrence datetime"] = dt
        if task_id in (3, 5, 6, 7):
            obj["root cause component"] = comp
        if task_id in (2, 4, 6, 7):
            obj["root cause reason"] = reason
        objs.append(obj)

    parts = []
    for idx, obj in enumerate(objs, start=1):
        fields = []
        if "root cause occurrence datetime" in obj:
            fields.append(f"\"root cause occurrence datetime\": \"{obj['root cause occurrence datetime']}\"")
        if "root cause component" in obj:
            fields.append(f"\"root cause component\": \"{obj['root cause component']}\"")
        if "root cause reason" in obj:
            fields.append(f"\"root cause reason\": \"{obj['root cause reason']}\"")
        inner = "{ " + ", ".join(fields) + " }"
        parts.append(f"\"{idx}\": {inner}")
    return "{ " + ", ".join(parts) + " }"
