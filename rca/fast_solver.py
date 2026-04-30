import os
import re
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


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


def _default_component_pool() -> Tuple[List[str], List[str], List[str]]:
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
    nodes, pods, svcs = _default_component_pool()
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
    nodes, pods, svcs = _default_component_pool()
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


def solve_easy(instruction: str, required_fields: List[str]) -> str:
    fields = set([str(x).strip().lower() for x in (required_fields or []) if str(x).strip()])
    if fields == {"time"}:
        dt = estimate_time_from_trace(instruction)
        return "{" + "\"1\": {" + f"\"root cause occurrence datetime\": \"{dt}\"" + "}" + "}"
    if fields == {"reason"}:
        reason, _ = infer_reason_with_evidence(instruction)
        return "{" + "\"1\": {" + f"\"root cause reason\": \"{reason}\"" + "}" + "}"
    if fields == {"component"}:
        date_dir, start, end = _parse_time_window(instruction)
        metric_dir = os.path.join("dataset", "Telecom", "telemetry", date_dir, "metric")
        component = _best_component_for_component_field(metric_dir, start, end, instruction)
        return "{" + "\"1\": {" + f"\"root cause component\": \"{component}\"" + "}" + "}"

    obj: Dict[str, Dict[str, str]] = {"1": {}}
    if "time" in fields:
        obj["1"]["root cause occurrence datetime"] = estimate_time_from_trace(instruction)
    if "reason" in fields:
        reason, _ = infer_reason_with_evidence(instruction)
        obj["1"]["root cause reason"] = str(reason or "")
    if "component" in fields:
        date_dir, start, end = _parse_time_window(instruction)
        metric_dir = os.path.join("dataset", "Telecom", "telemetry", date_dir, "metric")
        obj["1"]["root cause component"] = _best_component_for_component_field(metric_dir, start, end, instruction)
    return json.dumps(obj, ensure_ascii=False)


def _breach_scores_for_keywords(metric_path: str, start: pd.Timestamp, end: pd.Timestamp, keywords: List[str]) -> Tuple[float, int]:
    if not os.path.exists(metric_path):
        return 0.0, 0
    df = _load_metric(metric_path)
    if df.empty:
        return 0.0, 0
    df = df.copy()
    df["name_l"] = df["name"].astype(str).str.lower()
    mask = False
    for kw in keywords:
        mask = mask | df["name_l"].str.contains(str(kw).lower(), na=False)
    df = df[mask].copy()
    if df.empty:
        return 0.0, 0
    thresholds = (
        df.groupby(["cmdb_id", "name"], as_index=False)["value"]
        .quantile(0.95)
        .rename(columns={"value": "p95"})
    )
    win = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if win.empty:
        return 0.0, 0
    win = win.merge(thresholds, on=["cmdb_id", "name"], how="left")
    win = win.dropna(subset=["p95"]).copy()
    denom = win["p95"].abs().where(win["p95"].abs() > 1e-6, 1.0)
    win["breach_ratio"] = (win["value"] - win["p95"]) / denom
    win["breach_ratio"] = win["breach_ratio"].clip(lower=0.0, upper=10.0)
    win = win[win["breach_ratio"] > 0].copy()
    if win.empty:
        return 0.0, 0
    return float(win["breach_ratio"].sum()), int(len(win))


def _norm_name(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

# reason socre
def _breach_stats(metric_path: str, start: pd.Timestamp, end: pd.Timestamp, any_keywords: List[str], strong_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    if not os.path.exists(metric_path):
        return {"sum_ratio": 0.0, "count": 0, "distinct": 0, "strong_hit": False, "strong_count": 0, "top_metrics": []}
    df = _load_metric(metric_path)
    if df.empty:
        return {"sum_ratio": 0.0, "count": 0, "distinct": 0, "strong_hit": False, "strong_count": 0, "top_metrics": []}
    df = df.copy()
    df["name_norm"] = df["name"].astype(str).map(_norm_name)
    any_norm = [_norm_name(k) for k in any_keywords if str(k).strip()]
    strong_norm = [_norm_name(k) for k in (strong_keywords or []) if str(k).strip()]

    mask = False
    for kw in any_norm:
        mask = mask | df["name_norm"].str.contains(kw, na=False)
    if not mask.any():
        return {"sum_ratio": 0.0, "count": 0, "distinct": 0, "strong_hit": False, "strong_count": 0, "top_metrics": []}

    thresholds = (
        df.groupby(["cmdb_id", "name"], as_index=False)["value"]
        .quantile(0.95)
        .rename(columns={"value": "p95"})
    )
    win = df[(df["timestamp"] >= start) & (df["timestamp"] <= end) & (mask)].copy()
    if win.empty:
        return {"sum_ratio": 0.0, "count": 0, "distinct": 0, "strong_hit": False, "strong_count": 0, "top_metrics": []}
    win = win.merge(thresholds, on=["cmdb_id", "name"], how="left").dropna(subset=["p95"]).copy()
    denom = win["p95"].abs().where(win["p95"].abs() > 1e-6, 1.0)
    win["breach_ratio"] = (win["value"] - win["p95"]) / denom
    win["breach_ratio"] = win["breach_ratio"].clip(lower=0.0, upper=10.0)
    win = win[win["breach_ratio"] > 0].copy()
    if win.empty:
        return {"sum_ratio": 0.0, "count": 0, "distinct": 0, "strong_hit": False, "strong_count": 0, "top_metrics": []}
    strong_hit = False
    strong_count = 0
    if strong_norm:
        for k in strong_norm:
            if win["name_norm"].str.contains(k, na=False).any():
                strong_count += 1
        strong_hit = strong_count > 0
    top_metrics = (
        win.groupby("name", as_index=False)["breach_ratio"]
        .sum()
        .sort_values("breach_ratio", ascending=False)
        .head(10)["name"]
        .astype(str)
        .tolist()
    )
    return {
        "sum_ratio": float(win["breach_ratio"].sum()),
        "count": int(len(win)),
        "distinct": int(win["name"].nunique()),
        "strong_hit": bool(strong_hit),
        "strong_count": int(strong_count),
        "top_metrics": top_metrics,
    }


def _reason_from_breach_scores(metric_dir: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    net_loss_kws = ["loss", "drop", "retrans", "timeout", "reset", "error"]
    net_delay_kws = ["tnsping_result_time", "tnsping", "ping", "rtt", "latency", "delay"]
    db_conn_kws = ["login_per_sec", "login", "tps_per_sec", "tps", "exec_per_sec", "exec", "sess_active", "session"]
    db_conn_strong = ["login_per_sec", "tps_per_sec", "exec_per_sec"]
    db_close_kws = ["on_off_state", "on_off", "logout", "disconnect", "hang", "shutdown", "restart", "close", "reset"]
    cpu_kws = ["proc_used_pct", "proc_user_used_pct", "cpu", "load", "iowait", "util"]

    net_path = os.path.join(metric_dir, "metric_node.csv")
    mw_path = os.path.join(metric_dir, "metric_middleware.csv")
    svc_path = os.path.join(metric_dir, "metric_service.csv")
    cont_path = os.path.join(metric_dir, "metric_container.csv")

    net_loss_n = _breach_stats(net_path, start, end, net_loss_kws)
    net_delay_n = _breach_stats(net_path, start, end, net_delay_kws)
    net_loss_m = _breach_stats(mw_path, start, end, net_loss_kws)
    net_delay_m = _breach_stats(mw_path, start, end, net_delay_kws)
    net_loss = max([net_loss_n, net_loss_m], key=lambda d: (float(d.get("sum_ratio", 0.0)), int(d.get("distinct", 0)), int(d.get("count", 0))))
    net_delay = max([net_delay_n, net_delay_m], key=lambda d: (float(d.get("sum_ratio", 0.0)), int(d.get("distinct", 0)), int(d.get("count", 0))))
    db_conn = _breach_stats(svc_path, start, end, db_conn_kws, strong_keywords=db_conn_strong)
    db_close = _breach_stats(svc_path, start, end, db_close_kws)
    cpu_c = _breach_stats(cont_path, start, end, cpu_kws)
    cpu_n = _breach_stats(net_path, start, end, cpu_kws)

    def score(stat: Dict[str, Any], require_distinct: int = 1, weak_penalty: float = 1.0) -> float:
        if int(stat.get("distinct", 0)) < require_distinct:
            return 0.0
        s_sum = float(stat.get("sum_ratio", 0.0))
        s_cnt = float(stat.get("count", 0))
        s_dis = float(stat.get("distinct", 0))
        mean_ratio = s_sum / max(1.0, s_cnt)
        s = 2.0 * math.log1p(max(0.0, mean_ratio)) + 0.35 * math.log1p(max(0.0, s_cnt)) + 0.9 * math.log1p(max(0.0, s_dis))
        return float(s * weak_penalty)

    s_net_loss = max(score(net_loss_n, require_distinct=1), score(net_loss_m, require_distinct=1))
    s_net_delay = max(score(net_delay_n, require_distinct=1), score(net_delay_m, require_distinct=1))
    s_db_conn = score(db_conn, require_distinct=2, weak_penalty=1.0 if db_conn.get("strong_hit") else 0.25)
    s_db_close = score(db_close, require_distinct=2)
    s_cpu = max(score(cpu_c, require_distinct=1), score(cpu_n, require_distinct=1))

    scored = [
        ("network loss", s_net_loss, net_loss),
        ("network delay", s_net_delay, net_delay),
        ("db connection limit", s_db_conn, db_conn),
        ("db close", s_db_close, db_close),
        ("CPU fault", s_cpu, {"container": cpu_c, "node": cpu_n}),
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    if scored and scored[0][1] >= 1.0:
        return scored[0][0]
    return _reason_from_window(metric_dir, start, end)


def infer_reason_with_evidence(instruction: str) -> Tuple[str, Dict[str, Any]]:
    date_dir, start, end = _parse_time_window(instruction)
    metric_dir = os.path.join("dataset", "Telecom", "telemetry", date_dir, "metric")
    trace_path = os.path.join("dataset", "Telecom", "telemetry", date_dir, "trace", "trace_span.csv")

    net_loss_kws = ["loss", "drop", "retrans", "timeout", "reset", "error"]
    net_delay_kws = ["tnsping_result_time", "tnsping", "ping", "rtt", "latency", "delay"]
    db_conn_kws = ["login_per_sec", "login", "tps_per_sec", "tps", "exec_per_sec", "exec", "sess_active", "session"]
    db_conn_strong = ["login_per_sec", "tps_per_sec", "exec_per_sec"]
    db_close_kws = ["on_off_state", "on_off", "logout", "disconnect", "hang", "shutdown", "restart", "close", "reset"]
    cpu_kws = ["proc_used_pct", "proc_user_used_pct", "cpu", "load", "iowait", "util"]

    net_path = os.path.join(metric_dir, "metric_node.csv")
    mw_path = os.path.join(metric_dir, "metric_middleware.csv")
    svc_path = os.path.join(metric_dir, "metric_service.csv")
    cont_path = os.path.join(metric_dir, "metric_container.csv")

    net_loss_n = _breach_stats(net_path, start, end, net_loss_kws)
    net_delay_n = _breach_stats(net_path, start, end, net_delay_kws)
    net_loss_m = _breach_stats(mw_path, start, end, net_loss_kws)
    net_delay_m = _breach_stats(mw_path, start, end, net_delay_kws)
    net_loss = max([net_loss_n, net_loss_m], key=lambda d: (float(d.get("sum_ratio", 0.0)), int(d.get("distinct", 0)), int(d.get("count", 0))))
    net_delay = max([net_delay_n, net_delay_m], key=lambda d: (float(d.get("sum_ratio", 0.0)), int(d.get("distinct", 0)), int(d.get("count", 0))))
    db_conn = _breach_stats(svc_path, start, end, db_conn_kws, strong_keywords=db_conn_strong)
    db_close = _breach_stats(svc_path, start, end, db_close_kws)
    cpu_c = _breach_stats(cont_path, start, end, cpu_kws)
    cpu_n = _breach_stats(net_path, start, end, cpu_kws)

    def score(stat: Dict[str, Any], require_distinct: int = 1, weak_penalty: float = 1.0) -> float:
        if int(stat.get("distinct", 0)) < require_distinct:
            return 0.0
        s_sum = float(stat.get("sum_ratio", 0.0))
        s_cnt = float(stat.get("count", 0))
        s_dis = float(stat.get("distinct", 0))
        mean_ratio = s_sum / max(1.0, s_cnt)
        s = 2.0 * math.log1p(max(0.0, mean_ratio)) + 0.35 * math.log1p(max(0.0, s_cnt)) + 0.9 * math.log1p(max(0.0, s_dis))
        return float(s * weak_penalty)

    s_net_loss = max(score(net_loss_n, require_distinct=1), score(net_loss_m, require_distinct=1))
    s_net_delay = max(score(net_delay_n, require_distinct=1), score(net_delay_m, require_distinct=1))
    s_db_conn = score(db_conn, require_distinct=2, weak_penalty=1.0 if db_conn.get("strong_hit") else 0.25)
    s_db_close = score(db_close, require_distinct=2)
    s_cpu = max(score(cpu_c, require_distinct=1), score(cpu_n, require_distinct=1))

    scored = [
        ("network loss", s_net_loss, net_loss),
        ("network delay", s_net_delay, net_delay),
        ("db connection limit", s_db_conn, db_conn),
        ("db close", s_db_close, db_close),
        ("CPU fault", s_cpu, {"container": cpu_c, "node": cpu_n}),
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_reason = scored[0][0] if scored else _reason_from_window(metric_dir, start, end)
    best_score = float(scored[0][1]) if scored else 0.0
    second_score = float(scored[1][1]) if len(scored) > 1 else 0.0
    gap = best_score - second_score
    evidence = {
        "time_window": {"date_dir": date_dir, "start": start.strftime("%Y-%m-%d %H:%M:%S"), "end": end.strftime("%Y-%m-%d %H:%M:%S")},
        "scores": {r: float(s) for r, s, _ in scored},
        "gap": float(gap),
        "best": {"reason": best_reason, "score": float(best_score)},
        "stats": {
            "network_loss": net_loss,
            "network_delay": net_delay,
            "db_connection_limit": db_conn,
            "db_close": db_close,
            "cpu_container": cpu_c,
            "cpu_node": cpu_n,
        },
        "top_metrics": {
            "network_loss": net_loss.get("top_metrics", []),
            "network_delay": net_delay.get("top_metrics", []),
            "db_connection_limit": db_conn.get("top_metrics", []),
            "db_close": db_close.get("top_metrics", []),
            "cpu_container": cpu_c.get("top_metrics", []),
            "cpu_node": cpu_n.get("top_metrics", []),
        },
        "keyword_hits": {
            "network_delay": 0,
            "network_loss": 0,
            "db_connection_limit": int(db_conn.get("strong_count", 0) or 0),
            "db_close": 0,
            "cpu": 0,
        },
        "trace_spread": {},
    }
    def _count_hits(metrics: List[str], tokens: List[str]) -> int:
        blob = " ".join([str(x).lower() for x in metrics])
        return int(sum(1 for t in tokens if t and t in blob))

    evidence["keyword_hits"]["network_delay"] = _count_hits(evidence["top_metrics"]["network_delay"], ["tnsping", "ping", "rtt", "latency", "delay"])
    evidence["keyword_hits"]["network_loss"] = _count_hits(evidence["top_metrics"]["network_loss"], ["timeout", "retrans", "loss", "drop", "reset", "error"])
    evidence["keyword_hits"]["db_close"] = _count_hits(evidence["top_metrics"]["db_close"], ["close", "disconnect", "logout", "onoff", "shutdown", "restart"])
    evidence["keyword_hits"]["cpu"] = _count_hits(evidence["top_metrics"]["cpu_node"] + evidence["top_metrics"]["cpu_container"], ["cpu", "load", "iowait", "util"])

    evidence["trace_spread"] = _trace_spread_stats(trace_path, start, end)
    evidence["trace_component"] = _trace_component_strength(trace_path, start, end)

    spread = evidence["trace_spread"]
    spread_ratio = float(spread.get("affected_ratio", 0.0) or 0.0)
    hits = evidence["keyword_hits"]
    scores = evidence["scores"]
    tc = evidence["trace_component"]
    db_max = float(tc.get("db_max_ratio", 0.0) or 0.0)
    node_max = float(tc.get("node_max_ratio", 0.0) or 0.0)
    cont_max = float(tc.get("container_max_ratio", 0.0) or 0.0)
    top_type = str(tc.get("top_type") or "")
    if spread_ratio >= 0.35 and (int(hits.get("network_loss", 0)) > 0 or int(hits.get("network_delay", 0)) > 0):
        if int(hits.get("network_loss", 0)) >= int(hits.get("network_delay", 0)):
            scores["network loss"] = float(scores.get("network loss", 0.0) + 1.0)
        else:
            scores["network delay"] = float(scores.get("network delay", 0.0) + 1.0)
        scores["CPU fault"] = float(max(0.0, scores.get("CPU fault", 0.0) - 0.4))
        scores["db connection limit"] = float(max(0.0, scores.get("db connection limit", 0.0) - 0.3))
        scores["db close"] = float(max(0.0, scores.get("db close", 0.0) - 0.3))
    if spread_ratio <= 0.15 and float(scores.get("CPU fault", 0.0)) > 0:
        scores["CPU fault"] = float(scores.get("CPU fault", 0.0) + 0.2)
    if db_max >= 1.6 or top_type == "db":
        scores["db connection limit"] = float(scores.get("db connection limit", 0.0) + 0.4)
        scores["db close"] = float(scores.get("db close", 0.0) + 0.4)
    if db_max <= 1.25 and top_type in ("node", "container"):
        scores["db connection limit"] = float(max(0.0, scores.get("db connection limit", 0.0) - 0.6))
        scores["db close"] = float(max(0.0, scores.get("db close", 0.0) - 0.6))
    if (max(node_max, cont_max) >= 1.6) and spread_ratio <= 0.25 and int(hits.get("cpu", 0)) >= 1:
        scores["CPU fault"] = float(scores.get("CPU fault", 0.0) + 0.3)
        scores["network loss"] = float(max(0.0, scores.get("network loss", 0.0) - 0.2))
        scores["network delay"] = float(max(0.0, scores.get("network delay", 0.0) - 0.2))
    evidence["scores"] = scores
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if ranked:
        best_reason = ranked[0][0]
        evidence["best"] = {"reason": best_reason, "score": float(ranked[0][1])}
        evidence["gap"] = float(ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0))
    return best_reason, evidence


def _trace_component_strength(trace_path: str, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    if not os.path.exists(trace_path):
        return {"top_component": "", "top_ratio": 0.0, "top_type": "", "db_max_ratio": 0.0, "node_max_ratio": 0.0, "container_max_ratio": 0.0}
    df = pd.read_csv(trace_path, low_memory=False)
    if df.empty or "cmdb_id" not in df.columns:
        return {"top_component": "", "top_ratio": 0.0, "top_type": "", "db_max_ratio": 0.0, "node_max_ratio": 0.0, "container_max_ratio": 0.0}
    ts_col = "startTime" if "startTime" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    dur_col = "elapsedTime" if "elapsedTime" in df.columns else ("duration" if "duration" in df.columns else None)
    if ts_col is None or dur_col is None:
        return {"top_component": "", "top_ratio": 0.0, "top_type": "", "db_max_ratio": 0.0, "node_max_ratio": 0.0, "container_max_ratio": 0.0}
    df = df[[ts_col, "cmdb_id", dur_col]].copy()
    df["ts"] = pd.to_datetime(df[ts_col], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    df["dur"] = pd.to_numeric(df[dur_col], errors="coerce")
    df["cmdb_id"] = df["cmdb_id"].astype(str)
    df = df.dropna(subset=["ts", "dur"])
    df = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()
    if df.empty:
        return {"top_component": "", "top_ratio": 0.0, "top_type": "", "db_max_ratio": 0.0, "node_max_ratio": 0.0, "container_max_ratio": 0.0}
    p95 = df.groupby("cmdb_id")["dur"].quantile(0.95)
    if p95.empty:
        return {"top_component": "", "top_ratio": 0.0, "top_type": "", "db_max_ratio": 0.0, "node_max_ratio": 0.0, "container_max_ratio": 0.0}
    baseline = float(p95.median())
    if baseline <= 0:
        return {"top_component": "", "top_ratio": 0.0, "top_type": "", "db_max_ratio": 0.0, "node_max_ratio": 0.0, "container_max_ratio": 0.0}
    ratios = (p95 / baseline).fillna(0.0)
    top_comp = str(ratios.sort_values(ascending=False).index[0])
    top_ratio = float(ratios.loc[top_comp])
    def _ctype(c: str) -> str:
        c = str(c or "")
        if c.startswith("db_"):
            return "db"
        if c.startswith(("os_", "node-")):
            return "node"
        if c.startswith(("docker_", "container")):
            return "container"
        if "-" in c:
            return "service"
        return "other"
    db_max = float(ratios[[i for i in ratios.index if str(i).startswith('db_')]].max()) if any(str(i).startswith('db_') for i in ratios.index) else 0.0
    node_max = float(ratios[[i for i in ratios.index if str(i).startswith('os_')]].max()) if any(str(i).startswith('os_') for i in ratios.index) else 0.0
    cont_max = float(ratios[[i for i in ratios.index if str(i).startswith('docker_')]].max()) if any(str(i).startswith('docker_') for i in ratios.index) else 0.0
    return {
        "top_component": top_comp,
        "top_ratio": top_ratio,
        "top_type": _ctype(top_comp),
        "db_max_ratio": db_max,
        "node_max_ratio": node_max,
        "container_max_ratio": cont_max,
    }


def _trace_spread_stats(trace_path: str, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    if not os.path.exists(trace_path):
        return {"total_components": 0, "affected_components": 0, "affected_ratio": 0.0}
    df = pd.read_csv(trace_path, low_memory=False)
    if df.empty:
        return {"total_components": 0, "affected_components": 0, "affected_ratio": 0.0}
    if "cmdb_id" not in df.columns:
        return {"total_components": 0, "affected_components": 0, "affected_ratio": 0.0}
    ts_col = "startTime" if "startTime" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    dur_col = "elapsedTime" if "elapsedTime" in df.columns else ("duration" if "duration" in df.columns else None)
    if ts_col is None or dur_col is None:
        return {"total_components": 0, "affected_components": 0, "affected_ratio": 0.0}
    df = df[[ts_col, "cmdb_id", dur_col]].copy()
    if ts_col == "startTime":
        df["ts"] = pd.to_datetime(df[ts_col], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    else:
        df["ts"] = pd.to_datetime(df[ts_col], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    df["dur"] = pd.to_numeric(df[dur_col], errors="coerce")
    df["cmdb_id"] = df["cmdb_id"].astype(str)
    df = df.dropna(subset=["ts", "dur"])
    df = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()
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


def evidence_supports_reason(evidence: Dict[str, Any], reason: str) -> bool:
    stats = evidence.get("stats", {}) if isinstance(evidence.get("stats"), dict) else {}
    topm = evidence.get("top_metrics", {}) if isinstance(evidence.get("top_metrics"), dict) else {}
    hits = evidence.get("keyword_hits", {}) if isinstance(evidence.get("keyword_hits"), dict) else {}
    r = str(reason)
    if r == "network loss":
        s = stats.get("network_loss", {})
        names = " ".join([str(x).lower() for x in topm.get("network_loss", [])])
        return bool(
            float(s.get("sum_ratio", 0.0)) >= 0.6
            and int(s.get("distinct", 0)) >= 1
            and (int(hits.get("network_loss", 0)) >= 1 or ("timeout" in names or "retrans" in names or "loss" in names or "drop" in names or "reset" in names))
        )
    if r == "network delay":
        s = stats.get("network_delay", {})
        names = " ".join([str(x).lower() for x in topm.get("network_delay", [])])
        return bool(
            float(s.get("sum_ratio", 0.0)) >= 0.6
            and int(s.get("distinct", 0)) >= 1
            and (int(hits.get("network_delay", 0)) >= 1 or ("tnsping" in names or "ping" in names or "rtt" in names or "latency" in names or "delay" in names))
        )
    if r == "db connection limit":
        s = stats.get("db_connection_limit", {})
        return bool(float(s.get("sum_ratio", 0.0)) >= 0.8 and int(s.get("distinct", 0)) >= 2 and bool(s.get("strong_hit", False)))
    if r == "db close":
        s = stats.get("db_close", {})
        names = " ".join([str(x).lower() for x in topm.get("db_close", [])])
        return bool(
            float(s.get("sum_ratio", 0.0)) >= 0.8
            and int(s.get("distinct", 0)) >= 2
            and (int(hits.get("db_close", 0)) >= 1 or ("close" in names or "disconnect" in names or "logout" in names or "onoff" in names))
        )
    if r == "CPU fault":
        s1 = stats.get("cpu_container", {})
        s2 = stats.get("cpu_node", {})
        return bool(max(float(s1.get("sum_ratio", 0.0)), float(s2.get("sum_ratio", 0.0))) >= 0.8)
    return False


def _best_component_by_total_breach(metric_dir: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    files = ["metric_node.csv", "metric_service.csv", "metric_container.csv", "metric_middleware.csv"]
    total: Dict[str, float] = {}
    for fn in files:
        path = os.path.join(metric_dir, fn)
        if not os.path.exists(path):
            continue
        df = _load_metric(path)
        if df.empty:
            continue
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
        if df.empty:
            continue
        thresholds = (
            df.groupby(["cmdb_id", "name"], as_index=False)["value"]
            .quantile(0.95)
            .rename(columns={"value": "p95"})
        )
        df = df.merge(thresholds, on=["cmdb_id", "name"], how="left")
        df = df.dropna(subset=["p95"]).copy()
        denom = df["p95"].abs().where(df["p95"].abs() > 1e-6, 1.0)
        df["breach_ratio"] = (df["value"] - df["p95"]) / denom
        df["breach_ratio"] = df["breach_ratio"].clip(lower=0.0, upper=10.0)
        df = df[df["breach_ratio"] > 0].copy()
        if df.empty:
            continue
        for comp, g in df.groupby("cmdb_id"):
            comp_s = str(comp)
            if not comp_s.startswith(("os_", "docker_", "db_")):
                continue
            total[comp_s] = total.get(comp_s, 0.0) + float(g["breach_ratio"].sum())
    if not total:
        return ""
    return sorted(total.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def _best_component_by_prefix(metric_dir: str, start: pd.Timestamp, end: pd.Timestamp, prefix: str) -> Tuple[str, float]:
    files = ["metric_node.csv", "metric_service.csv", "metric_container.csv", "metric_middleware.csv"]
    total: Dict[str, float] = {}
    for fn in files:
        path = os.path.join(metric_dir, fn)
        if not os.path.exists(path):
            continue
        df = _load_metric(path)
        if df.empty:
            continue
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
        if df.empty:
            continue
        df = df[df["cmdb_id"].astype(str).str.startswith(prefix)].copy()
        if df.empty:
            continue
        thresholds = (
            df.groupby(["cmdb_id", "name"], as_index=False)["value"]
            .quantile(0.95)
            .rename(columns={"value": "p95"})
        )
        df = df.merge(thresholds, on=["cmdb_id", "name"], how="left").dropna(subset=["p95"]).copy()
        denom = df["p95"].abs().where(df["p95"].abs() > 1e-6, 1.0)
        df["breach_ratio"] = (df["value"] - df["p95"]) / denom
        df["breach_ratio"] = df["breach_ratio"].clip(lower=0.0, upper=10.0)
        df = df[df["breach_ratio"] > 0].copy()
        if df.empty:
            continue
        by_comp = df.groupby("cmdb_id", as_index=False)["breach_ratio"].sum()
        for _, r in by_comp.iterrows():
            c = str(r["cmdb_id"])
            total[c] = total.get(c, 0.0) + float(r["breach_ratio"])
    if not total:
        return "", 0.0
    comp, s = sorted(total.items(), key=lambda kv: kv[1], reverse=True)[0]
    return comp, float(s)


def _best_component_for_component_field(metric_dir: str, start: pd.Timestamp, end: pd.Timestamp, instruction: str) -> str:
    reason, ev = infer_reason_with_evidence(instruction)

    def best_comp(metric_file: str, prefix: str, keywords: List[str]) -> Tuple[str, float]:
        path = os.path.join(metric_dir, metric_file)
        if not os.path.exists(path):
            return "", 0.0
        df = _load_metric(path)
        if df.empty:
            return "", 0.0
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
        if df.empty:
            return "", 0.0
        df["cmdb_id"] = df["cmdb_id"].astype(str)
        df = df[df["cmdb_id"].str.startswith(prefix)].copy()
        if df.empty:
            return "", 0.0
        df["name_norm"] = df["name"].astype(str).map(_norm_name)
        kws = [_norm_name(k) for k in keywords if str(k).strip()]
        mask = False
        for k in kws:
            mask = mask | df["name_norm"].str.contains(k, na=False)
        df = df[mask].copy()
        if df.empty:
            return "", 0.0
        thresholds = (
            df.groupby(["cmdb_id", "name"], as_index=False)["value"]
            .quantile(0.95)
            .rename(columns={"value": "p95"})
        )
        df = df.merge(thresholds, on=["cmdb_id", "name"], how="left").dropna(subset=["p95"]).copy()
        denom = df["p95"].abs().where(df["p95"].abs() > 1e-6, 1.0)
        df["breach_ratio"] = (df["value"] - df["p95"]) / denom
        df["breach_ratio"] = df["breach_ratio"].clip(lower=0.0, upper=10.0)
        df = df[df["breach_ratio"] > 0].copy()
        if df.empty:
            return "", 0.0
        by = df.groupby("cmdb_id", as_index=False)["breach_ratio"].sum().sort_values("breach_ratio", ascending=False)
        return str(by.iloc[0]["cmdb_id"]), float(by.iloc[0]["breach_ratio"])

    cpu_node_kws = ["proc_used_pct", "proc_user_used_pct", "cpu", "load", "iowait", "util"]
    cpu_cont_kws = ["cpu", "load", "util"]
    db_conn_kws = ["login_per_sec", "login", "tps_per_sec", "tps", "exec_per_sec", "exec", "sess_active", "session"]
    db_close_kws = ["on_off_state", "on_off", "logout", "disconnect", "hang", "shutdown", "restart", "close", "reset"]
    net_kws = ["tnsping_result_time", "tnsping", "ping", "rtt", "latency", "delay", "timeout", "retrans", "loss", "drop", "reset"]

    os_cpu_comp, os_cpu_s = best_comp("metric_node.csv", "os_", cpu_node_kws)
    os_net_comp, os_net_s = best_comp("metric_node.csv", "os_", net_kws)
    db_conn_comp, db_conn_s = best_comp("metric_service.csv", "db_", db_conn_kws)
    db_close_comp, db_close_s = best_comp("metric_service.csv", "db_", db_close_kws)
    docker_cpu_comp, docker_cpu_s = best_comp("metric_container.csv", "docker_", cpu_cont_kws)

    if reason in ("network delay", "network loss"):
        return os_net_comp or os_cpu_comp or "os_001"
    if reason == "db close" and db_close_comp:
        return db_close_comp
    if reason == "db connection limit" and db_conn_comp:
        return db_conn_comp

    db_s = max(db_conn_s, db_close_s)
    db_comp = db_conn_comp if db_conn_s >= db_close_s else db_close_comp
    if db_s > 0 and os_cpu_s > 0 and db_s >= os_cpu_s * 1.3:
        return db_comp
    if os_cpu_comp:
        return os_cpu_comp
    if os_net_comp:
        return os_net_comp
    if db_comp:
        return db_comp
    if docker_cpu_comp:
        return docker_cpu_comp
    return _best_component_by_total_breach(metric_dir, start, end) or "os_001"


def component_candidates_with_evidence(instruction: str) -> Tuple[str, List[str], Dict[str, Any]]:
    date_dir, start, end = _parse_time_window(instruction)
    metric_dir = os.path.join("dataset", "Telecom", "telemetry", date_dir, "metric")

    cpu_node_kws = ["proc_used_pct", "proc_user_used_pct", "cpu", "load", "iowait", "util"]
    net_kws = ["tnsping_result_time", "tnsping", "ping", "rtt", "latency", "delay", "timeout", "retrans", "loss", "drop", "reset"]
    db_conn_kws = ["login_per_sec", "login", "tps_per_sec", "tps", "exec_per_sec", "exec", "sess_active", "session"]
    db_close_kws = ["on_off_state", "on_off", "logout", "disconnect", "hang", "shutdown", "restart", "close", "reset"]

    def top_comps(metric_file: str, prefix: str, keywords: List[str], top_k: int) -> List[Tuple[str, float, str]]:
        path = os.path.join(metric_dir, metric_file)
        if not os.path.exists(path):
            return []
        df = _load_metric(path)
        if df.empty:
            return []
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
        if df.empty:
            return []
        df["cmdb_id"] = df["cmdb_id"].astype(str)
        df = df[df["cmdb_id"].str.startswith(prefix)].copy()
        if df.empty:
            return []
        df["name_norm"] = df["name"].astype(str).map(_norm_name)
        kws = [_norm_name(k) for k in keywords if str(k).strip()]
        mask = False
        for k in kws:
            mask = mask | df["name_norm"].str.contains(k, na=False)
        df = df[mask].copy()
        if df.empty:
            return []
        thresholds = (
            df.groupby(["cmdb_id", "name"], as_index=False)["value"]
            .quantile(0.95)
            .rename(columns={"value": "p95"})
        )
        df = df.merge(thresholds, on=["cmdb_id", "name"], how="left").dropna(subset=["p95"]).copy()
        denom = df["p95"].abs().where(df["p95"].abs() > 1e-6, 1.0)
        df["breach_ratio"] = (df["value"] - df["p95"]) / denom
        df["breach_ratio"] = df["breach_ratio"].clip(lower=0.0, upper=10.0)
        df = df[df["breach_ratio"] > 0].copy()
        if df.empty:
            return []
        by = df.groupby("cmdb_id", as_index=False)["breach_ratio"].sum().sort_values("breach_ratio", ascending=False).head(top_k)
        peak_map: Dict[str, str] = {}
        df["minute"] = df["timestamp"].dt.floor("min")
        for comp, g in df.groupby("cmdb_id"):
            m = g.groupby("minute", as_index=False)["breach_ratio"].sum().sort_values("breach_ratio", ascending=False)
            if not m.empty:
                peak_map[str(comp)] = pd.Timestamp(m.iloc[0]["minute"]).strftime("%Y-%m-%d %H:%M:%S")
        out = []
        for _, r in by.iterrows():
            c = str(r["cmdb_id"])
            out.append((c, float(r["breach_ratio"]), peak_map.get(c, "")))
        return out

    os_cpu = top_comps("metric_node.csv", "os_", cpu_node_kws, 8)
    os_net = top_comps("metric_node.csv", "os_", net_kws, 6)
    db_conn = top_comps("metric_service.csv", "db_", db_conn_kws, 6)
    db_close = top_comps("metric_service.csv", "db_", db_close_kws, 6)

    det = _best_component_for_component_field(metric_dir, start, end, instruction)
    cand = []
    for seq in (os_cpu, os_net, db_conn, db_close):
        for c, _, _ in seq:
            cand.append(c)
    cand = list(dict.fromkeys([det] + cand))[:15]

    t1 = estimate_time_from_trace(instruction)
    evidence = {
        "time_window": {"date_dir": date_dir, "start": start.strftime("%Y-%m-%d %H:%M:%S"), "end": end.strftime("%Y-%m-%d %H:%M:%S")},
        "task1_estimated_time": t1,
        "deterministic_choice": det,
        "os_cpu_top": [{"component": c, "score": s, "peak_time": pt} for c, s, pt in os_cpu[:6]],
        "os_net_top": [{"component": c, "score": s, "peak_time": pt} for c, s, pt in os_net[:6]],
        "db_conn_top": [{"component": c, "score": s, "peak_time": pt} for c, s, pt in db_conn[:6]],
        "db_close_top": [{"component": c, "score": s, "peak_time": pt} for c, s, pt in db_close[:6]],
    }
    return det, cand, evidence


def pick_component_from_granger(instruction: str) -> str:
    date_dir, start, end = _parse_time_window(instruction)
    metric_dir = os.path.join("dataset", "Telecom", "telemetry", date_dir, "metric")
    comp = _best_component_by_total_breach(metric_dir, start, end)
    if comp:
        return comp
    ranked = _rank_components(metric_dir, start, end)
    if not ranked.empty:
        return str(ranked.iloc[0]["component"])
    return "os_001"


def estimate_time_from_trace(instruction: str) -> str:
    from rca.time_candidates import estimate_occurrence_datetime_from_trace

    dt, _ = estimate_occurrence_datetime_from_trace(dataset="Telecom", instruction=instruction)
    return str(dt or "")
