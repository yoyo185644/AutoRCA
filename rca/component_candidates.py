import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class ComponentEvidence:
    dataset: str
    date_dir: str
    start_local: str
    end_local: str
    candidates: List[str]
    metric_scores: Dict[str, float] = field(default_factory=dict)
    trace_scores: Dict[str, float] = field(default_factory=dict)
    combined_scores: Dict[str, float] = field(default_factory=dict)
    peak_times: Dict[str, str] = field(default_factory=dict)
    first_times: Dict[str, str] = field(default_factory=dict)
    top_metrics: Dict[str, List[str]] = field(default_factory=dict)
    estimated_failure_time: str = ""
    local_start_s: int = 0
    local_end_s: int = 0
    top1: str = ""
    top2: str = ""
    gap_norm: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _norm_name(s: str) -> str:
    s = str(s or "").lower()
    return re.sub(r"[^a-z0-9]+", "", s)


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


def parse_time_window_generic(instruction: str, tz: str = "Asia/Shanghai") -> Tuple[str, pd.Timestamp, pd.Timestamp]:
    text = str(instruction)
    text_l = text.lower()
    date = ""
    y = mo = d = None
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", text)
    if m:
        mon, day, year = m.groups()
        mo = _MONTH[mon.lower()]
        d = int(day)
        y = int(year)
        date = f"{y:04d}-{mo:02d}-{d:02d}"
    else:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if m:
            date = m.group(1)
            y, mo, d = [int(x) for x in date.split("-")]

    hm1 = hm2 = ""
    m = re.search(r"from\s*(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})", text_l)
    if m:
        hm1, hm2 = m.groups()
    else:
        m = re.search(r"between\s*(\d{1,2}:\d{2})\s*(?:and|to)\s*(\d{1,2}:\d{2})", text_l)
        if m:
            hm1, hm2 = m.groups()

    if date and hm1 and hm2 and y is not None and mo is not None and d is not None:
        start = pd.Timestamp(f"{date} {hm1}:00", tz=tz)
        end = pd.Timestamp(f"{date} {hm2}:00", tz=tz)
        return f"{y:04d}_{mo:02d}_{d:02d}", start, end

    raise ValueError("Cannot parse time window from instruction.")


def _to_epoch_seconds(ts: pd.Timestamp) -> int:
    if ts.tz is None:
        ts = ts.tz_localize("Asia/Shanghai")
    return int(ts.tz_convert("UTC").timestamp())


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


def _metric_component_scores(
    metric_dir: str, start_s: int, end_s: int, top_k_metrics: int = 8
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    paths = [os.path.join(metric_dir, p) for p in os.listdir(metric_dir) if p.endswith(".csv")]
    total: Dict[str, float] = {}
    peak_time: Dict[str, str] = {}
    first_time: Dict[str, str] = {}
    top_metrics: Dict[str, List[str]] = {}

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

        by_metric = (
            win.groupby(["component", "metric"], as_index=False)["breach_ratio"]
            .sum()
            .sort_values(["component", "breach_ratio"], ascending=[True, False])
        )
        topm = by_metric.groupby("component", as_index=False).head(int(top_k_metrics))
        by_comp = topm.groupby("component", as_index=False)["breach_ratio"].sum()
        for _, r in by_comp.iterrows():
            c = str(r["component"])
            total[c] = total.get(c, 0.0) + float(r["breach_ratio"])

        win["minute"] = (win["ts_s"] // 60) * 60
        for c, g in win.groupby("component"):
            m = g.groupby("minute", as_index=False)["breach_ratio"].sum().sort_values("breach_ratio", ascending=False)
            if not m.empty and str(c) not in peak_time:
                peak_s = int(m.iloc[0]["minute"])
                peak_time[str(c)] = pd.to_datetime(peak_s, unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
            if not m.empty and str(c) not in first_time:
                ms = g.groupby("minute", as_index=False)["breach_ratio"].sum().sort_values("minute")
                mx = float(ms["breach_ratio"].max()) if not ms.empty else 0.0
                if mx > 0:
                    thr = 0.30 * mx
                    cand = ms[ms["breach_ratio"] >= thr]
                    if not cand.empty:
                        first_s = int(cand.iloc[0]["minute"])
                        first_time[str(c)] = pd.to_datetime(first_s, unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
            mt = (
                g.groupby("metric", as_index=False)["breach_ratio"]
                .sum()
                .sort_values("breach_ratio", ascending=False)
                .head(top_k_metrics)["metric"]
                .astype(str)
                .tolist()
            )
            if mt and str(c) not in top_metrics:
                top_metrics[str(c)] = mt

    return total, peak_time, first_time, top_metrics


def _trace_component_scores(trace_path: str, start_s: int, end_s: int) -> Dict[str, float]:
    if not os.path.exists(trace_path):
        return {}
    df = pd.read_csv(trace_path, low_memory=False)
    if df.empty or "cmdb_id" not in df.columns:
        return {}
    ts_col = "timestamp" if "timestamp" in df.columns else ("startTime" if "startTime" in df.columns else None)
    dur_col = "duration" if "duration" in df.columns else ("elapsedTime" if "elapsedTime" in df.columns else None)
    if ts_col is None or dur_col is None:
        return {}
    df = df[[ts_col, "cmdb_id", dur_col]].copy()
    df["ts_s"] = _ts_to_seconds(df[ts_col])
    df["cmdb_id"] = df["cmdb_id"].astype(str)
    df["dur"] = pd.to_numeric(df[dur_col], errors="coerce")
    df = df.dropna(subset=["ts_s", "cmdb_id", "dur"])
    df = df[(df["ts_s"] >= start_s) & (df["ts_s"] <= end_s)].copy()
    if df.empty:
        return {}
    p95 = df.groupby("cmdb_id")["dur"].quantile(0.95)
    baseline = float(p95.median()) if not p95.empty else 0.0
    if baseline <= 0:
        return {}
    out = ((p95 / baseline) - 1.0).clip(lower=0.0)
    return {str(k): float(v) for k, v in out.items()}


def _estimate_failure_time_from_trace(trace_path: str, start_s: int, end_s: int) -> int:
    if not os.path.exists(trace_path):
        return int((start_s // 60) * 60)
    df = pd.read_csv(trace_path, low_memory=False)
    if df.empty:
        return int((start_s // 60) * 60)
    ts_col = "timestamp" if "timestamp" in df.columns else ("startTime" if "startTime" in df.columns else None)
    dur_col = "duration" if "duration" in df.columns else ("elapsedTime" if "elapsedTime" in df.columns else None)
    if ts_col is None or dur_col is None:
        return int((start_s // 60) * 60)
    df = df[[ts_col, dur_col]].copy()
    df["ts_s"] = _ts_to_seconds(df[ts_col])
    df["dur"] = pd.to_numeric(df[dur_col], errors="coerce")
    df = df.dropna(subset=["ts_s", "dur"])
    df = df[(df["ts_s"] >= start_s) & (df["ts_s"] <= end_s)].copy()
    if df.empty:
        return int((start_s // 60) * 60)
    df["minute"] = (df["ts_s"] // 60) * 60
    p95 = df.groupby("minute")["dur"].quantile(0.95).sort_index()
    if p95.empty:
        return int((start_s // 60) * 60)
    baseline = float(p95.median())
    diff = p95.diff().fillna(0.0)
    thr_level = baseline * 1.2
    if baseline > 0:
        mx = float(p95.max())
        if mx > 0:
            strong = p95[p95 >= max(thr_level, 0.8 * mx)]
            if not strong.empty:
                return int(strong.index[0])
    thr_jump = max(1.0, baseline * 0.1)
    candidates = p95[(p95 >= thr_level) & (diff >= thr_jump)]
    if not candidates.empty:
        return int(candidates.index[0])
    return int(p95.sort_values(ascending=False).index[0])


def _comp_type(comp: str) -> str:
    c = str(comp or "")
    if c.startswith(("os_", "node-")):
        return "node"
    if c.startswith(("docker_", "container")):
        return "container"
    if c.startswith("db_"):
        return "db"
    if "-" in c:
        return "service"
    return "other"


def build_component_candidates(
    dataset: str,
    instruction: str,
    top_k: int = 12,
    w_metric: float = 1.0,
    w_trace: float = 1.0,
    center_time_s: Optional[int] = None,
    delta_s: int = 300,
) -> ComponentEvidence:
    date_dir, start, end = parse_time_window_generic(instruction)
    metric_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "metric")
    trace_path = os.path.join("dataset", dataset, "telemetry", date_dir, "trace", "trace_span.csv")
    start_s, end_s = _to_epoch_seconds(start), _to_epoch_seconds(end)

    t_star = int(center_time_s) if center_time_s is not None else _estimate_failure_time_from_trace(trace_path, start_s, end_s)
    delta_s = int(delta_s) if int(delta_s) > 0 else 300
    local_start_s = max(int(start_s), int(t_star) - delta_s)
    local_end_s = min(int(end_s), int(t_star) + delta_s)
    if local_end_s <= local_start_s:
        local_start_s, local_end_s = int(start_s), int(end_s)

    metric_scores, peak_times, first_times, top_metrics = _metric_component_scores(metric_dir, local_start_s, local_end_s)
    trace_scores = _trace_component_scores(trace_path, local_start_s, local_end_s)

    def norm(m: Dict[str, float]) -> Dict[str, float]:
        if not m:
            return {}
        mx = max(m.values())
        if mx <= 1e-12:
            return {k: 0.0 for k in m.keys()}
        return {k: float(v) / mx for k, v in m.items()}

    nm = norm(metric_scores)
    nt = norm(trace_scores)

    comps = set(nm.keys()) | set(nt.keys())
    combined = {c: w_metric * nm.get(c, 0.0) + w_trace * nt.get(c, 0.0) for c in comps}
    align_max_s = 300
    align_w = 0.70
    onset_max_s = 600
    onset_w = 0.10
    lead_max_s = 900
    lead_w = 0.25
    peak_s_map: Dict[str, int] = {}
    for c, ts in peak_times.items():
        try:
            peak_s_map[str(c)] = int(pd.Timestamp(ts, tz="Asia/Shanghai").tz_convert("UTC").timestamp())
        except Exception:
            continue
    first_s_map: Dict[str, int] = {}
    for c, ts in first_times.items():
        try:
            first_s_map[str(c)] = int(pd.Timestamp(ts, tz="Asia/Shanghai").tz_convert("UTC").timestamp())
        except Exception:
            continue

    combined: Dict[str, float] = {}
    for c in comps:
        base = w_metric * nm.get(c, 0.0) + w_trace * nt.get(c, 0.0)
        if c in peak_s_map:
            dist = abs(float(peak_s_map[c]) - float(t_star))
            align = max(0.0, 1.0 - dist / float(align_max_s))
            base = base + align_w * align
        if c in first_s_map:
            dist0 = abs(float(first_s_map[c]) - float(t_star))
            onset = max(0.0, 1.0 - dist0 / float(onset_max_s))
            base = base + onset_w * onset
            if _comp_type(c) in ("db", "service"):
                lead = float(t_star) - float(first_s_map[c])
                lead_score = max(0.0, min(1.0, lead / float(lead_max_s)))
                base = base + lead_w * lead_score
        combined[c] = float(base)

    first_vals = [v for v in first_s_map.values() if isinstance(v, int)]
    if first_vals:
        median_first = float(pd.Series(first_vals).median())
        lead_max_s = 900.0
        lead_w = 0.25
        for c, f in first_s_map.items():
            if nm.get(c, 0.0) < 0.15 and nt.get(c, 0.0) < 0.15:
                continue
            lead = max(0.0, (median_first - float(f)) / lead_max_s)
            lead = min(1.0, lead)
            combined[c] = float(combined.get(c, 0.0) + lead_w * lead)

    trace_top = ""
    trace_top_nt = 0.0
    if nt:
        trace_top = sorted(nt.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        trace_top_nt = float(nt.get(trace_top, 0.0))
        if trace_top and _comp_type(trace_top) in ("db", "service") and trace_top in combined and trace_top_nt >= 0.85:
            combined[trace_top] = float(combined[trace_top] + 0.35)
        if trace_top and _comp_type(trace_top) in ("node", "container") and trace_top_nt >= 0.85:
            for c in list(combined.keys()):
                if _comp_type(c) == "db" and float(nt.get(c, 0.0)) < 0.55:
                    combined[c] = float(max(0.0, combined[c] - 0.65))

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    by_type: Dict[str, List[str]] = {}
    for c, _ in ranked:
        by_type.setdefault(_comp_type(c), []).append(c)
    n_each = max(1, int(max(1, int(top_k)) // 3))
    candidates: List[str] = []
    for t in ["node", "service", "db", "container", "other"]:
        for c in by_type.get(t, [])[:n_each]:
            candidates.append(c)
    for c, _ in ranked:
        if len(candidates) >= max(1, int(top_k)):
            break
        if c not in candidates:
            candidates.append(c)
    ranked_cand = sorted(candidates, key=lambda c: combined.get(c, 0.0), reverse=True)
    top1 = ranked_cand[0] if ranked_cand else ""
    top2 = ranked_cand[1] if len(ranked_cand) > 1 else ""
    s1 = combined.get(top1, 0.0)
    s2 = combined.get(top2, 0.0) if top2 else 0.0
    gap_norm = abs(s1 - s2) / max(abs(s1), 1e-9)
    if top1 and top2 and _comp_type(top1) == _comp_type(top2) and float(gap_norm) < 0.05:
        p1 = peak_s_map.get(top1)
        p2 = peak_s_map.get(top2)
        if p1 is not None and p2 is not None:
            d1 = abs(float(p1) - float(t_star))
            d2 = abs(float(p2) - float(t_star))
            if d2 + 1e-9 < d1:
                top1, top2 = top2, top1
                s1, s2 = s2, s1
                gap_norm = abs(s1 - s2) / max(abs(s1), 1e-9)
        f1 = first_s_map.get(top1)
        f2 = first_s_map.get(top2)
        if f1 is not None and f2 is not None:
            if float(f2) + 1e-9 < float(f1):
                top1, top2 = top2, top1
                s1, s2 = s2, s1
                gap_norm = abs(s1 - s2) / max(abs(s1), 1e-9)
        if trace_top and trace_top in (top1, top2) and _comp_type(trace_top) in ("db", "service"):
            top1 = trace_top
            s1 = combined.get(top1, 0.0)
            top2 = top2 if top2 != top1 else (ranked_cand[2] if len(ranked_cand) > 2 else "")
            s2 = combined.get(top2, 0.0) if top2 else 0.0
            gap_norm = abs(s1 - s2) / max(abs(s1), 1e-9)

    t_star_local = pd.to_datetime(int(t_star), unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
    expl = f"metric_w={w_metric}, trace_w={w_trace}, align_w={align_w}, onset_w={onset_w}, lead_w={lead_w}, t*={t_star_local}, local=[{local_start_s},{local_end_s}], top1={top1}, top2={top2}, gap_norm={gap_norm:.4f}"
    return ComponentEvidence(
        dataset=dataset,
        date_dir=date_dir,
        start_local=start.strftime("%Y-%m-%d %H:%M:%S"),
        end_local=end.strftime("%Y-%m-%d %H:%M:%S"),
        candidates=candidates,
        metric_scores=metric_scores,
        trace_scores=trace_scores,
        combined_scores=combined,
        peak_times=peak_times,
        first_times=first_times,
        top_metrics=top_metrics,
        estimated_failure_time=t_star_local,
        local_start_s=int(local_start_s),
        local_end_s=int(local_end_s),
        top1=top1,
        top2=top2,
        gap_norm=float(gap_norm),
        explanation=expl,
    )




def apply_component_to_prediction(prediction: str, component: str) -> str:
    try:
        obj = json.loads(prediction)
        if not isinstance(obj, dict):
            obj = {}
    except Exception:
        obj = {}
    if "1" not in obj or not isinstance(obj.get("1"), dict):
        obj["1"] = {}
    obj["1"]["root cause component"] = str(component)
    return json.dumps(obj, ensure_ascii=False)
