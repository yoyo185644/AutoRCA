from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from rca.adapter.utils.component_candidates import _normalize_metric_file, parse_time_window_generic
from rca.common.io import resolve_date_dir
from rca.index.extractor import extract_trace_topology_edges
from rca.index.telemetry_cache import TelemetryCache
from rca.adapter.utils.time_candidates import TimeCandidate


@dataclass(frozen=True)
class CandidateEvidence:
    component: str
    reason: str
    metric_timeline: List[Dict[str, Any]] = field(default_factory=list)
    trace_timeline: List[Dict[str, Any]] = field(default_factory=list)
    causal_evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    score: float = 0.0
    score_gap_vs_next: float = 0.0
    first_anomaly_s: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReasonEvidence:
    """Deviation/rank/contrast-based reason evidence -- replaces the old
    "does a reason keyword appear in this component's top-8 anomalous
    metrics" binary check, which turned out to be non-discriminative: during
    a real incident, CPU/memory/disk metrics on the same host frequently
    breach together, so multiple *different* reason labels on the *same*
    component (and even on different, wrong components) would all come back
    "matches" with similar hit counts -- verified empirically on Bank row1
    (Redis02/"high memory usage" and IG01/"high disk I/O read usage" both
    scored hits=3 with the old check, giving no way to tell them apart).

    This version scores each reason by the actual deviation magnitude of its
    metric group (not just presence/absence), ranks that reason against every
    other known reason on the same component, and -- when a candidate list is
    supplied -- percentile-ranks it against the other candidates' own claimed
    (component, reason) pairs. `support` collapses these into one label so a
    consuming agent has a clear signal, but the raw numbers are also exposed
    so the agent (or a human) can second-guess the label.
    """

    component: str
    reason: str
    reason_known: bool
    support: str
    reason_deviation: float
    reason_rank_on_component: int
    gap_to_second_reason: float
    cross_candidate_percentile: Optional[float]
    matched_metrics: List[Dict[str, Any]] = field(default_factory=list)
    competing_reasons: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimeEvidence:
    datetime: str
    source: str
    metric_at_this_time: List[Dict[str, Any]] = field(default_factory=list)
    trace_at_this_time: Dict[str, Any] = field(default_factory=dict)
    metric_1min_before: List[Dict[str, Any]] = field(default_factory=list)
    metric_1min_after: List[Dict[str, Any]] = field(default_factory=list)
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_epoch_s_local(dt: str) -> Optional[int]:
    text = str(dt or "").strip()
    if not text:
        return None
    try:
        return int(pd.Timestamp(text, tz="Asia/Shanghai").tz_convert("UTC").timestamp())
    except Exception:
        return None


def _minute_bucket_s(ts_s: int) -> int:
    return int((int(ts_s) // 60) * 60)


def _minute_label_local(ts_s: int) -> str:
    return pd.to_datetime(int(ts_s), unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%H:%M")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


@lru_cache(maxsize=64)
def _metric_long_cached(dataset: str, date_dir: str) -> pd.DataFrame:
    metric_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "metric")
    if not os.path.isdir(metric_dir):
        return pd.DataFrame(columns=["ts_s", "component", "metric", "value"])
    frames = []
    for name in os.listdir(metric_dir):
        if not name.endswith(".csv"):
            continue
        p = os.path.join(metric_dir, name)
        try:
            long = _normalize_metric_file(p)
        except Exception:
            continue
        if not long.empty:
            frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["ts_s", "component", "metric", "value"])
    out = pd.concat(frames, ignore_index=True)
    out["ts_s"] = pd.to_numeric(out["ts_s"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["ts_s", "component", "metric", "value"])
    out["ts_s"] = out["ts_s"].astype("int64")
    out["component"] = out["component"].astype(str)
    out["metric"] = out["metric"].astype(str)
    return out[["ts_s", "component", "metric", "value"]]


@lru_cache(maxsize=64)
def _metric_p95_map_cached(dataset: str, date_dir: str) -> pd.DataFrame:
    long = _metric_long_cached(dataset, date_dir)
    if long.empty:
        return pd.DataFrame(columns=["component", "metric", "p95"])
    p95 = (
        long.groupby(["component", "metric"], as_index=False)["value"]
        .quantile(0.95)
        .rename(columns={"value": "p95"})
    )
    p95["p95"] = pd.to_numeric(p95["p95"], errors="coerce")
    return p95.dropna(subset=["p95"]).copy()


@lru_cache(maxsize=64)
def _trace_minute_p95_cached(dataset: str, date_dir: str) -> pd.DataFrame:
    trace_path = os.path.join("dataset", dataset, "telemetry", date_dir, "trace", "trace_span.csv")
    if not os.path.exists(trace_path):
        return pd.DataFrame(columns=["minute", "component", "p95"])
    try:
        cached = TelemetryCache.get_trace_minute_p95(dataset, date_dir, trace_path)
        if not cached.empty:
            out = cached.copy()
            out["minute"] = pd.to_numeric(out["minute"], errors="coerce")
            out["p95"] = pd.to_numeric(out["p95"], errors="coerce")
            out["component"] = out["component"].astype(str)
            out = out.dropna(subset=["minute", "component", "p95"])
            out["minute"] = out["minute"].astype("int64")
            return out[["minute", "component", "p95"]]
    except Exception:
        pass
    return pd.DataFrame(columns=["minute", "component", "p95"])


def _p95_lookup(p95_df: pd.DataFrame, component: str, metric: str) -> float:
    if p95_df.empty:
        return 0.0
    sub = p95_df[(p95_df["component"] == str(component)) & (p95_df["metric"] == str(metric))]
    if sub.empty:
        return 0.0
    return _safe_float(sub.iloc[0]["p95"], 0.0)


def _global_p95_lookup(p95_df: pd.DataFrame, metric: str) -> float:
    if p95_df.empty:
        return 0.0
    sub = p95_df[p95_df["metric"] == str(metric)]
    if sub.empty:
        return 0.0
    return _safe_float(sub["p95"].median(), 0.0)


def _pick_metrics_for_component(metric_long: pd.DataFrame, p95_df: pd.DataFrame, component: str, start_s: int, end_s: int, top_k: int) -> List[str]:
    if metric_long.empty or p95_df.empty or not str(component or "").strip():
        return []
    win = metric_long[(metric_long["component"] == str(component)) & (metric_long["ts_s"] >= int(start_s)) & (metric_long["ts_s"] <= int(end_s))].copy()
    if win.empty:
        return []
    win = win.merge(p95_df, on=["component", "metric"], how="left").dropna(subset=["p95"]).copy()
    denom = win["p95"].abs().where(win["p95"].abs() > 1e-6, 1.0)
    win["breach_ratio"] = ((win["value"] - win["p95"]) / denom).clip(lower=0.0, upper=10.0)
    win = win[win["breach_ratio"] > 0].copy()
    if win.empty:
        return []
    by = win.groupby("metric", as_index=False)["breach_ratio"].sum().sort_values("breach_ratio", ascending=False)
    return by.head(int(top_k))["metric"].astype(str).tolist()


def _pick_metrics_global(metric_long: pd.DataFrame, p95_df: pd.DataFrame, start_s: int, end_s: int, top_k: int) -> List[str]:
    if metric_long.empty or p95_df.empty:
        return []
    win = metric_long[(metric_long["ts_s"] >= int(start_s)) & (metric_long["ts_s"] <= int(end_s))].copy()
    if win.empty:
        return []
    win = win.merge(p95_df, on=["component", "metric"], how="left").dropna(subset=["p95"]).copy()
    denom = win["p95"].abs().where(win["p95"].abs() > 1e-6, 1.0)
    win["breach_ratio"] = ((win["value"] - win["p95"]) / denom).clip(lower=0.0, upper=10.0)
    win = win[win["breach_ratio"] > 0].copy()
    if win.empty:
        return []
    by = win.groupby("metric", as_index=False)["breach_ratio"].sum().sort_values("breach_ratio", ascending=False)
    return by.head(int(top_k))["metric"].astype(str).tolist()


def _metric_minute_summary(
    metric_long: pd.DataFrame,
    p95_df: pd.DataFrame,
    component: str,
    metrics: Sequence[str],
    start_s: int,
    end_s: int,
    limit_per_metric: int = 6,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    if metric_long.empty or not metrics:
        return [], None
    sub = metric_long[
        (metric_long["component"] == str(component))
        & (metric_long["metric"].isin([str(m) for m in metrics]))
        & (metric_long["ts_s"] >= int(start_s))
        & (metric_long["ts_s"] <= int(end_s))
    ].copy()
    if sub.empty:
        return [], None
    sub["minute"] = (sub["ts_s"].astype(int) // 60) * 60
    agg = sub.groupby(["metric", "minute"], as_index=False)["value"].mean()
    agg["baseline"] = agg.apply(lambda r: _p95_lookup(p95_df, component, str(r["metric"])), axis=1)
    agg["ratio"] = agg.apply(lambda r: (_safe_float(r["value"]) / _safe_float(r["baseline"], 1e-9)) if _safe_float(r["baseline"]) > 0 else None, axis=1)
    agg = agg.sort_values(["metric", "minute"]).reset_index(drop=True)

    first_breach_by_metric: Dict[str, int] = {}
    for m, g in agg.groupby("metric"):
        gg = g.copy()
        gg["is_breach"] = gg.apply(lambda r: _safe_float(r["value"]) > _safe_float(r["baseline"]), axis=1)
        hits = gg[gg["is_breach"]]
        if not hits.empty:
            first_breach_by_metric[str(m)] = int(hits.iloc[0]["minute"])

    rows: List[Dict[str, Any]] = []
    first_any: Optional[int] = None
    for m, g in agg.groupby("metric"):
        m = str(m)
        g = g.copy()
        fb = first_breach_by_metric.get(m)
        if fb is not None:
            if first_any is None or int(fb) < int(first_any):
                first_any = int(fb)
        g["is_first_breach"] = g["minute"].astype(int).apply(lambda x: bool(fb is not None and int(x) == int(fb)))
        g["anom_strength"] = g.apply(
            lambda r: (_safe_float(r["value"]) - _safe_float(r["baseline"])) / max(_safe_float(r["baseline"]), 1e-9)
            if _safe_float(r["baseline"]) > 0
            else 0.0,
            axis=1,
        )
        key = g[(g["is_first_breach"]) | (g["anom_strength"] >= 0.10)].copy()
        if key.empty:
            continue
        key = key.sort_values(["anom_strength", "minute"], ascending=[False, True]).head(int(limit_per_metric))
        key = key.sort_values("minute")
        for _, r in key.iterrows():
            rows.append(
                {
                    "minute": _minute_label_local(int(r["minute"])),
                    "metric": str(r["metric"]),
                    "value": round(_safe_float(r["value"]), 4),
                    "baseline": round(_safe_float(r["baseline"]), 4),
                    "ratio": round(_safe_float(r["ratio"]), 4) if r.get("ratio") is not None else None,
                    "is_first_breach": bool(r["is_first_breach"]),
                }
            )
    rows = sorted(rows, key=lambda x: (x.get("minute", ""), x.get("metric", "")))
    return rows, first_any


def _metric_minute_summary_global(
    metric_long: pd.DataFrame,
    p95_df: pd.DataFrame,
    metrics: Sequence[str],
    start_s: int,
    end_s: int,
    limit_per_metric: int = 6,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    if metric_long.empty or not metrics:
        return [], None
    sub = metric_long[
        (metric_long["metric"].isin([str(m) for m in metrics]))
        & (metric_long["ts_s"] >= int(start_s))
        & (metric_long["ts_s"] <= int(end_s))
    ].copy()
    if sub.empty:
        return [], None
    sub["minute"] = (sub["ts_s"].astype(int) // 60) * 60
    agg = sub.groupby(["metric", "minute"], as_index=False)["value"].mean()
    agg["baseline"] = agg["metric"].astype(str).apply(lambda m: _global_p95_lookup(p95_df, m))
    agg["ratio"] = agg.apply(lambda r: (_safe_float(r["value"]) / _safe_float(r["baseline"], 1e-9)) if _safe_float(r["baseline"]) > 0 else None, axis=1)
    agg = agg.sort_values(["metric", "minute"]).reset_index(drop=True)

    first_breach_by_metric: Dict[str, int] = {}
    for m, g in agg.groupby("metric"):
        gg = g.copy()
        gg["is_breach"] = gg.apply(lambda r: _safe_float(r["value"]) > _safe_float(r["baseline"]), axis=1)
        hits = gg[gg["is_breach"]]
        if not hits.empty:
            first_breach_by_metric[str(m)] = int(hits.iloc[0]["minute"])

    rows: List[Dict[str, Any]] = []
    first_any: Optional[int] = None
    for m, g in agg.groupby("metric"):
        m = str(m)
        g = g.copy()
        fb = first_breach_by_metric.get(m)
        if fb is not None:
            if first_any is None or int(fb) < int(first_any):
                first_any = int(fb)
        g["is_first_breach"] = g["minute"].astype(int).apply(lambda x: bool(fb is not None and int(x) == int(fb)))
        g["anom_strength"] = g.apply(
            lambda r: (_safe_float(r["value"]) - _safe_float(r["baseline"])) / max(_safe_float(r["baseline"]), 1e-9)
            if _safe_float(r["baseline"]) > 0
            else 0.0,
            axis=1,
        )
        key = g[(g["is_first_breach"]) | (g["anom_strength"] >= 0.10)].copy()
        if key.empty:
            continue
        key = key.sort_values(["anom_strength", "minute"], ascending=[False, True]).head(int(limit_per_metric))
        key = key.sort_values("minute")
        for _, r in key.iterrows():
            rows.append(
                {
                    "minute": _minute_label_local(int(r["minute"])),
                    "metric": str(r["metric"]),
                    "value": round(_safe_float(r["value"]), 4),
                    "baseline": round(_safe_float(r["baseline"]), 4),
                    "ratio": round(_safe_float(r["ratio"]), 4) if r.get("ratio") is not None else None,
                    "is_first_breach": bool(r["is_first_breach"]),
                }
            )
    rows = sorted(rows, key=lambda x: (x.get("minute", ""), x.get("metric", "")))
    return rows, first_any


def _trace_minute_summary(
    trace_minute_p95: pd.DataFrame,
    component: str,
    start_s: int,
    end_s: int,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    if trace_minute_p95.empty:
        return []
    start_min = _minute_bucket_s(int(start_s))
    end_min = _minute_bucket_s(int(end_s))
    win = trace_minute_p95[
        (trace_minute_p95["minute"] >= int(start_min))
        & (trace_minute_p95["minute"] <= int(end_min))
        & (trace_minute_p95["component"] == str(component))
    ].copy()
    if win.empty:
        return []
    win = win.sort_values("minute").reset_index(drop=True)
    pre_n = max(5, int(len(win) * 0.2))
    baseline = _safe_float(win.iloc[:pre_n]["p95"].median(), 0.0)
    win["ratio"] = win["p95"].apply(lambda v: _safe_float(v) / max(baseline, 1e-9) if baseline > 0 else None)
    win["anom_strength"] = win["ratio"].apply(lambda r: _safe_float(r) - 1.0 if r is not None else 0.0)
    key = win[(win["anom_strength"] >= 0.15)].copy()
    if key.empty:
        top = win.sort_values("anom_strength", ascending=False).head(int(limit)).copy()
    else:
        top = pd.concat([key.head(int(limit)), win.head(2)], ignore_index=True).drop_duplicates(subset=["minute"]).copy()
        top = top.sort_values("minute").head(int(limit)).copy()
    out = []
    for _, r in top.iterrows():
        out.append(
            {
                "minute": _minute_label_local(int(r["minute"])),
                "p95": round(_safe_float(r["p95"]), 4),
                "baseline": round(float(baseline), 4),
                "ratio": round(_safe_float(r["ratio"]), 4) if r.get("ratio") is not None else None,
            }
        )
    return out


_REASON_TOKENS: Dict[str, List[str]] = {
    # Telecom
    "network loss": ["loss", "drop", "retrans", "timeout", "reset", "error"],
    "network delay": ["tnsping", "ping", "rtt", "latency", "delay"],
    "db connection limit": ["login", "session", "sess", "tps", "exec", "connection", "conn"],
    "db close": ["on_off", "logout", "disconnect", "hang", "shutdown", "restart", "close", "reset"],
    "CPU fault": ["proc", "cpu", "load", "iowait", "util"],
    # Bank -- previously missing entirely, so the reason-consistency check was
    # a silent no-op for every Bank query (verified empirically: zero
    # evidence_for/evidence_against lines for any of these 8 labels).
    "JVM Out of Memory (OOM) Heap": ["jvm", "heap", "oom", "gc", "outofmemory"],
    "high CPU usage": ["cpu", "util", "load", "processor"],
    "high JVM CPU load": ["jvm", "cpu", "load", "util"],
    "high disk I/O read usage": ["disk", "read", "r_s", "rkb", "await", "svctm", "io"],
    "high disk space usage": ["disk", "space", "used", "usage", "fs_"],
    "high memory usage": ["mem", "memory", "rss", "used"],
    "network latency": ["latency", "delay", "rtt", "ping"],
    "network packet loss": ["loss", "drop", "packet", "retrans"],
    # Market/cloudbed -- also previously missing (except a couple entries
    # duplicated from the out-of-sync copy in rca/index/indexes.py).
    "container CPU load": ["container", "cpu", "util", "load"],
    "container memory load": ["container", "mem", "memory", "rss"],
    "container network latency": ["container", "latency", "delay", "rtt"],
    "container network packet corruption": ["container", "corrupt", "checksum", "crc", "error"],
    "container network packet retransmission": ["container", "retrans", "retry"],
    "container packet loss": ["container", "loss", "drop", "packet"],
    "container process termination": ["container", "restart", "kill", "terminate", "exit", "oom"],
    "container read I/O load": ["container", "read", "r_s", "rkb", "io"],
    "container write I/O load": ["container", "write", "w_s", "wkb", "io"],
    "node CPU load": ["node", "cpu", "load", "util"],
    "node CPU spike": ["node", "cpu", "spike", "load", "util"],
    "node disk read I/O consumption": ["node", "disk", "read", "r_s", "rkb", "io"],
    "node disk space consumption": ["node", "disk", "space", "used", "usage", "fs_"],
    "node disk write I/O consumption": ["node", "disk", "write", "w_s", "wkb", "io"],
    "node memory consumption": ["node", "mem", "memory", "rss"],
}

# Same vocabulary, exposed under the name used by the deviation-based reason
# evidence path below. Deliberately the *same dict object* (not a copy) so it
# cannot independently drift out of sync the way the old evidence_builder /
# indexes.py copies did.
REASON_METRIC_GROUPS: Dict[str, List[str]] = _REASON_TOKENS

# Each dataset only ever uses its own small reason vocabulary (verified
# against every dataset's record.csv). Reason-ranking has to be restricted to
# this per-dataset subset, not the full 28-label union across all 4 datasets
# -- otherwise e.g. Market-only labels like "container read I/O load" (whose
# tokens include the 2-char substring "io", which matches unrelated Bank
# metrics like "CPUWio") end up "winning" the ranking on Bank components purely
# from token collisions, even though that reason could never actually apply
# there. Confirmed empirically: without this restriction, Bank's Redis02 and
# IG01 both ranked "container read I/O load" above their own true reason.
DATASET_REASON_VOCAB: Dict[str, Tuple[str, ...]] = {
    "Bank": (
        "JVM Out of Memory (OOM) Heap",
        "high CPU usage",
        "high JVM CPU load",
        "high disk I/O read usage",
        "high disk space usage",
        "high memory usage",
        "network latency",
        "network packet loss",
    ),
    "Telecom": (
        "CPU fault",
        "db close",
        "db connection limit",
        "network delay",
        "network loss",
    ),
    "Market": (
        "container CPU load",
        "container memory load",
        "container network latency",
        "container network packet corruption",
        "container network packet retransmission",
        "container packet loss",
        "container process termination",
        "container read I/O load",
        "container write I/O load",
        "node CPU load",
        "node CPU spike",
        "node disk read I/O consumption",
        "node disk space consumption",
        "node disk write I/O consumption",
        "node memory consumption",
    ),
}


def _reason_vocab_for_dataset(dataset: str) -> Tuple[str, ...]:
    ds = str(dataset or "")
    if ds.startswith("Market"):
        return DATASET_REASON_VOCAB["Market"]
    if ds in DATASET_REASON_VOCAB:
        return DATASET_REASON_VOCAB[ds]
    # Unknown dataset name -- fall back to the full vocabulary rather than
    # silently scoring nothing.
    return tuple(REASON_METRIC_GROUPS.keys())

# Thresholds for build_reason_evidence()'s `support` label. "strong" requires
# the queried reason to be the single best-fitting reason on this component by
# a clear margin, and (if peer candidates were supplied) to also stand out
# against them -- not just "some tokens matched somewhere".
_REASON_STRONG_GAP = 1.0
_REASON_STRONG_PERCENTILE = 0.8

# A few reason labels are compound concepts: they only make sense if BOTH a
# domain marker AND a symptom marker are present on the same metric, not
# either one alone. Plain OR-token matching made these indistinguishable from
# their non-JVM counterpart: on real Bank data "high JVM CPU load" and "high
# CPU usage" scored identically on Redis02 (a non-JVM component) because both
# token lists share "cpu"/"load"/"util", so a generic OS-level CPU metric
# satisfied "JVM CPU load" even with zero actual JVM metrics in the match.
# These reasons require a match from every one of the listed token groups
# instead of any token from a single flat list.
_REASON_TOKEN_GROUPS_AND: Dict[str, Tuple[List[str], ...]] = {
    "high JVM CPU load": (["jvm"], ["cpu", "load", "util"]),
    "JVM Out of Memory (OOM) Heap": (["jvm"], ["heap", "oom", "gc", "outofmemory"]),
}


def _metric_matches_reason(metric_name: str, reason: str) -> bool:
    m = str(metric_name or "").lower()
    and_groups = _REASON_TOKEN_GROUPS_AND.get(str(reason or ""))
    if and_groups:
        return all(any(t and t.lower() in m for t in group) for group in and_groups)
    tokens = REASON_METRIC_GROUPS.get(str(reason or ""), [])
    return any(t and t.lower() in m for t in tokens)


@lru_cache(maxsize=64)
def _metric_baseline_stats_cached(dataset: str, date_dir: str) -> pd.DataFrame:
    """Per (component, metric) robust baseline center/spread -- median and IQR
    -- computed from that metric's own full-day distribution.

    This replaces p95-ratio as the normalization used for reason-evidence
    scoring. p95-ratio, (value - p95)/p95, is not comparable across metric
    types: a percent-of-100 metric that's already sitting at a high baseline
    (e.g. MEMUsedMemPerc) has structurally little room to post a large
    *relative* breach even during a real incident, while an already-near-zero
    metric (e.g. iowait%) can show a huge relative multiple for a modest
    absolute move. Verified on real Bank data: Redis02's true root cause,
    "high memory usage", scored lower under breach_ratio than "high CPU
    usage" on the same component/window, purely from this scale mismatch, not
    because memory was actually less anomalous. A median/IQR z-score puts
    every metric on the same "how many of its own typical spreads away from
    its own typical value" scale before anything gets compared.
    """
    long = _metric_long_cached(dataset, date_dir)
    if long.empty:
        return pd.DataFrame(columns=["component", "metric", "median", "iqr"])
    g = long.groupby(["component", "metric"])["value"]
    median = g.median().rename("median")
    p25 = g.quantile(0.25).rename("p25")
    p75 = g.quantile(0.75).rename("p75")
    stats = pd.concat([median, p25, p75], axis=1).reset_index()
    stats["iqr"] = (stats["p75"] - stats["p25"]).abs()
    return stats[["component", "metric", "median", "iqr"]]


@lru_cache(maxsize=512)
def _component_all_metric_deviation_cached(dataset: str, date_dir: str, component: str, start_s: int, end_s: int) -> Dict[str, float]:
    """Mean robust z-score, |value - own_median| / (own_iqr + eps), per metric
    name, for ALL of a component's metrics in the window -- not capped to a
    top-K picked by overall magnitude the way `_pick_metrics_for_component`
    is. Reason-group membership has to be checked against the full metric
    set, otherwise a genuinely reason-relevant metric that didn't make some
    unrelated top-8 cut would silently score as "no evidence" for its own
    reason. Using each metric's own median/IQR (instead of a shared p95-ratio)
    is what makes scores comparable across metric types -- see
    `_metric_baseline_stats_cached` docstring for why that matters.
    """
    metric_long = _metric_long_cached(dataset, date_dir)
    baseline = _metric_baseline_stats_cached(dataset, date_dir)
    comp_s = str(component or "").strip()
    if metric_long.empty or baseline.empty or not comp_s:
        return {}
    win = metric_long[
        (metric_long["component"] == comp_s) & (metric_long["ts_s"] >= int(start_s)) & (metric_long["ts_s"] <= int(end_s))
    ].copy()
    if win.empty:
        return {}
    win = win.merge(baseline, on=["component", "metric"], how="left").dropna(subset=["median", "iqr"]).copy()
    eps = 1e-6
    # Only the "elevated above normal" direction counts -- these are all
    # resource-usage reasons ("high X usage"), a value dropping below its
    # median is not evidence for any of them.
    win["z"] = ((win["value"] - win["median"]) / (win["iqr"].abs() + eps)).clip(lower=0.0, upper=20.0)
    by = win.groupby("metric", as_index=False)["z"].mean()
    return {str(r["metric"]): float(r["z"]) for _, r in by.iterrows()}


@lru_cache(maxsize=512)
def _component_reason_scores_cached(dataset: str, date_dir: str, component: str, start_s: int, end_s: int) -> Dict[str, Dict[str, Any]]:
    """For every known reason label, how strongly does this component's
    metric deviation pattern support it: mean z-score across that reason's
    matched metrics (0.0 if none matched), plus the matched metrics
    themselves sorted by deviation magnitude. This is the per-component
    scoreboard that reason_rank_on_component / gap_to_second_reason /
    competing_reasons are all read from.
    """
    deviation_map = _component_all_metric_deviation_cached(dataset, date_dir, component, start_s, end_s)
    out: Dict[str, Dict[str, Any]] = {}
    for r in _reason_vocab_for_dataset(dataset):
        matched = [(m, v) for m, v in deviation_map.items() if _metric_matches_reason(m, r)]
        matched.sort(key=lambda mv: mv[1], reverse=True)
        score = (sum(v for _m, v in matched) / len(matched)) if matched else 0.0
        out[r] = {"score": float(score), "matched": matched}
    return out


def _reason_support_label(rank: int, gap: float, percentile: Optional[float], own_score: float) -> str:
    if own_score <= 0.0:
        return "not_supported"
    if rank == 1 and gap >= _REASON_STRONG_GAP and (percentile is None or percentile >= _REASON_STRONG_PERCENTILE):
        return "strong"
    if rank <= 2:
        return "weak"
    return "not_supported"


def build_reason_evidence(
    *,
    dataset: str,
    instruction: str,
    component: str,
    reason: str,
    all_candidates: Sequence[Tuple[str, str]] = (),
    top_matched_metrics: int = 5,
) -> ReasonEvidence:
    """Test whether `reason` is the metric-group that most strongly and
    distinctly explains `component`'s anomalous behavior in the instruction's
    time window -- as opposed to the old check of "did any of this reason's
    keywords appear in whatever 8 metrics happened to deviate most overall".

    `all_candidates` (optional list of (component, reason) pairs, e.g. the
    full candidate pool) lets this also report `cross_candidate_percentile`:
    where this specific (component, reason) pair's own deviation score falls
    among every other candidate's own claimed (component, reason) deviation
    score. That is what actually answers "is Redis02's memory anomaly more
    extreme than IG01's disk anomaly", not just "does Redis02 have some
    memory-looking metric".
    """
    date_dir, start, end = parse_time_window_generic(instruction)
    date_dir = resolve_date_dir(dataset, date_dir)
    start_s = int(start.tz_convert("UTC").timestamp())
    end_s = int(end.tz_convert("UTC").timestamp())

    comp_s = str(component or "").strip()
    reason_s = str(reason or "").strip()
    reason_known = reason_s in REASON_METRIC_GROUPS

    scores = _component_reason_scores_cached(str(dataset or ""), date_dir, comp_s, start_s, end_s)
    own = scores.get(reason_s, {"score": 0.0, "matched": []})
    own_score = float(own.get("score", 0.0) or 0.0)
    own_matched = list(own.get("matched", []) or [])[: max(1, int(top_matched_metrics))]

    ranked = sorted(scores.items(), key=lambda kv: float(kv[1].get("score", 0.0) or 0.0), reverse=True)
    rank = next((i for i, (r, _info) in enumerate(ranked, start=1) if r == reason_s), len(ranked) + 1)

    competing = [
        {"reason": r, "score": round(float(info.get("score", 0.0) or 0.0), 4)}
        for r, info in ranked
        if r != reason_s
    ][:3]
    second_best_score = float(competing[0]["score"]) if competing else 0.0
    gap = round(own_score - second_best_score, 4)

    cross_candidate_percentile: Optional[float] = None
    if all_candidates:
        peer_scores: List[float] = []
        for cand_comp, cand_reason in all_candidates:
            cand_comp_s = str(cand_comp or "").strip()
            if not cand_comp_s:
                continue
            try:
                peer_map = _component_reason_scores_cached(str(dataset or ""), date_dir, cand_comp_s, start_s, end_s)
            except Exception:
                continue
            peer_scores.append(float((peer_map.get(str(cand_reason or "").strip(), {}) or {}).get("score", 0.0) or 0.0))
        if peer_scores:
            at_or_below = sum(1 for s in peer_scores if s <= own_score)
            cross_candidate_percentile = round(at_or_below / float(len(peer_scores)), 4)

    support = _reason_support_label(rank, gap, cross_candidate_percentile, own_score)

    return ReasonEvidence(
        component=comp_s,
        reason=reason_s,
        reason_known=bool(reason_known),
        support=support,
        reason_deviation=round(own_score, 4),
        reason_rank_on_component=int(rank),
        gap_to_second_reason=float(gap),
        cross_candidate_percentile=cross_candidate_percentile,
        matched_metrics=[{"metric": m, "breach_ratio": round(float(v), 4)} for m, v in own_matched],
        competing_reasons=competing,
    )


def build_candidate_evidence(
    *,
    dataset: str,
    instruction: str,
    component: str,
    reason: str,
    score: float,
    score_gap_vs_next: float,
    all_candidate_components: Sequence[str],
    top_k_metrics: int = 8,
    topology_max_rows: Optional[int] = None,
    enable_trace: bool = True,
) -> CandidateEvidence:
    date_dir, start, end = parse_time_window_generic(instruction)
    date_dir = resolve_date_dir(dataset, date_dir)
    start_s = int(start.tz_convert("UTC").timestamp())
    end_s = int(end.tz_convert("UTC").timestamp())

    metric_long = _metric_long_cached(dataset, date_dir)
    p95_df = _metric_p95_map_cached(dataset, date_dir)
    trace_minute_p95 = _trace_minute_p95_cached(dataset, date_dir) if bool(enable_trace) else pd.DataFrame()

    comp_s = str(component or "").strip()
    if comp_s:
        metrics = _pick_metrics_for_component(metric_long, p95_df, comp_s, start_s, end_s, top_k=int(top_k_metrics))
        metric_timeline, first_anomaly_s = _metric_minute_summary(metric_long, p95_df, comp_s, metrics, start_s, end_s, limit_per_metric=6)
        trace_timeline = _trace_minute_summary(trace_minute_p95, comp_s, start_s, end_s, limit=10)
    else:
        metrics = _pick_metrics_global(metric_long, p95_df, start_s, end_s, top_k=int(top_k_metrics))
        metric_timeline, first_anomaly_s = _metric_minute_summary_global(metric_long, p95_df, metrics, start_s, end_s, limit_per_metric=6)
        trace_timeline = _trace_minute_summary(trace_minute_p95, "__all__", start_s, end_s, limit=10)

    edges = []
    if bool(enable_trace):
        try:
            edges = extract_trace_topology_edges(
                dataset=dataset,
                date_dir=date_dir,
                start_s=start_s,
                end_s=end_s,
                max_rows=(int(topology_max_rows) if topology_max_rows and int(topology_max_rows) > 0 else None),
            )
        except Exception:
            edges = []

    outgoing = [e for e in edges if str(getattr(e, "src", "")) == str(component)]
    incoming = [e for e in edges if str(getattr(e, "dst", "")) == str(component)]
    outgoing = sorted(outgoing, key=lambda e: int(getattr(e, "support_count", 0) or 0), reverse=True)[:10]
    incoming = sorted(incoming, key=lambda e: int(getattr(e, "support_count", 0) or 0), reverse=True)[:10]
    upstream_of = [str(getattr(e, "dst", "")) for e in outgoing if str(getattr(e, "dst", "")).strip()]
    impacted_by = [str(getattr(e, "src", "")) for e in incoming if str(getattr(e, "src", "")).strip()]

    other_first = []
    for c in all_candidate_components:
        if str(c) == str(component):
            continue
        try:
            metrics_c = _pick_metrics_for_component(metric_long, p95_df, str(c), start_s, end_s, top_k=5)
            _, first_c = _metric_minute_summary(metric_long, p95_df, str(c), metrics_c, start_s, end_s, limit_per_metric=3)
            if first_c is not None:
                other_first.append(int(first_c))
        except Exception:
            continue
    first_among = False
    if first_anomaly_s is not None and other_first:
        first_among = all(int(first_anomaly_s) <= int(x) for x in other_first)

    evidence_for: List[str] = []
    evidence_against: List[str] = []

    if first_anomaly_s is not None:
        evidence_for.append(f"Metric first breach among selected metrics at {_minute_label_local(int(first_anomaly_s))}.")
    if first_among:
        evidence_for.append("Metric first breach is earliest among component candidates.")
    if upstream_of:
        evidence_for.append(f"Trace topology shows propagation candidates from {component} to {', '.join(upstream_of[:5])}.")
    if impacted_by:
        evidence_against.append(f"Trace topology shows {component} is impacted by {', '.join(impacted_by[:5])}.")
    if not metric_timeline:
        evidence_against.append("No metric breaches found for this component within the window.")
    if not trace_timeline:
        evidence_against.append("No trace p95 evidence found for this component within the window.")

    reason_s = str(reason or "").strip()
    if reason_s and metrics:
        tokens = _REASON_TOKENS.get(reason_s, [])
        if tokens:
            blob = " ".join([str(m).lower() for m in metrics])
            hit = sum(1 for t in tokens if t and t.lower() in blob)
            if hit > 0:
                evidence_for.append(f'Reason "{reason_s}" matches anomalous metric keywords: hits={hit}.')
            else:
                evidence_against.append(f'Reason "{reason_s}" does not match top anomalous metric keywords in this window.')

    causal_evidence = {
        "first_anomaly_among_candidates": bool(first_among),
        "upstream_of": upstream_of[:8],
        "impacted_by": impacted_by[:8],
        "window_start_local": start.strftime("%Y-%m-%d %H:%M:%S"),
        "window_end_local": end.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return CandidateEvidence(
        component=str(component or ""),
        reason=str(reason or ""),
        metric_timeline=list(metric_timeline),
        trace_timeline=list(trace_timeline),
        causal_evidence=causal_evidence,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        score=float(score or 0.0),
        score_gap_vs_next=float(score_gap_vs_next or 0.0),
        first_anomaly_s=int(first_anomaly_s) if first_anomaly_s is not None else None,
    )


def _metric_at_minute(
    *,
    metric_long: pd.DataFrame,
    p95_df: pd.DataFrame,
    component: str,
    metrics: Sequence[str],
    minute_s: int,
    first_breach_by_metric: Mapping[str, Optional[int]],
) -> List[Dict[str, Any]]:
    if metric_long.empty or not metrics:
        return []
    if str(component) == "__all__":
        sub = metric_long[
            (metric_long["metric"].isin([str(m) for m in metrics]))
            & (metric_long["ts_s"] >= int(minute_s))
            & (metric_long["ts_s"] < int(minute_s) + 60)
        ].copy()
    else:
        sub = metric_long[
            (metric_long["component"] == str(component))
            & (metric_long["metric"].isin([str(m) for m in metrics]))
            & (metric_long["ts_s"] >= int(minute_s))
            & (metric_long["ts_s"] < int(minute_s) + 60)
        ].copy()
    if sub.empty:
        return []
    out = []
    for m, g in sub.groupby("metric"):
        m = str(m)
        v = _safe_float(g["value"].mean(), 0.0)
        baseline = _global_p95_lookup(p95_df, m) if str(component) == "__all__" else _p95_lookup(p95_df, component, m)
        ratio = (v / baseline) if baseline > 0 else None
        fb = first_breach_by_metric.get(m)
        out.append(
            {
                "metric": m,
                "value": round(float(v), 4),
                "baseline": round(float(baseline), 4),
                "ratio": round(float(ratio), 4) if ratio is not None else None,
                "is_first_breach": bool(fb is not None and int(fb) == int(minute_s) and baseline > 0 and v > baseline),
            }
        )
    out.sort(key=lambda x: (_safe_float(x.get("ratio"), 0.0), x.get("metric", "")), reverse=True)
    return out


def _trace_at_minute(trace_minute_p95: pd.DataFrame, component: str, minute_s: int, baseline: float) -> Dict[str, Any]:
    if trace_minute_p95.empty:
        return {}
    sub = trace_minute_p95[
        (trace_minute_p95["component"] == str(component)) & (trace_minute_p95["minute"] == int(minute_s))
    ]
    if sub.empty:
        return {}
    p95 = _safe_float(sub.iloc[0]["p95"], 0.0)
    ratio = (p95 / baseline) if baseline > 0 else None
    return {
        "p95": round(float(p95), 4),
        "baseline": round(float(baseline), 4),
        "ratio": round(float(ratio), 4) if ratio is not None else None,
    }


def build_time_evidence(
    *,
    dataset: str,
    instruction: str,
    time_candidate: TimeCandidate,
    component: str = "",
    reason: str = "",
    metric_focus_metrics: Optional[Sequence[str]] = None,
    metric_focus_top_k: int = 8,
    enable_trace: bool = True,
) -> TimeEvidence:
    date_dir, start, end = parse_time_window_generic(instruction)
    date_dir = resolve_date_dir(dataset, date_dir)
    start_s = int(start.tz_convert("UTC").timestamp())
    end_s = int(end.tz_convert("UTC").timestamp())

    metric_long = _metric_long_cached(dataset, date_dir)
    p95_df = _metric_p95_map_cached(dataset, date_dir)
    trace_minute_p95 = _trace_minute_p95_cached(dataset, date_dir) if bool(enable_trace) else pd.DataFrame()

    t_s = _to_epoch_s_local(str(time_candidate.datetime))
    if t_s is None:
        minute_s = _minute_bucket_s(int(start_s))
    else:
        minute_s = _minute_bucket_s(int(t_s))
    minute_s = min(max(int(minute_s), _minute_bucket_s(int(start_s))), _minute_bucket_s(int(end_s)))

    focus_component = str(component or "")
    if not focus_component:
        focus_component = "__all__"

    metrics = [str(m) for m in (metric_focus_metrics or []) if str(m).strip()]
    if not metrics and focus_component != "__all__":
        metrics = _pick_metrics_for_component(metric_long, p95_df, focus_component, start_s, end_s, top_k=int(metric_focus_top_k))
    if not metrics and focus_component == "__all__":
        metrics = _pick_metrics_global(metric_long, p95_df, start_s, end_s, top_k=int(metric_focus_top_k))

    first_breach_by_metric: Dict[str, Optional[int]] = {}
    if metrics and (not metric_long.empty) and focus_component != "__all__":
        sub = metric_long[
            (metric_long["component"] == focus_component)
            & (metric_long["metric"].isin(metrics))
            & (metric_long["ts_s"] >= int(start_s))
            & (metric_long["ts_s"] <= int(end_s))
        ].copy()
        if not sub.empty:
            sub["minute"] = (sub["ts_s"].astype(int) // 60) * 60
            agg = sub.groupby(["metric", "minute"], as_index=False)["value"].mean()
            agg["baseline"] = agg.apply(lambda r: _p95_lookup(p95_df, focus_component, str(r["metric"])), axis=1)
            agg["is_breach"] = agg.apply(lambda r: _safe_float(r["value"]) > _safe_float(r["baseline"]), axis=1)
            for m, g in agg.sort_values(["metric", "minute"]).groupby("metric"):
                hits = g[g["is_breach"]]
                first_breach_by_metric[str(m)] = int(hits.iloc[0]["minute"]) if not hits.empty else None
    if metrics and (not metric_long.empty) and focus_component == "__all__":
        sub = metric_long[
            (metric_long["metric"].isin(metrics))
            & (metric_long["ts_s"] >= int(start_s))
            & (metric_long["ts_s"] <= int(end_s))
        ].copy()
        if not sub.empty:
            sub["minute"] = (sub["ts_s"].astype(int) // 60) * 60
            agg = sub.groupby(["metric", "minute"], as_index=False)["value"].mean()
            agg["baseline"] = agg["metric"].astype(str).apply(lambda m: _global_p95_lookup(p95_df, m))
            agg["is_breach"] = agg.apply(lambda r: _safe_float(r["value"]) > _safe_float(r["baseline"]), axis=1)
            for m, g in agg.sort_values(["metric", "minute"]).groupby("metric"):
                hits = g[g["is_breach"]]
                first_breach_by_metric[str(m)] = int(hits.iloc[0]["minute"]) if not hits.empty else None

    pre_baseline = 0.0
    if not trace_minute_p95.empty:
        start_min = _minute_bucket_s(int(start_s))
        end_min = _minute_bucket_s(int(end_s))
        tr = trace_minute_p95[
            (trace_minute_p95["component"] == focus_component)
            & (trace_minute_p95["minute"] >= int(start_min))
            & (trace_minute_p95["minute"] <= int(end_min))
        ].copy()
        if not tr.empty:
            tr = tr.sort_values("minute")
            pre_n = max(5, int(len(tr) * 0.2))
            pre_baseline = _safe_float(tr.iloc[:pre_n]["p95"].median(), 0.0)

    metric_at = _metric_at_minute(
        metric_long=metric_long,
        p95_df=p95_df,
        component=focus_component,
        metrics=metrics,
        minute_s=minute_s,
        first_breach_by_metric=first_breach_by_metric,
    )
    metric_before = _metric_at_minute(
        metric_long=metric_long,
        p95_df=p95_df,
        component=focus_component,
        metrics=metrics,
        minute_s=max(int(start_s), int(minute_s) - 60),
        first_breach_by_metric=first_breach_by_metric,
    )
    metric_after = _metric_at_minute(
        metric_long=metric_long,
        p95_df=p95_df,
        component=focus_component,
        metrics=metrics,
        minute_s=min(int(end_s) - 60, int(minute_s) + 60),
        first_breach_by_metric=first_breach_by_metric,
    )
    trace_at = _trace_at_minute(trace_minute_p95, focus_component, minute_s, pre_baseline)

    evidence_for: List[str] = []
    evidence_against: List[str] = []

    first_breaches = [m for m in metric_at if bool(m.get("is_first_breach"))]
    if first_breaches:
        for m in first_breaches[:2]:
            evidence_for.append(
                f"{m.get('metric')} first breached here: {m.get('value')} vs baseline {m.get('baseline')} (ratio={m.get('ratio')})."
            )
    severe = [m for m in metric_at if _safe_float(m.get("ratio"), 0.0) >= 2.0]
    if severe:
        evidence_for.append(f"Metric severe anomaly at this minute: {', '.join([str(m.get('metric')) for m in severe[:3]])}.")

    tr_ratio = trace_at.get("ratio")
    if tr_ratio is not None:
        if _safe_float(tr_ratio) < 1.2:
            evidence_for.append("Trace not yet impacted at this minute.")
        elif _safe_float(tr_ratio) >= 2.0:
            evidence_against.append("Trace already severely impacted at this minute; onset may be earlier.")

    if not metric_at and focus_component != "__all__":
        evidence_against.append("No component metric values at this minute.")

    return TimeEvidence(
        datetime=pd.to_datetime(int(minute_s), unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S"),
        source=str(time_candidate.source or ""),
        metric_at_this_time=metric_at,
        trace_at_this_time=trace_at,
        metric_1min_before=metric_before,
        metric_1min_after=metric_after,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        score=float(time_candidate.score or 0.0),
    )


def compute_uncertainty(
    *,
    candidates: Sequence[CandidateEvidence],
    time_candidates: Sequence[TimeEvidence],
    task_fields: Sequence[str],
    score_gap_threshold: float = 0.15,
    top1_score_floor: float = 0.30,
    time_gap_threshold: float = 0.20,
    conflict_seconds: int = 300,
) -> Dict[str, Any]:
    result = {"need_llm": False, "reasons": [], "focus": []}
    cand_list = [c for c in candidates or [] if isinstance(c, CandidateEvidence)]
    time_list = [t for t in time_candidates or [] if isinstance(t, TimeEvidence)]
    fields = [str(x) for x in (task_fields or []) if str(x).strip()]

    if len(cand_list) >= 2:
        score_gap = float(cand_list[0].score) - float(cand_list[1].score)
        if score_gap < float(score_gap_threshold):
            result["need_llm"] = True
            result["reasons"].append(f"Top2 score gap too small: {score_gap:.3f}")
            if "component" in fields or "reason" in fields:
                result["focus"].append("component_reason")

        if float(cand_list[0].score) < float(top1_score_floor):
            result["need_llm"] = True
            result["reasons"].append(f"Top1 score too low: {float(cand_list[0].score):.3f}")
            if "component" in fields or "reason" in fields:
                result["focus"].append("component_reason")

    if "time" in fields and len(time_list) >= 2:
        time_gap = float(time_list[0].score) - float(time_list[1].score)
        if time_gap < float(time_gap_threshold):
            result["need_llm"] = True
            result["reasons"].append(f"Time candidates too close: gap={time_gap:.3f}")
            result["focus"].append("time")

    if "time" in fields and cand_list and time_list:
        comp_first = cand_list[0].first_anomaly_s
        trace_onset_s = None
        for t in time_list:
            if str(t.source).startswith("trace_") and ("onset" in str(t.source) or "stable" in str(t.source)):
                trace_onset_s = _to_epoch_s_local(str(t.datetime))
                break
        if comp_first is not None and trace_onset_s is not None:
            lag = abs(int(comp_first) - int(trace_onset_s))
            if int(lag) > int(conflict_seconds):
                result["need_llm"] = True
                result["reasons"].append(f"Component metric onset and trace onset conflict: {lag}s gap")
                result["focus"].append("time")

    result["focus"] = list(dict.fromkeys(result["focus"]))
    return result


def build_candidate_pairs(
    *,
    component_scores: Mapping[str, float],
    reason_scores: Optional[Mapping[str, float]],
    component_candidates: Sequence[str],
    reason_candidates: Sequence[str],
    need_component: bool,
    need_reason: bool,
    top_pairs: int = 3,
) -> List[Tuple[str, str, float]]:
    comps = [str(c) for c in (component_candidates or []) if str(c).strip()]
    reasons = [str(r) for r in (reason_candidates or []) if str(r).strip()]

    cs = {str(k): _safe_float(v) for k, v in (component_scores or {}).items()}
    rs = {str(k): _safe_float(v) for k, v in (reason_scores or {}).items()} if isinstance(reason_scores, Mapping) else {}

    max_c = max([abs(v) for v in cs.values()] + [0.0])
    max_r = max([abs(v) for v in rs.values()] + [0.0])

    def cn(c: str) -> float:
        if not need_component:
            return 0.0
        if max_c <= 1e-12:
            return 0.0
        return max(0.0, _safe_float(cs.get(c, 0.0)) / max_c)

    def rn(r: str) -> float:
        if not need_reason:
            return 0.0
        if max_r <= 1e-12:
            return 0.0
        return max(0.0, _safe_float(rs.get(r, 0.0)) / max_r)

    if not need_component:
        comps = [""]
    if not need_reason:
        reasons = [""]
    if not comps:
        comps = [""]
    if not reasons:
        reasons = [""]

    pairs: List[Tuple[str, str, float]] = []
    for c in comps[: max(1, min(len(comps), 8))]:
        for r in reasons[: max(1, min(len(reasons), 8))]:
            score = 0.0
            if need_component and need_reason:
                score = 0.55 * cn(c) + 0.45 * rn(r)
            elif need_component:
                score = cn(c)
            elif need_reason:
                score = rn(r)
            pairs.append((str(c), str(r), float(score)))

    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    return pairs[: max(1, int(top_pairs))]
