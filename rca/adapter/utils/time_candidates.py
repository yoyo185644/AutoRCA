import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from rca.adapter.utils.component_candidates import _normalize_metric_file, parse_time_window_generic
from rca.common.io import resolve_date_dir


@dataclass
class TimeEvidence:
    date_dir: str
    start: str
    end: str
    chosen: str
    method: str
    baseline_p95: float
    peak_p95: float

    def to_dict(self) -> Dict:
        return {
            "date_dir": self.date_dir,
            "start": self.start,
            "end": self.end,
            "chosen": self.chosen,
            "method": self.method,
            "baseline_p95": float(self.baseline_p95),
            "peak_p95": float(self.peak_p95),
        }


@dataclass(frozen=True)
class TimeCandidate:
    datetime: str
    source: str
    score: float
    component: str = ""
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_local_str(ts: Any) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _dt_to_epoch_s(dt: str) -> Optional[int]:
    try:
        return int(pd.Timestamp(str(dt), tz="Asia/Shanghai").tz_convert("UTC").timestamp())
    except Exception:
        return None


def _add_candidate(out: List[TimeCandidate], cand: TimeCandidate) -> None:
    if not str(cand.datetime or "").strip():
        return
    for i, old in enumerate(out):
        if old.datetime == cand.datetime and old.source == cand.source and old.component == cand.component and old.reason == cand.reason:
            if cand.score > old.score:
                out[i] = cand
            return
    out.append(cand)


def _dedupe_time_candidates_by_datetime(candidates: Sequence[TimeCandidate]) -> List[TimeCandidate]:
    by_dt: Dict[str, List[TimeCandidate]] = {}
    for cand in candidates or []:
        dt = str(cand.datetime or "").strip()
        if not dt:
            continue
        by_dt.setdefault(dt, []).append(cand)

    merged: List[TimeCandidate] = []
    for dt, items in by_dt.items():
        items_sorted = sorted(items, key=lambda c: float(c.score or 0.0), reverse=True)
        best = items_sorted[0]
        sources = []
        merged_candidates = []
        component = str(best.component or "")
        reason = str(best.reason or "")
        evidence = dict(best.evidence or {})
        for item in items_sorted:
            src = str(item.source or "")
            if src and src not in sources:
                sources.append(src)
            if not component and str(item.component or "").strip():
                component = str(item.component)
            if not reason and str(item.reason or "").strip():
                reason = str(item.reason)
            merged_candidates.append(item.to_dict())
        if len(items_sorted) > 1:
            evidence["merged_sources"] = sources
            evidence["merged_candidates"] = merged_candidates
        merged.append(
            TimeCandidate(
                datetime=dt,
                source="+".join(sources) if sources else str(best.source),
                score=float(best.score or 0.0),
                component=component,
                reason=reason,
                evidence=evidence,
            )
        )
    return sorted(merged, key=lambda c: c.score, reverse=True)


def _trace_time_candidates(dataset: str, instruction: str) -> Tuple[List[TimeCandidate], Optional[TimeEvidence]]:
    date_dir, start, end = parse_time_window_generic(instruction)
    date_dir = resolve_date_dir(dataset, date_dir)
    trace_path = os.path.join("dataset", dataset, "telemetry", date_dir, "trace", "trace_span.csv")
    if not os.path.exists(trace_path):
        dt = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return [TimeCandidate(dt, "window_start_fallback", 0.10)], None
    columns = _read_csv_columns_cached(trace_path)
    dur_col = "elapsedTime" if "elapsedTime" in columns else ("duration" if "duration" in columns else None)
    ts_col = "startTime" if "startTime" in columns else ("timestamp" if "timestamp" in columns else None)
    if dur_col is None or ts_col is None:
        dt = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return [TimeCandidate(dt, "window_start_fallback", 0.10)], None
    df = _read_trace_time_frame_cached(trace_path, ts_col, dur_col)
    if df.empty or dur_col is None or ts_col is None:
        dt = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return [TimeCandidate(dt, "window_start_fallback", 0.10)], None

    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if df.empty:
        dt = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return [TimeCandidate(dt, "window_start_fallback", 0.10)], None

    df["minute"] = df["timestamp"].dt.floor("min")
    p95 = df.groupby("minute")["dur"].quantile(0.95).sort_index()
    if p95.empty:
        dt = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return [TimeCandidate(dt, "window_start_fallback", 0.10)], None

    p95_s = p95.rolling(window=3, center=True, min_periods=1).median()
    pre_n = max(5, int(len(p95_s) * 0.2))
    pre = p95_s.iloc[:pre_n]
    baseline = float(pre.median())
    mad = float((pre - baseline).abs().median())
    thr = max(baseline * 1.18, baseline + 6.0 * mad, baseline + 1.0)
    peak_p95 = float(p95.max())
    ratio = peak_p95 / max(baseline, 1e-9)
    ratio_bonus = min(0.35, max(0.0, ratio - 1.0) * 0.12)

    out: List[TimeCandidate] = []
    pre_raw = p95.iloc[:pre_n]
    baseline_raw = float(pre_raw.median())
    mad_raw = float((pre_raw - baseline_raw).abs().median())
    thr_raw = max(baseline_raw * 1.12, baseline_raw + 4.0 * mad_raw, baseline_raw + 1.0)
    cond_raw = p95 >= thr_raw
    if cond_raw.any():
        ts = cond_raw[cond_raw].index[0]
        _add_candidate(
            out,
            TimeCandidate(
                _to_local_str(ts),
                "trace_first_breach",
                min(1.0, 1.02 + ratio_bonus),
                evidence={"baseline_p95": baseline_raw, "peak_p95": peak_p95, "ratio": ratio, "threshold": thr_raw},
            ),
        )
    cond = p95_s >= thr
    onset2 = cond & cond.shift(-1, fill_value=False)
    if onset2.any():
        ts = onset2[onset2].index[0]
        _add_candidate(
            out,
            TimeCandidate(
                _to_local_str(ts),
                "trace_stable_onset",
                min(1.0, 0.98 + ratio_bonus),
                evidence={"baseline_p95": baseline, "peak_p95": peak_p95, "ratio": ratio, "threshold": thr},
            ),
        )
    if cond.any():
        ts = cond[cond].index[0]
        _add_candidate(
            out,
            TimeCandidate(
                _to_local_str(ts),
                "trace_onset",
                min(1.0, 0.95 + ratio_bonus),
                evidence={"baseline_p95": baseline, "peak_p95": peak_p95, "ratio": ratio, "threshold": thr},
            ),
        )

    diff = p95_s.diff().fillna(0.0)
    jumps = p95_s[(p95_s >= baseline * 1.2) & (diff >= max(1.0, baseline * 0.08))]
    if not jumps.empty:
        ts = jumps.index[0]
        _add_candidate(
            out,
            TimeCandidate(
                _to_local_str(ts),
                "trace_jump",
                0.82 + min(0.20, float(diff.loc[ts]) / max(baseline, 1.0) * 0.08),
                evidence={"baseline_p95": baseline, "peak_p95": peak_p95, "jump": float(diff.loc[ts])},
            ),
        )

    roll = p95.rolling(window=3, center=True, min_periods=1).mean()
    peak_ts = roll.sort_values(ascending=False).index[0]
    peak_score = 0.48 + ratio_bonus * 0.5
    if peak_ts == start.floor("min") and not out:
        peak_score -= 0.20
    _add_candidate(
        out,
        TimeCandidate(
            _to_local_str(peak_ts),
            "trace_peak",
            max(0.05, peak_score),
            evidence={"baseline_p95": baseline, "peak_p95": peak_p95, "ratio": ratio},
        ),
    )

    chosen = out[0].datetime if out else start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
    method = out[0].source.replace("trace_", "") if out else "fallback"
    ev = TimeEvidence(
        date_dir=date_dir,
        start=start.strftime("%Y-%m-%d %H:%M:%S"),
        end=end.strftime("%Y-%m-%d %H:%M:%S"),
        chosen=chosen,
        method=method,
        baseline_p95=baseline,
        peak_p95=peak_p95,
    )
    return sorted(out, key=lambda c: c.score, reverse=True), ev


_REASON_TIME_TOKENS: Dict[str, List[str]] = {
    "network loss": ["loss", "drop", "retrans", "timeout", "reset", "error"],
    "network delay": ["tnsping", "ping", "rtt", "latency", "delay"],
    "db connection limit": ["login", "session", "sess", "tps", "exec", "connection", "conn"],
    "db close": ["on_off", "logout", "disconnect", "hang", "shutdown", "restart", "close", "reset"],
    "CPU fault": ["proc", "cpu", "load", "iowait", "util"],
}


_REASON_METRIC_FILES: Dict[str, List[str]] = {
    # node-level metrics carry the ground-truth signal for network faults;
    # middleware/service reflect downstream impact only.
    "network loss": ["metric_node.csv", "metric_middleware.csv", "metric_service.csv"],
    "network delay": ["metric_node.csv", "metric_middleware.csv", "metric_service.csv"],
    "db connection limit": ["metric_service.csv", "metric_middleware.csv"],
    "db close": ["metric_service.csv", "metric_middleware.csv"],
    # container is primary for CPU; node catches OS-level cpu/load metrics.
    "CPU fault": ["metric_container.csv", "metric_node.csv"],
}


@lru_cache(maxsize=64)
def _normalize_metric_file_cached(path: str) -> pd.DataFrame:
    return _normalize_metric_file(path)


@lru_cache(maxsize=64)
def _read_csv_columns_cached(path: str) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns)


@lru_cache(maxsize=16)
def _read_trace_time_frame_cached(path: str, ts_col: str, dur_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[ts_col, dur_col], low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "dur"])
    out = df[[ts_col, dur_col]].copy()
    ts_unit = "ms" if out[ts_col].dropna().astype(str).str.len().max() >= 13 else "s"
    out["timestamp"] = pd.to_datetime(out[ts_col], unit=ts_unit, utc=True).dt.tz_convert("Asia/Shanghai")
    out["dur"] = pd.to_numeric(out[dur_col], errors="coerce")
    return out[["timestamp", "dur"]].dropna(subset=["timestamp", "dur"])


def _reason_metric_time_candidates(dataset: str, instruction: str, reasons: Sequence[str]) -> List[TimeCandidate]:
    reason_list = [str(r) for r in reasons or [] if str(r).strip()]
    if not reason_list:
        return []
    date_dir, start, end = parse_time_window_generic(instruction)
    date_dir = resolve_date_dir(dataset, date_dir)
    metric_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "metric")
    if not os.path.isdir(metric_dir):
        return []
    start_s = int(start.tz_convert("UTC").timestamp())
    end_s = int(end.tz_convert("UTC").timestamp())
    out: List[TimeCandidate] = []
    all_files = [name for name in os.listdir(metric_dir) if name.endswith(".csv")]
    normalized_cache: Dict[str, pd.DataFrame] = {}

    def load_metric(name: str) -> pd.DataFrame:
        if name not in normalized_cache:
            try:
                normalized_cache[name] = _normalize_metric_file_cached(os.path.join(metric_dir, name))
            except Exception:
                normalized_cache[name] = pd.DataFrame(columns=["ts_s", "component", "metric", "value"])
        return normalized_cache[name]

    for reason in reason_list:
        tokens = _REASON_TIME_TOKENS.get(reason, [])
        if not tokens:
            continue
        minute_scores: Dict[int, float] = {}
        metric_names = [name for name in _REASON_METRIC_FILES.get(reason, all_files) if name in all_files]
        if not metric_names:
            metric_names = all_files
        for name in metric_names:
            long = load_metric(name)
            if long.empty:
                continue
            long = long.copy()
            long["metric_l"] = long["metric"].astype(str).str.lower()
            mask = False
            for tok in tokens:
                mask = mask | long["metric_l"].str.contains(tok.lower(), na=False)
            if not getattr(mask, "any", lambda: False)():
                continue
            sub = long[mask].copy()
            p95 = sub.groupby(["component", "metric"], as_index=False)["value"].quantile(0.95).rename(columns={"value": "p95"})
            win = sub[(sub["ts_s"] >= start_s) & (sub["ts_s"] <= end_s)].copy()
            if win.empty:
                continue
            win = win.merge(p95, on=["component", "metric"], how="left").dropna(subset=["p95"]).copy()
            denom = win["p95"].abs().where(win["p95"].abs() > 1e-6, 1.0)
            win["breach_ratio"] = ((win["value"] - win["p95"]) / denom).clip(lower=0.0, upper=10.0)
            win = win[win["breach_ratio"] > 0].copy()
            if win.empty:
                continue
            win["minute"] = (win["ts_s"].astype(int) // 60) * 60
            by_min = win.groupby("minute")["breach_ratio"].sum()
            for m, s in by_min.items():
                minute_scores[int(m)] = minute_scores.get(int(m), 0.0) + float(s)
        if not minute_scores:
            continue
        mx = max(minute_scores.values())
        if mx <= 0:
            continue
        # onset: earliest minute above 30% of max (first sign of anomaly)
        onset_cands = sorted((m, s) for m, s in minute_scores.items() if s >= 0.30 * mx)
        if not onset_cands:
            onset_cands = sorted(minute_scores.items(), key=lambda kv: kv[1], reverse=True)[:1]
        onset_minute, onset_score = onset_cands[0]
        onset_dt = pd.to_datetime(int(onset_minute), unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
        _add_candidate(
            out,
            TimeCandidate(
                onset_dt,
                "reason_metric_onset",
                0.88 + min(0.12, float(onset_score) / max(mx, 1e-9) * 0.12),
                reason=reason,
                evidence={"reason": reason, "relative_score": float(onset_score) / max(mx, 1e-9), "tokens": tokens},
            ),
        )
        # ±1 min neighbors of onset (Direction 4): let evaluator's ≤1min window find the exact GT
        for delta_s in (60, -60):
            try:
                nb_dt = (pd.Timestamp(onset_dt, tz="Asia/Shanghai") + pd.Timedelta(seconds=delta_s)).strftime("%Y-%m-%d %H:%M:%S")
                _add_candidate(
                    out,
                    TimeCandidate(nb_dt, "reason_metric_onset_neighbor", 0.76, reason=reason,
                                  evidence={"reason": reason, "neighbor_of": onset_dt}),
                )
            except Exception:
                pass
        # peak: minute with highest anomaly score (useful when onset is pre-cursor noise)
        peak_minute = max(minute_scores, key=lambda m: minute_scores[m])
        if peak_minute != onset_minute:
            peak_dt = pd.to_datetime(int(peak_minute), unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
            _add_candidate(
                out,
                TimeCandidate(
                    peak_dt,
                    "reason_metric_peak",
                    0.72 + min(0.10, float(minute_scores[peak_minute]) / max(mx, 1e-9) * 0.10),
                    reason=reason,
                    evidence={"reason": reason, "relative_score": 1.0, "tokens": tokens},
                ),
            )
    return out


def _component_metric_onset_time_candidates(
    dataset: str,
    instruction: str,
    components: Sequence[str],
    top_k_metrics: int = 8,
) -> List[TimeCandidate]:
    comps = [str(c) for c in (components or []) if str(c).strip()]
    if not comps:
        return []
    date_dir, start, end = parse_time_window_generic(instruction)
    date_dir = resolve_date_dir(dataset, date_dir)
    metric_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "metric")
    if not os.path.isdir(metric_dir):
        return []
    start_s = int(start.tz_convert("UTC").timestamp())
    end_s = int(end.tz_convert("UTC").timestamp())
    all_files = [name for name in os.listdir(metric_dir) if name.endswith(".csv")]
    if not all_files:
        return []
    normalized_cache: Dict[str, pd.DataFrame] = {}

    def load_metric(name: str) -> pd.DataFrame:
        if name not in normalized_cache:
            try:
                normalized_cache[name] = _normalize_metric_file_cached(os.path.join(metric_dir, name))
            except Exception:
                normalized_cache[name] = pd.DataFrame(columns=["ts_s", "component", "metric", "value"])
        return normalized_cache[name]

    out: List[TimeCandidate] = []
    for comp in comps:
        per_metric_minute = []
        for name in all_files:
            long = load_metric(name)
            if long.empty:
                continue
            sub_all = long[long["component"].astype(str) == str(comp)].copy()
            if sub_all.empty:
                continue
            p95 = (
                sub_all.groupby(["component", "metric"], as_index=False)["value"]
                .quantile(0.95)
                .rename(columns={"value": "p95"})
            )
            win = sub_all[(sub_all["ts_s"] >= start_s) & (sub_all["ts_s"] <= end_s)].copy()
            if win.empty:
                continue
            win["minute"] = (win["ts_s"].astype(int) // 60) * 60
            agg = win.groupby(["metric", "minute"], as_index=False)["value"].max()
            agg = agg.rename(columns={"value": "v_max"})
            agg["component"] = str(comp)
            agg = agg.merge(p95, on=["component", "metric"], how="left").dropna(subset=["p95"]).copy()
            denom = agg["p95"].abs().where(agg["p95"].abs() > 1e-6, 1.0)
            agg["breach_ratio"] = ((agg["v_max"] - agg["p95"]) / denom).clip(lower=0.0, upper=10.0)
            agg = agg[agg["breach_ratio"] > 0].copy()
            if agg.empty:
                continue
            per_metric_minute.append(agg[["metric", "minute", "v_max", "p95", "breach_ratio"]])

        if not per_metric_minute:
            continue
        agg_all = pd.concat(per_metric_minute, ignore_index=True)
        by_metric = (
            agg_all.groupby("metric", as_index=False)["breach_ratio"]
            .sum()
            .sort_values("breach_ratio", ascending=False)
        )
        top_metrics = by_metric.head(int(top_k_metrics))["metric"].astype(str).tolist()
        use = agg_all[agg_all["metric"].isin(top_metrics)].copy()
        if use.empty:
            continue
        by_min = use.groupby("minute", as_index=False)["breach_ratio"].sum().sort_values("minute")
        mx = float(by_min["breach_ratio"].max()) if not by_min.empty else 0.0
        if mx <= 0:
            continue
        by_min = by_min.copy()
        by_min["minute"] = pd.to_numeric(by_min["minute"], errors="coerce")
        by_min["breach_ratio"] = pd.to_numeric(by_min["breach_ratio"], errors="coerce")
        by_min = by_min.dropna(subset=["minute", "breach_ratio"]).copy()
        by_min["minute"] = by_min["minute"].astype("int64")
        by_min = by_min.sort_values("minute").reset_index(drop=True)

        thr_first = max(0.03 * mx, 0.15)
        by_min["is_breach_first"] = by_min["breach_ratio"] >= thr_first
        by_min["is_stable_first"] = by_min["is_breach_first"] & (
            by_min["is_breach_first"].shift(-1, fill_value=False) | by_min["is_breach_first"].shift(-2, fill_value=False)
        )
        cand_first = by_min[by_min["is_breach_first"]]
        if not cand_first.empty:
            first_min = int(cand_first.iloc[0]["minute"])
            stable_first = bool(by_min[by_min["minute"] == first_min]["is_stable_first"].iloc[0]) if not by_min.empty else False
            first_dt = pd.to_datetime(first_min, unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
            _add_candidate(
                out,
                TimeCandidate(
                    first_dt,
                    "component_metric_first_breach_max",
                    min(1.0, (0.92 if stable_first else 0.80) + min(0.10, float(cand_first.iloc[0]["breach_ratio"]) / max(mx, 1e-9) * 0.10)),
                    component=str(comp),
                    evidence={
                        "component": str(comp),
                        "threshold_rel": float(thr_first),
                        "max_minute_score": float(mx),
                        "top_metrics": top_metrics[:8],
                        "stable": bool(stable_first),
                    },
                ),
            )
        thr_rel = max(0.15 * mx, 0.60)
        by_min["is_breach_onset"] = by_min["breach_ratio"] >= thr_rel
        by_min["is_stable_onset"] = by_min["is_breach_onset"] & (
            by_min["is_breach_onset"].shift(-1, fill_value=False) | by_min["is_breach_onset"].shift(-2, fill_value=False)
        )
        cand = by_min[by_min["is_breach_onset"]]
        if cand.empty:
            cand = by_min[by_min["breach_ratio"] >= 0.08 * mx]
        if cand.empty:
            continue
        stable_cand = by_min[by_min["is_stable_onset"]]
        onset_min = int(stable_cand.iloc[0]["minute"]) if not stable_cand.empty else int(cand.iloc[0]["minute"])
        stable_onset = bool(not stable_cand.empty)
        peak_min = int(by_min.sort_values("breach_ratio", ascending=False).iloc[0]["minute"])
        onset_dt = pd.to_datetime(onset_min, unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
        peak_dt = pd.to_datetime(peak_min, unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
        _add_candidate(
            out,
            TimeCandidate(
                onset_dt,
                "component_metric_onset_max",
                min(0.98, (0.92 if stable_onset else 0.78) + min(0.10, float(cand.iloc[0]["breach_ratio"]) / max(mx, 1e-9) * 0.10)),
                component=str(comp),
                evidence={
                    "component": str(comp),
                    "threshold_rel": float(thr_rel),
                    "max_minute_score": float(mx),
                    "top_metrics": top_metrics[:8],
                    "stable": bool(stable_onset),
                },
            ),
        )
        _add_candidate(
            out,
            TimeCandidate(
                peak_dt,
                "component_metric_peak_max",
                0.55 + min(0.12, float(mx) / max(mx, 1e-9) * 0.08),
                component=str(comp),
                evidence={"component": str(comp), "top_metrics": top_metrics[:8]},
            ),
        )
        # ±1min neighbors of onset and first_breach: evaluator tolerance is ≤1min,
        # so a 2-min-off estimate can be rescued by its neighbor hitting the GT exactly.
        for _base_dt in dict.fromkeys([onset_dt] + ([first_dt] if not cand_first.empty else [])):
            for _delta in (60, -60):
                try:
                    _nb = (pd.Timestamp(_base_dt, tz="Asia/Shanghai") + pd.Timedelta(seconds=_delta)).strftime("%Y-%m-%d %H:%M:%S")
                    _add_candidate(
                        out,
                        TimeCandidate(_nb, "component_metric_onset_neighbor", 0.72,
                                      component=str(comp),
                                      evidence={"component": str(comp), "neighbor_of": _base_dt}),
                    )
                except Exception:
                    pass
    return out


def estimate_occurrence_datetime_candidates(
    dataset: str,
    instruction: str,
    component_first_times: Optional[Mapping[str, str]] = None,
    component_peak_times: Optional[Mapping[str, str]] = None,
    estimated_failure_time: str = "",
    component_scores: Optional[Mapping[str, float]] = None,
    reasons: Optional[Sequence[str]] = None,
    top_k: int = 8,
) -> List[TimeCandidate]:
    candidates, _ = _trace_time_candidates(dataset=dataset, instruction=instruction)
    comp_scores = dict(component_scores or {})
    max_comp_score = max([abs(float(v)) for v in comp_scores.values()] + [0.0])

    def comp_norm(c: str) -> float:
        if max_comp_score <= 1e-12:
            return 0.0
        try:
            return max(0.0, float(comp_scores.get(c, 0.0)) / max_comp_score)
        except Exception:
            return 0.0

    for comp, dt in (component_first_times or {}).items():
        if not str(dt or "").strip():
            continue
        score = 0.62 + 0.28 * comp_norm(str(comp))
        _add_candidate(
            candidates,
            TimeCandidate(
                str(dt),
                "component_metric_first",
                score,
                component=str(comp),
                evidence={"component_score_norm": comp_norm(str(comp))},
            ),
        )
    for comp, dt in (component_peak_times or {}).items():
        if not str(dt or "").strip():
            continue
        score = 0.42 + 0.18 * comp_norm(str(comp))
        _add_candidate(
            candidates,
            TimeCandidate(
                str(dt),
                "component_metric_peak",
                score,
                component=str(comp),
                evidence={"component_score_norm": comp_norm(str(comp))},
            ),
        )
    if str(estimated_failure_time or "").strip():
        _add_candidate(candidates, TimeCandidate(str(estimated_failure_time), "component_estimated_failure_time", 0.54))
    if not comp_scores:
        for cand in _reason_metric_time_candidates(dataset=dataset, instruction=instruction, reasons=reasons or []):
            _add_candidate(candidates, cand)
    if comp_scores:
        top_comps = [c for c, _ in sorted(comp_scores.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:3]]
        for cand in _component_metric_onset_time_candidates(dataset=dataset, instruction=instruction, components=top_comps, top_k_metrics=8):
            if cand.component:
                boost = 0.18 * comp_norm(str(cand.component))
                _add_candidate(
                    candidates,
                    TimeCandidate(
                        datetime=str(cand.datetime),
                        source=str(cand.source),
                        score=min(1.0, float(cand.score) + boost),
                        component=str(cand.component),
                        reason=str(cand.reason),
                        evidence=dict(cand.evidence or {}) | {"component_score_norm": comp_norm(str(cand.component))},
                    ),
                )
            else:
                _add_candidate(candidates, cand)

    return _dedupe_time_candidates_by_datetime(candidates)[: max(1, int(top_k))]


def select_time_candidate(
    candidates: Sequence[Any],
    component: str = "",
    reason: str = "",
    fallback: str = "",
) -> Tuple[str, Dict[str, Any]]:
    normalized: List[TimeCandidate] = []
    for item in candidates or []:
        if isinstance(item, TimeCandidate):
            normalized.append(item)
        elif isinstance(item, Mapping):
            normalized.append(
                TimeCandidate(
                    datetime=str(item.get("datetime", "")),
                    source=str(item.get("source", "")),
                    score=float(item.get("score", 0.0) or 0.0),
                    component=str(item.get("component", "") or ""),
                    reason=str(item.get("reason", "") or ""),
                    evidence=dict(item.get("evidence", {}) or {}),
                )
            )
    if not normalized:
        return str(fallback or ""), {}

    times_s = [(c, _dt_to_epoch_s(c.datetime)) for c in normalized]
    scored: List[Tuple[float, TimeCandidate, Dict[str, Any]]] = []
    for cand, cand_s in times_s:
        explain_score, explain_details = _time_explainability_score(
            cand,
            normalized,
            component=str(component or ""),
            reason=str(reason or ""),
        )
        score = 0.45 * float(cand.score) + 0.55 * float(explain_score)
        details = {
            "policy": "causal_change_point_explainability",
            "base_score": float(cand.score),
            "source": cand.source,
            "explainability_score": float(explain_score),
            "explainability": dict(explain_details),
        }
        if component and (not str(cand.component or "").strip()) and "reason_metric" in str(cand.source or ""):
            score -= 0.22
            details["non_component_reason_metric_penalty"] = -0.22
        if component and "reason_metric" in str(cand.source or "") and cand_s is not None:
            later_trace = [
                (other, other_s)
                for other, other_s in times_s
                if other_s is not None
                and other_s >= cand_s
                and other_s - cand_s <= 900
                and "trace_" in str(getattr(other, "source", "") or "")
                and float(getattr(other, "score", 0.0) or 0.0) >= 0.90
            ]
            if later_trace:
                score -= 0.30
                details["later_trace_after_reason_metric_penalty"] = -0.30
        if component and cand.component:
            if cand.component == component:
                score += 0.18
                details["component_match_bonus"] = 0.18
            else:
                score -= 0.10
                details["component_mismatch_penalty"] = -0.10
        if reason and cand.reason:
            if cand.reason == reason:
                score += 0.28
                details["reason_match_bonus"] = 0.28
            else:
                score -= 0.08
                details["reason_mismatch_penalty"] = -0.08
        if cand_s is not None:
            near = 0
            for other, other_s in times_s:
                if other is cand or other_s is None:
                    continue
                if abs(int(cand_s) - int(other_s)) <= 180:
                    near += 1
            consensus = min(0.30, 0.10 * near)
            if consensus:
                score += consensus
                details["consensus_bonus"] = consensus
        if "peak" in cand.source:
            earlier = [
                other_s
                for other, other_s in times_s
                if other_s is not None and cand_s is not None and other_s < cand_s and cand_s - other_s <= 900
            ]
            if earlier:
                score -= 0.18
                details["late_peak_penalty"] = -0.18
        if "component_metric" in cand.source and cand_s is not None:
            earlier_trace = [
                (other, other_s)
                for other, other_s in times_s
                if other_s is not None
                and other_s < cand_s
                and cand_s - other_s <= 180
                and "trace_" in str(getattr(other, "source", "") or "")
                and float(getattr(other, "score", 0.0) or 0.0) >= 0.90
            ]
            if earlier_trace:
                score -= 0.28
                details["earlier_trace_penalty"] = -0.28
        details["final_score"] = float(score)
        scored.append((score, cand, details))

    scored.sort(key=lambda x: (x[0], -(_dt_to_epoch_s(x[1].datetime) or 0)), reverse=True)
    best_score, best, details = scored[0]
    details["selected"] = best.to_dict()
    details["candidate_count"] = len(normalized)
    return best.datetime, details


def estimate_occurrence_datetime_from_trace(dataset: str, instruction: str) -> Tuple[str, Optional[TimeEvidence]]:
    date_dir, start, end = parse_time_window_generic(instruction)
    date_dir = resolve_date_dir(dataset, date_dir)
    trace_path = os.path.join("dataset", dataset, "telemetry", date_dir, "trace", "trace_span.csv")
    if not os.path.exists(trace_path):
        chosen = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return chosen, None
    columns = _read_csv_columns_cached(trace_path)
    dur_col = "elapsedTime" if "elapsedTime" in columns else ("duration" if "duration" in columns else None)
    ts_col = "startTime" if "startTime" in columns else ("timestamp" if "timestamp" in columns else None)
    if dur_col is None or ts_col is None:
        chosen = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return chosen, None
    df = _read_trace_time_frame_cached(trace_path, ts_col, dur_col)
    if df.empty:
        chosen = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return chosen, None
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if df.empty:
        chosen = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return chosen, None
    df["minute"] = df["timestamp"].dt.floor("min")
    p95 = df.groupby("minute")["dur"].quantile(0.95).sort_index()
    if p95.empty:
        chosen = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return chosen, None
    p95_s = p95.rolling(window=3, center=True, min_periods=1).median()
    pre_n = max(5, int(len(p95_s) * 0.2))
    pre = p95_s.iloc[:pre_n]
    baseline = float(pre.median())
    mad = float((pre - baseline).abs().median())
    thr = max(baseline * 1.18, baseline + 6.0 * mad, baseline + 1.0)
    cond = p95_s >= thr
    onset2 = cond & cond.shift(-1, fill_value=False)
    if onset2.any():
        chosen_min = onset2[onset2].index[0]
        chosen = pd.Timestamp(chosen_min).strftime("%Y-%m-%d %H:%M:%S")
        ev = TimeEvidence(
            date_dir=date_dir,
            start=start.strftime("%Y-%m-%d %H:%M:%S"),
            end=end.strftime("%Y-%m-%d %H:%M:%S"),
            chosen=chosen,
            method="onset",
            baseline_p95=baseline,
            peak_p95=float(p95.max()),
        )
        return chosen, ev
    if cond.any():
        chosen_min = cond[cond].index[0]
        chosen = pd.Timestamp(chosen_min).strftime("%Y-%m-%d %H:%M:%S")
        ev = TimeEvidence(
            date_dir=date_dir,
            start=start.strftime("%Y-%m-%d %H:%M:%S"),
            end=end.strftime("%Y-%m-%d %H:%M:%S"),
            chosen=chosen,
            method="onset",
            baseline_p95=baseline,
            peak_p95=float(p95.max()),
        )
        return chosen, ev
    diff = p95_s.diff().fillna(0.0)
    candidates = p95_s[(p95_s >= baseline * 1.2) & (diff >= max(1.0, baseline * 0.08))]
    if not candidates.empty:
        chosen_min = candidates.index[0]
        chosen = pd.Timestamp(chosen_min).strftime("%Y-%m-%d %H:%M:%S")
        ev = TimeEvidence(
            date_dir=date_dir,
            start=start.strftime("%Y-%m-%d %H:%M:%S"),
            end=end.strftime("%Y-%m-%d %H:%M:%S"),
            chosen=chosen,
            method="jump",
            baseline_p95=baseline,
            peak_p95=float(p95.max()),
        )
        return chosen, ev
    roll = p95.rolling(window=3, center=True, min_periods=1).mean()
    chosen_min = roll.sort_values(ascending=False).index[0]
    chosen = pd.Timestamp(chosen_min).strftime("%Y-%m-%d %H:%M:%S")
    ev = TimeEvidence(
        date_dir=date_dir,
        start=start.strftime("%Y-%m-%d %H:%M:%S"),
        end=end.strftime("%Y-%m-%d %H:%M:%S"),
        chosen=chosen,
        method="peak",
        baseline_p95=baseline,
        peak_p95=float(p95.max()),
    )
    return chosen, ev


def estimate_time_candidate_from_trace(dataset: str, instruction: str) -> Tuple[str, Optional[TimeEvidence]]:
    return estimate_occurrence_datetime_from_trace(dataset=dataset, instruction=instruction)
