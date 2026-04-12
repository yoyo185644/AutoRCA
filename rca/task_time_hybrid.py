import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from rca.task3_hybrid import parse_time_window_generic


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


def estimate_occurrence_datetime_from_trace(dataset: str, instruction: str) -> Tuple[str, Optional[TimeEvidence]]:
    date_dir, start, end = parse_time_window_generic(instruction)
    trace_path = os.path.join("dataset", dataset, "telemetry", date_dir, "trace", "trace_span.csv")
    if not os.path.exists(trace_path):
        chosen = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return chosen, None
    df = pd.read_csv(trace_path, low_memory=False)
    dur_col = "elapsedTime" if "elapsedTime" in df.columns else ("duration" if "duration" in df.columns else None)
    if df.empty or dur_col is None:
        chosen = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return chosen, None
    ts_col = "startTime" if "startTime" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if ts_col is None:
        chosen = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return chosen, None
    df = df[[ts_col, dur_col]].copy()
    ts_unit = "ms" if df[ts_col].dropna().astype(str).str.len().max() >= 13 else "s"
    df["timestamp"] = pd.to_datetime(df[ts_col], unit=ts_unit, utc=True).dt.tz_convert("Asia/Shanghai")
    df["dur"] = pd.to_numeric(df[dur_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "dur"])
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if df.empty:
        chosen = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return chosen, None
    df["minute"] = df["timestamp"].dt.floor("min")
    p95 = df.groupby("minute")["dur"].quantile(0.95).sort_index()
    if p95.empty:
        chosen = start.floor("min").strftime("%Y-%m-%d %H:%M:%S")
        return chosen, None
    baseline = float(p95.median())
    diff = p95.diff().fillna(0.0)
    candidates = p95[(p95 >= baseline * 1.2) & (diff >= max(1.0, baseline * 0.1))]
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
    chosen_min = p95.sort_values(ascending=False).index[0]
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
