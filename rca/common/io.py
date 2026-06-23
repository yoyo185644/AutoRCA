from __future__ import annotations

import os
from typing import Any

import pandas as pd


def read_csv_with_python_fallback(path: str, **kwargs: Any) -> pd.DataFrame:
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


def timestamp_to_seconds(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    sample = float(s.dropna().iloc[0])
    if sample > 10**12:
        return (s // 1000).astype("Int64")
    return s.astype("Int64")


def to_local_string(ts_s: int, tz: str = "Asia/Shanghai") -> str:
    return pd.to_datetime(int(ts_s), unit="s", utc=True).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")


def parse_local_time_s(value: str, fallback: int = 0, tz: str = "Asia/Shanghai") -> int:
    if not str(value or "").strip():
        return int(fallback)
    try:
        return int(pd.Timestamp(value, tz=tz).tz_convert("UTC").timestamp())
    except Exception:
        return int(fallback)


def resolve_date_dir(dataset: str, date_dir: str) -> str:
    base = os.path.join("dataset", str(dataset), "telemetry")
    raw = str(date_dir or "").strip()
    if not raw:
        return raw
    candidates = [raw]
    if "-" in raw:
        candidates.append(raw.replace("-", "_"))
    if "_" in raw:
        candidates.append(raw.replace("_", "-"))
    for cand in candidates:
        if os.path.isdir(os.path.join(base, cand)):
            return cand
    return raw
