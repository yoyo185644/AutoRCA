from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


_TOPOLOGY_CACHE: Dict[str, pd.DataFrame] = {}
_HEADER_CACHE: Dict[str, List[str]] = {}
_LISTDIR_CACHE: Dict[str, List[str]] = {}


def _cached_listdir(directory: str) -> List[str]:
    """In-process cache for os.listdir."""
    key = str(directory)
    if key not in _LISTDIR_CACHE:
        try:
            _LISTDIR_CACHE[key] = os.listdir(key)
        except Exception:
            _LISTDIR_CACHE[key] = []
    return list(_LISTDIR_CACHE[key])


def _cached_header(path: str) -> List[str]:
    """In-process cache for reading CSV headers."""
    key = str(path)
    if key not in _HEADER_CACHE:
        try:
            _HEADER_CACHE[key] = list(pd.read_csv(key, nrows=0).columns)
        except Exception:
            _HEADER_CACHE[key] = []
    return list(_HEADER_CACHE[key])


def _chunk_merge_topology(trace_path: str, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Vectorize edge counting and compress dtypes; output is identical to row-wise counting."""
    c = TelemetryCache._trace_columns(trace_path)
    needed = [x for x in [c["ts"], c["trace"], c["component"], c["span"], c["parent"], c["ds"], c["service"]] if x]
    if not needed or not c["component"]:
        return pd.DataFrame(columns=["minute", "src", "dst", "edge_type", "support_count"])

    edge_counts: Counter[Tuple[int, str, str, str]] = Counter()
    for chunk in TelemetryCache._read_csv_chunks(trace_path, usecols=needed, chunksize=TelemetryCache._chunk_size, nrows=max_rows):
        if chunk.empty:
            continue
        for col in [c.get("trace"), c.get("span"), c.get("parent"), c.get("component"), c.get("service"), c.get("ds")]:
            if col and col in chunk.columns:
                chunk[col] = chunk[col].astype("category")
        if c["ts"] and c["ts"] in chunk.columns:
            ts_s = TelemetryCache._timestamp_to_s(chunk[c["ts"]])
            chunk = chunk.copy()
            chunk["minute"] = ((ts_s.astype("Int64") // 60) * 60).astype("Int64")
            chunk = chunk.dropna(subset=["minute"])
        else:
            chunk = chunk.copy()
            chunk["minute"] = 0

        if c["trace"] and c["span"] and c["parent"] and all(x in chunk.columns for x in [c["trace"], c["span"], c["parent"], c["component"]]):
            child = chunk[[c["trace"], c["span"], c["parent"], c["component"], "minute"]].dropna().copy()
            child.columns = ["trace_id", "span_id", "parent_id", "child_component", "minute"]
            parent = chunk[[c["trace"], c["span"], c["component"]]].dropna().copy()
            parent.columns = ["trace_id", "parent_id", "parent_component"]
            merged = child.merge(parent, on=["trace_id", "parent_id"], how="inner")
            if not merged.empty:
                merged = merged.dropna(subset=["minute", "child_component", "parent_component"])
                counts = merged.groupby(["minute", "child_component", "parent_component"], observed=True).size()
                for (minute, src, dst), count in counts.items():
                    src_s = str(src or "").strip()
                    dst_s = str(dst or "").strip()
                    if src_s and dst_s and src_s != dst_s and int(count) > 0:
                        edge_counts[(int(minute), src_s, dst_s, "trace_latency")] += int(count)

        if c["ds"] and c["ds"] in chunk.columns and c["component"] in chunk.columns:
            pair = chunk[[c["component"], c["ds"], "minute"]].dropna().copy()
            if not pair.empty:
                pair = pair.rename(columns={c["component"]: "dst", c["ds"]: "src"})
                pair = pair.dropna(subset=["minute", "src", "dst"])
                counts = pair.groupby(["minute", "src", "dst"], observed=True).size()
                for (minute, src, dst), count in counts.items():
                    src_s = str(src or "").strip()
                    dst_s = str(dst or "").strip()
                    if src_s and dst_s and src_s != dst_s and int(count) > 0:
                        edge_counts[(int(minute), src_s, dst_s, "db_dependency")] += int(count)

    rows = [
        {
            "minute": int(minute),
            "src": str(src),
            "dst": str(dst),
            "edge_type": str(edge_type),
            "support_count": int(count),
        }
        for (minute, src, dst, edge_type), count in edge_counts.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["minute", "src", "dst", "edge_type", "support_count"])
    return pd.DataFrame(rows)


def _get_topology_edges(trace_path: str, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Return topology edges for trace_path.
    Computes once per process, caches in-process for subsequent queries.
    """
    row_cap = int(max_rows or 0)
    key = str(trace_path) if row_cap <= 0 else f"{str(trace_path)}::max_rows={row_cap}"
    if key not in _TOPOLOGY_CACHE:
        _TOPOLOGY_CACHE[key] = _chunk_merge_topology(str(trace_path), max_rows=max_rows)
    return _TOPOLOGY_CACHE[key]


class TelemetryCache:
    """Process-level cache for aggregated telemetry features.

    The cache stores compact aggregations rather than raw telemetry frames. It is
    shared by all queries in the current Python process, so repeated queries on
    the same dataset/date_dir reuse the same trace summaries.
    """

    _trace_minute_p95: Dict[Tuple[str, str], pd.DataFrame] = {}
    _topology_edges: Dict[Tuple[str, str, int], pd.DataFrame] = {}
    _metric_scores: Dict[Tuple[str, str], pd.DataFrame] = {}
    _chunk_size = 500_000

    @classmethod
    def get_trace_minute_p95(cls, dataset: str, date_dir: str, trace_path: str) -> pd.DataFrame:
        key = (str(dataset), str(date_dir))
        if key not in cls._trace_minute_p95:
            cls._trace_minute_p95[key] = cls._compute_trace_minute_p95(trace_path)
        return cls._trace_minute_p95[key]

    @classmethod
    def get_topology_edges(
        cls,
        dataset: str,
        date_dir: str,
        trace_path: str,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        row_cap = int(max_rows or 0)
        key = (str(dataset), str(date_dir), row_cap)
        if key not in cls._topology_edges:
            cls._topology_edges[key] = cls._compute_topology_edges(trace_path, max_rows=max_rows)
        return cls._topology_edges[key]

    @classmethod
    def get_metric_scores(cls, dataset: str, date_dir: str, metric_paths: Iterable[str]) -> pd.DataFrame:
        key = (str(dataset), str(date_dir))
        if key not in cls._metric_scores:
            cls._metric_scores[key] = cls._compute_metric_scores(metric_paths)
        return cls._metric_scores[key]

    @classmethod
    def _read_header(cls, path: str) -> List[str]:
        return _cached_header(path)

    @classmethod
    def _read_csv_chunks(cls, path: str, usecols: List[str], chunksize: int, nrows: Optional[int] = None):
        kwargs = {
            "usecols": list(dict.fromkeys(usecols)),
            "chunksize": int(chunksize),
            "low_memory": False,
        }
        if nrows is not None and int(nrows) > 0:
            kwargs["nrows"] = int(nrows)
        try:
            yield from pd.read_csv(path, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if "Try engine='python'" not in msg and "Error tokenizing data" not in msg:
                raise
            kwargs = dict(kwargs)
            kwargs["engine"] = "python"
            kwargs.pop("low_memory", None)
            yield from pd.read_csv(path, **kwargs)

    @staticmethod
    def _timestamp_to_s(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if s.dropna().empty:
            return s
        v = float(s.dropna().iloc[0])
        if v > 10**12:
            return (s // 1000).astype("Int64")
        return s.astype("Int64")

    @staticmethod
    def _trace_columns(path: str) -> Dict[str, Optional[str]]:
        cols = set(TelemetryCache._read_header(path))
        return {
            "ts": "timestamp" if "timestamp" in cols else ("startTime" if "startTime" in cols else None),
            "component": "cmdb_id" if "cmdb_id" in cols else ("serviceName" if "serviceName" in cols else None),
            "duration": "duration" if "duration" in cols else ("elapsedTime" if "elapsedTime" in cols else None),
            "trace": "trace_id" if "trace_id" in cols else ("traceId" if "traceId" in cols else None),
            "span": "span_id" if "span_id" in cols else ("id" if "id" in cols else None),
            "parent": "parent_span" if "parent_span" in cols else ("parent_id" if "parent_id" in cols else ("pid" if "pid" in cols else None)),
            "ds": "dsName" if "dsName" in cols else None,
            "service": "serviceName" if "serviceName" in cols else None,
        }

    @classmethod
    def _compute_trace_minute_p95(cls, trace_path: str) -> pd.DataFrame:
        c = cls._trace_columns(trace_path)
        if not c["ts"] or not c["component"] or not c["duration"]:
            return pd.DataFrame(columns=["minute", "component", "p95", "count"])

        usecols = [c["ts"], c["component"], c["duration"]]
        # Collect only the three columns we need from each chunk; avoid the
        # Python-level dict accumulation that was copying large lists per group.
        parts: List[pd.DataFrame] = []
        for chunk in cls._read_csv_chunks(trace_path, usecols=usecols, chunksize=cls._chunk_size):
            if chunk.empty:
                continue
            ts_s = cls._timestamp_to_s(chunk[c["ts"]])
            dur = pd.to_numeric(chunk[c["duration"]], errors="coerce")
            comp = chunk[c["component"]].astype("category")
            work = pd.DataFrame({"ts_s": ts_s, "component": comp, "dur": dur}).dropna(
                subset=["ts_s", "component", "dur"]
            )
            if work.empty:
                continue
            work["minute"] = (work["ts_s"].astype("int64") // 60) * 60
            parts.append(work[["minute", "component", "dur"]])

        if not parts:
            return pd.DataFrame(columns=["minute", "component", "p95", "count"])

        # Single concat + two groupby-quantile passes (Cython path, no Python loop)
        merged = pd.concat(parts, ignore_index=True)
        merged["component"] = merged["component"].astype(str)

        per_comp = (
            merged.groupby(["minute", "component"], sort=False)["dur"]
            .agg(p95=lambda x: float(np.percentile(x, 95)), count="count")
            .reset_index()
        )

        # Global "__all__" row per minute (same semantics as before)
        per_minute = (
            merged.groupby("minute", sort=False)["dur"]
            .agg(p95=lambda x: float(np.percentile(x, 95)), count="count")
            .reset_index()
        )
        per_minute["component"] = "__all__"

        result = pd.concat([per_comp, per_minute[["minute", "component", "p95", "count"]]], ignore_index=True)
        return result.sort_values(["minute", "component"]).reset_index(drop=True)

    @classmethod
    def _compute_topology_edges(cls, trace_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        return _get_topology_edges(trace_path, max_rows=max_rows)

    @classmethod
    def _compute_metric_scores(cls, metric_paths: Iterable[str]) -> pd.DataFrame:
        frames = []
        for path in metric_paths:
            try:
                for chunk in cls._read_csv_chunks(path, usecols=cls._read_header(path), chunksize=200_000):
                    frames.append(chunk)
            except Exception:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @classmethod
    def invalidate(cls, dataset: str, date_dir: str) -> None:
        key = (str(dataset), str(date_dir))
        cls._trace_minute_p95.pop(key, None)
        cls._metric_scores.pop(key, None)
        for k in list(cls._topology_edges.keys()):
            if k[0] == key[0] and k[1] == key[1]:
                cls._topology_edges.pop(k, None)

    @classmethod
    def clear_all(cls) -> None:
        cls._trace_minute_p95.clear()
        cls._topology_edges.clear()
        cls._metric_scores.clear()

    @classmethod
    def memory_usage_mb(cls) -> dict:
        def frame_mb(values) -> float:
            total = 0
            for df in values:
                try:
                    total += int(df.memory_usage(deep=True).sum())
                except Exception:
                    total += 0
            return total / 1e6

        return {
            "trace_minute_p95_mb": frame_mb(cls._trace_minute_p95.values()),
            "topology_edges_mb": frame_mb(cls._topology_edges.values()),
            "metric_scores_mb": frame_mb(cls._metric_scores.values()),
        }
