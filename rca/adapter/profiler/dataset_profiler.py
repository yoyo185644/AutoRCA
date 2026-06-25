from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from rca.common.config import load_dataset_config
from rca.common.io import read_csv_with_python_fallback, timestamp_to_seconds
from rca.adapter.profiler.schema import (
    CandidateBudgetProfile,
    DatasetProfile,
    MetricFamilyProfile,
    TelemetryAvailability,
)


_METRIC_FILE_HINTS = ("metric", "kpi", "container", "node", "service", "runtime", "mesh")
_TRACE_FILE_HINTS = ("trace", "span")
_LOG_FILE_HINTS = ("log", "event")
_COMPONENT_COLUMNS = (
    "cmdb_id",
    "component",
    "component_id",
    "container",
    "instance",
    "node",
    "service",
    "serviceName",
    "pod",
    "host",
)
_TIME_COLUMNS = ("timestamp", "time", "ts", "startTime", "start_time", "datetime", "date")
_METRIC_NAME_COLUMNS = ("metric", "metric_name", "name", "kpi", "item", "指标")
_VALUE_COLUMNS = (
    "value",
    "val",
    "mean",
    "avg",
    "avg_time",
    "count",
    "num",
    "success",
    "succee_num",
    "succee_rate",
    "rr",
    "sr",
    "mrt",
    "duration",
    "elapsedTime",
)


class DatasetProfiler:
    """Build an unlabeled telemetry profile for adaptive BASIL planning.

    The profiler intentionally avoids scoring_points, ground truth, and result
    CSVs. It only scans telemetry schema and distributions so the generated
    parameters are data-derived rather than tuned on test labels.
    """

    def __init__(
        self,
        dataset: str,
        *,
        telemetry_root: Optional[str] = None,
        max_files: int = 80,
        max_rows_per_file: int = 200_000,
    ) -> None:
        self.dataset = str(dataset)
        self.config = load_dataset_config(dataset)
        self.telemetry_root = telemetry_root or self._resolve_telemetry_root()
        self.max_files = int(max_files)
        self.max_rows_per_file = int(max_rows_per_file)

    def build(self) -> DatasetProfile:
        warnings: List[str] = []
        files = self._list_files()
        if not files:
            warnings.append(f"no telemetry files found under {self.telemetry_root}")

        metric_files = [p for p in files if self._classify_file(p) == "metric"]
        trace_files = [p for p in files if self._classify_file(p) == "trace"]
        log_files = [p for p in files if self._classify_file(p) == "log"]

        component_values: set[str] = set()
        metric_names: set[str] = set()
        component_types: Counter[str] = Counter()
        family_stats: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        intervals: List[float] = []
        min_ts: Optional[int] = None
        max_ts: Optional[int] = None

        for path in metric_files[: self.max_files]:
            try:
                file_profile = self._profile_metric_file(path)
            except Exception as exc:
                warnings.append(f"failed to profile {path}: {exc}")
                continue
            component_values.update(file_profile.get("components", set()))
            metric_names.update(file_profile.get("metrics", set()))
            component_types.update(file_profile.get("component_types", Counter()))
            intervals.extend(file_profile.get("intervals", []))
            for family, stats in dict(file_profile.get("families", {})).items():
                family_stats[str(family)].append(dict(stats))
            ts_min = file_profile.get("min_ts")
            ts_max = file_profile.get("max_ts")
            if ts_min is not None:
                min_ts = int(ts_min) if min_ts is None else min(min_ts, int(ts_min))
            if ts_max is not None:
                max_ts = int(ts_max) if max_ts is None else max(max_ts, int(ts_max))

        family_profiles = {
            family: self._merge_family_stats(family, items)
            for family, items in family_stats.items()
            if items
        }
        median_interval = self._median(intervals)
        component_count = len(component_values)
        metric_count = len(metric_names)
        budget = self._estimate_candidate_budget(
            component_count=component_count,
            metric_count=metric_count,
            family_profiles=family_profiles,
        )

        day_count = len(
            [
                name
                for name in os.listdir(self.telemetry_root)
                if os.path.isdir(os.path.join(self.telemetry_root, name))
            ]
        ) if os.path.isdir(self.telemetry_root) else 0

        return DatasetProfile(
            dataset=self.dataset,
            telemetry_root=self.telemetry_root,
            timezone=str(self.config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai"),
            availability=TelemetryAvailability(
                metric_files=len(metric_files),
                trace_files=len(trace_files),
                log_files=len(log_files),
            ),
            day_count=day_count,
            component_count=component_count,
            metric_count=metric_count,
            component_types=dict(component_types),
            metric_families=family_profiles,
            time_span_seconds=max(0, int(max_ts - min_ts)) if min_ts is not None and max_ts is not None else 0,
            median_sampling_interval_s=median_interval,
            typical_anomaly_duration_s=self._estimate_duration(median_interval),
            typical_propagation_lag_s=self._estimate_lag(median_interval, bool(trace_files)),
            budget=budget,
            warnings=warnings,
        )

    def _resolve_telemetry_root(self) -> str:
        paths = self.config.get("paths", {}) if isinstance(self.config, Mapping) else {}
        template = str(paths.get("telemetry_root", "dataset/{dataset}/telemetry"))
        return template.format(dataset=self.dataset)

    def _list_files(self) -> List[str]:
        if not os.path.isdir(self.telemetry_root):
            return []
        out: List[str] = []
        for root, _dirs, names in os.walk(self.telemetry_root):
            for name in names:
                if not name.lower().endswith((".csv", ".tsv")):
                    continue
                out.append(os.path.join(root, name))
        return sorted(out)

    def _classify_file(self, path: str) -> str:
        name = os.path.basename(path).lower()
        parent = os.path.basename(os.path.dirname(path)).lower()
        text = f"{parent}/{name}"
        if any(tok in text for tok in _LOG_FILE_HINTS):
            return "log"
        if any(tok in text for tok in _TRACE_FILE_HINTS):
            return "trace"
        if any(tok in text for tok in _METRIC_FILE_HINTS):
            return "metric"
        return "metric"

    def _profile_metric_file(self, path: str) -> Dict[str, Any]:
        sep = "\t" if path.lower().endswith(".tsv") else ","
        header = read_csv_with_python_fallback(path, nrows=0, sep=sep)
        columns = [str(c) for c in header.columns]
        usecols = self._select_usecols(columns)
        if not usecols:
            return {}
        df = read_csv_with_python_fallback(path, usecols=usecols, nrows=self.max_rows_per_file, sep=sep, low_memory=False)
        if df.empty:
            return {}

        comp_col = self._first_column(df.columns, _COMPONENT_COLUMNS)
        time_col = self._first_column(df.columns, _TIME_COLUMNS)
        metric_col = self._first_column(df.columns, _METRIC_NAME_COLUMNS)
        value_cols = self._value_columns(df)

        components = set()
        component_types: Counter[str] = Counter()
        if comp_col:
            components = {str(x).strip() for x in df[comp_col].dropna().unique() if str(x).strip()}
            component_types.update(self._component_type(x) for x in components)

        metrics = set()
        if metric_col:
            metrics.update(str(x).strip() for x in df[metric_col].dropna().unique() if str(x).strip())
        metrics.update(value_cols)

        intervals: List[float] = []
        min_ts = None
        max_ts = None
        if time_col:
            ts = self._timestamp_seconds(df[time_col])
            clean = sorted(int(x) for x in ts.dropna().unique())
            if clean:
                min_ts, max_ts = clean[0], clean[-1]
                intervals.extend(self._positive_diffs(clean[:5000]))

        families: Dict[str, Dict[str, Any]] = {}
        for value_col in value_cols:
            values = pd.to_numeric(df[value_col], errors="coerce").dropna()
            if values.empty:
                continue
            family = self._metric_family(value_col)
            families[family] = self._summarize_values(
                family=family,
                values=values,
                metric_count=1,
                component_count=len(components),
                sample_count=len(values),
                sparsity=float(pd.to_numeric(df[value_col], errors="coerce").isna().mean()),
                interval_s=self._median(intervals),
            )

        if metric_col and "value" in df.columns:
            numeric = pd.to_numeric(df["value"], errors="coerce")
            for metric_name, group in df.assign(_value=numeric).dropna(subset=["_value"]).groupby(metric_col):
                family = self._metric_family(str(metric_name))
                values = group["_value"]
                families[family] = self._summarize_values(
                    family=family,
                    values=values,
                    metric_count=int(group[metric_col].nunique()),
                    component_count=len(components),
                    sample_count=len(values),
                    sparsity=float(group["_value"].isna().mean()),
                    interval_s=self._median(intervals),
                )

        return {
            "components": components,
            "metrics": metrics,
            "component_types": component_types,
            "families": families,
            "intervals": intervals,
            "min_ts": min_ts,
            "max_ts": max_ts,
        }

    def _select_usecols(self, columns: Sequence[str]) -> List[str]:
        out: List[str] = []
        for col in columns:
            low = str(col).strip().lower()
            if low in {x.lower() for x in _COMPONENT_COLUMNS + _TIME_COLUMNS + _METRIC_NAME_COLUMNS}:
                out.append(str(col))
                continue
            if low in {x.lower() for x in _VALUE_COLUMNS}:
                out.append(str(col))
                continue
            if self._looks_numeric_metric(col):
                out.append(str(col))
        return list(dict.fromkeys(out))

    def _value_columns(self, df: pd.DataFrame) -> List[str]:
        non_values = {str(x).lower() for x in _COMPONENT_COLUMNS + _TIME_COLUMNS + _METRIC_NAME_COLUMNS}
        out: List[str] = []
        for col in df.columns:
            if str(col).lower() in non_values:
                continue
            sample = pd.to_numeric(df[col], errors="coerce")
            if sample.notna().mean() >= 0.5:
                out.append(str(col))
        return out

    def _first_column(self, columns: Iterable[Any], candidates: Sequence[str]) -> Optional[str]:
        by_lower = {str(c).lower(): str(c) for c in columns}
        for cand in candidates:
            if cand.lower() in by_lower:
                return by_lower[cand.lower()]
        return None

    def _timestamp_seconds(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            return timestamp_to_seconds(numeric)
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return (parsed.astype("int64") // 10**9).where(parsed.notna())

    def _positive_diffs(self, values: Sequence[int]) -> List[float]:
        out: List[float] = []
        prev = None
        for value in values:
            if prev is not None and int(value) > int(prev):
                out.append(float(int(value) - int(prev)))
            prev = int(value)
        return out

    def _summarize_values(
        self,
        *,
        family: str,
        values: pd.Series,
        metric_count: int,
        component_count: int,
        sample_count: int,
        sparsity: float,
        interval_s: Optional[float],
    ) -> Dict[str, Any]:
        q50 = float(values.quantile(0.50))
        q95 = float(values.quantile(0.95))
        q99 = float(values.quantile(0.99))
        mad = float((values - q50).abs().median())
        robust_scale = max(abs(q50), abs(q95) * 0.05, 1.0)
        return {
            "family": family,
            "metric_count": int(metric_count),
            "component_count": int(component_count),
            "sample_count": int(sample_count),
            "median_sampling_interval_s": interval_s,
            "sparsity": float(sparsity),
            "noise_mad_ratio": float(mad / robust_scale),
            "burstiness_p99_p50": float(q99 / robust_scale),
            "value_p50": q50,
            "value_p95": q95,
            "value_p99": q99,
        }

    def _merge_family_stats(self, family: str, items: Sequence[Mapping[str, Any]]) -> MetricFamilyProfile:
        weight_sum = sum(int(x.get("sample_count", 0) or 0) for x in items) or len(items) or 1

        def wavg(key: str, default: float = 0.0) -> float:
            total = 0.0
            for item in items:
                w = int(item.get("sample_count", 0) or 0) or 1
                total += float(item.get(key, default) or default) * w
            return float(total / max(weight_sum, 1))

        intervals = [float(x.get("median_sampling_interval_s")) for x in items if x.get("median_sampling_interval_s") is not None]
        return MetricFamilyProfile(
            family=family,
            metric_count=sum(int(x.get("metric_count", 0) or 0) for x in items),
            component_count=max(int(x.get("component_count", 0) or 0) for x in items),
            sample_count=sum(int(x.get("sample_count", 0) or 0) for x in items),
            median_sampling_interval_s=self._median(intervals),
            sparsity=wavg("sparsity"),
            noise_mad_ratio=wavg("noise_mad_ratio"),
            burstiness_p99_p50=max(1.0, wavg("burstiness_p99_p50", 1.0)),
            value_p50=wavg("value_p50"),
            value_p95=wavg("value_p95"),
            value_p99=wavg("value_p99"),
        )

    def _estimate_candidate_budget(
        self,
        *,
        component_count: int,
        metric_count: int,
        family_profiles: Mapping[str, MetricFamilyProfile],
    ) -> CandidateBudgetProfile:
        scale = math.sqrt(max(component_count, 1))
        avg_noise = self._mean([p.noise_mad_ratio for p in family_profiles.values()])
        avg_burst = self._mean([p.burstiness_p99_p50 for p in family_profiles.values()], default=1.0)
        uncertainty = min(2.0, max(0.0, avg_noise)) + min(2.0, max(0.0, avg_burst - 1.0) / 10.0)
        top_k = int(max(6, min(48, round(scale * (1.4 + uncertainty)))))
        reason_top_r = int(max(3, min(8, round(3 + uncertainty))))
        time_top_k = int(max(5, min(16, round(5 + uncertainty * 3))))
        entropy_hint = float(min(1.0, uncertainty / 3.0))
        elbow_hint = int(max(6, min(top_k, round(scale * 1.6))))
        return CandidateBudgetProfile(
            component_count=int(component_count),
            metric_count=int(metric_count),
            estimated_top_k=top_k,
            estimated_reason_top_r=reason_top_r,
            estimated_time_top_k=time_top_k,
            score_entropy_hint=entropy_hint,
            score_elbow_hint=elbow_hint,
        )

    def _estimate_duration(self, interval_s: Optional[float]) -> int:
        if interval_s is None or interval_s <= 0:
            return 300
        return int(max(60, min(900, round(float(interval_s) * 5))))

    def _estimate_lag(self, interval_s: Optional[float], has_trace: bool) -> int:
        base = 2 if has_trace else 4
        if interval_s is None or interval_s <= 0:
            return 300
        return int(max(60, min(900, round(float(interval_s) * base))))

    def _component_type(self, component: str) -> str:
        c = str(component or "").lower()
        if c.startswith("db_") or "mysql" in c or "database" in c:
            return "db"
        if c.startswith("os_") or c.startswith("node-") or "node" in c:
            return "node"
        if c.startswith("docker_") or "container" in c:
            return "container"
        if "redis" in c:
            return "cache"
        if c:
            return "service"
        return "unknown"

    def _metric_family(self, metric: str) -> str:
        text = str(metric or "").lower()
        if any(tok in text for tok in ("cpu", "processor", "load", "iowait")):
            return "cpu"
        if any(tok in text for tok in ("mem", "memory", "rss", "heap")):
            return "memory"
        if any(tok in text for tok in ("disk", "io", "read", "write", "await", "iops")):
            return "disk_io"
        if any(tok in text for tok in ("net", "send", "recv", "packet", "loss", "latency", "delay", "rtt")):
            return "network"
        if any(tok in text for tok in ("db", "sql", "session", "login", "tps", "query", "exec", "commit")):
            return "database"
        if any(tok in text for tok in ("duration", "elapsed", "mrt", "latency")):
            return "latency"
        return "other"

    def _looks_numeric_metric(self, col: str) -> bool:
        text = str(col or "").lower()
        if text in {"id", "traceid", "spanid", "parentid"}:
            return False
        return bool(re.search(r"(cpu|mem|disk|io|net|count|rate|pct|time|latency|duration|rr|sr|mrt)", text))

    def _median(self, values: Sequence[float]) -> Optional[float]:
        clean = sorted(float(x) for x in values if x is not None and not math.isnan(float(x)))
        if not clean:
            return None
        mid = len(clean) // 2
        if len(clean) % 2:
            return float(clean[mid])
        return float((clean[mid - 1] + clean[mid]) / 2.0)

    def _mean(self, values: Sequence[float], default: float = 0.0) -> float:
        clean = [float(x) for x in values if x is not None and not math.isnan(float(x))]
        if not clean:
            return float(default)
        return float(sum(clean) / len(clean))
