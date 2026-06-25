from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Optional

from rca.common.config import load_dataset_config
from rca.adapter.profiler import AdaptiveKnobs, DatasetProfiler, build_adaptive_basil_config


def basil_enabled(args: Any) -> bool:
    return bool(getattr(args, "basil_index", False))


def infer_basil_task_type(*, required_fields: Optional[Iterable[str]] = None, task_id: Optional[int] = None) -> str:
    fields = {str(x).lower() for x in (required_fields or []) if str(x).strip()}
    if not fields and task_id is not None:
        tid = int(task_id)
        if tid == 1:
            fields = {"time"}
        elif tid == 2:
            fields = {"reason"}
        elif tid == 3:
            fields = {"component"}
        elif tid == 4:
            fields = {"time", "reason"}
        elif tid == 5:
            fields = {"component", "time"}
        elif tid == 6:
            fields = {"component", "reason"}
        elif tid == 7:
            fields = {"component", "reason", "time"}

    if fields == {"time"}:
        return "TIME_ONLY"
    if fields == {"reason"}:
        return "REASON_ONLY"
    if fields == {"component"}:
        return "COMPONENT_ONLY"
    if fields == {"reason", "time"}:
        return "REASON_TIME"
    if fields == {"component", "time"}:
        return "COMPONENT_TIME"
    if fields == {"component", "reason"}:
        return "COMPONENT_REASON"
    if fields == {"component", "reason", "time"}:
        return "COMPONENT_REASON_TIME"
    return "COMPONENT_REASON_TIME"


def basil_config_from_args(
    args: Any,
    top_k: int,
    *,
    required_fields: Optional[Iterable[str]] = None,
    task_id: Optional[int] = None,
) -> Dict[str, Any]:
    max_trace_rows_raw = getattr(args, "basil_max_trace_rows", None)
    try:
        max_trace_rows: Optional[int] = int(max_trace_rows_raw)
        if max_trace_rows <= 0:
            max_trace_rows = None
    except Exception:
        max_trace_rows = None
    task_type_override = str(getattr(args, "basil_task_type", "AUTO") or "AUTO").strip().upper()
    task_type = task_type_override if task_type_override != "AUTO" else infer_basil_task_type(required_fields=required_fields, task_id=task_id)
    return {
        "top_k": int(top_k),
        "bucket_size_s": int(getattr(args, "basil_bucket_size", 300)),
        "block_size": int(getattr(args, "basil_block_size", 32)),
        "max_lag_s": int(getattr(args, "basil_max_lag", 900)),
        "severity_floor": float(getattr(args, "basil_severity_floor", 0.0)),
        "w_local": float(getattr(args, "basil_w_local", 1.0)),
        "w_causal": float(getattr(args, "basil_w_causal", 0.45)),
        "w_reason": float(getattr(args, "basil_w_reason", 0.25)),
        "approximate_epsilon": float(getattr(args, "basil_approx_epsilon", 0.0)),
        "max_trace_rows": max_trace_rows,
        "persist_memory": bool(getattr(args, "basil_persist_index", True)),
        "memory_cache_dir": str(getattr(args, "basil_index_cache_dir", "test/cache/basil_index")),
        "similarity_ts_weight": float(getattr(args, "basil_similarity_ts_weight", 0.45)),
        "similarity_topology_weight": float(getattr(args, "basil_similarity_topology_weight", 0.30)),
        "similarity_pattern_weight": float(getattr(args, "basil_similarity_pattern_weight", 0.25)),
        "history_decay_lambda": float(getattr(args, "basil_history_decay_lambda", 0.03)),
        "historical_top_k": int(getattr(args, "basil_historical_top_k", 5)),
        "min_evidence_threshold": int(getattr(args, "basil_min_evidence_threshold", 2)),
        "task_type": task_type,
        "rerank_top_n": int(getattr(args, "basil_rerank_top_n", 20)),
        "candidate_reason_top_r": int(getattr(args, "basil_candidate_reason_top_r", 3)),
        "time_tolerance_s": int(getattr(args, "basil_time_tolerance_s", 300)),
    }


def apply_dataset_basil_config(config: Mapping[str, Any], *, dataset: str, args: Any = None) -> Dict[str, Any]:
    """Merge dataset-level BASIL index defaults into a loose config dict.

    Command-line values still win when they differ from the parser defaults.
    Dataset config is mainly used to provide per-dataset cache directories,
    bucket sizes, and query tolerances without hard-coding dataset branches.
    """

    out: Dict[str, Any] = dict(config or {})
    ds_cfg = load_dataset_config(dataset)
    index_cfg = ds_cfg.get("index", {}) if isinstance(ds_cfg, Mapping) else {}
    if not isinstance(index_cfg, Mapping):
        return out

    cache_cfg = index_cfg.get("cache", {})
    build_cfg = index_cfg.get("build", {})
    query_cfg = index_cfg.get("query", {})
    if not isinstance(cache_cfg, Mapping):
        cache_cfg = {}
    if not isinstance(build_cfg, Mapping):
        build_cfg = {}
    if not isinstance(query_cfg, Mapping):
        query_cfg = {}

    def arg_is_default(name: str, default: Any) -> bool:
        if args is None:
            return True
        return getattr(args, name, default) == default

    if "enabled" in cache_cfg and arg_is_default("basil_persist_index", True):
        out["persist_memory"] = bool(cache_cfg.get("enabled"))
    if cache_cfg.get("dir") and arg_is_default("basil_index_cache_dir", "test/cache/basil_index"):
        out["memory_cache_dir"] = str(cache_cfg.get("dir"))

    if "bucket_size_seconds" in build_cfg and arg_is_default("basil_bucket_size", 300):
        out["bucket_size_s"] = int(build_cfg.get("bucket_size_seconds") or out.get("bucket_size_s", 300))
    if "severity_floor" in build_cfg and arg_is_default("basil_severity_floor", 0.0):
        out["severity_floor"] = float(build_cfg.get("severity_floor") or 0.0)

    if "top_k" in query_cfg and int(out.get("top_k", 0) or 0) <= 0:
        out["top_k"] = int(query_cfg.get("top_k") or 12)
    if "time_tolerance_seconds" in query_cfg and arg_is_default("basil_time_tolerance_s", 300):
        out["time_tolerance_s"] = int(query_cfg.get("time_tolerance_seconds") or out.get("time_tolerance_s", 300))

    return out


def apply_adaptive_profile_config(config: Mapping[str, Any], *, dataset: str, args: Any = None) -> Dict[str, Any]:
    """Derive BASIL defaults from unlabeled telemetry profile.

    This is intentionally a default generator, not a hard override. If a legacy
    command-line parameter differs from its parser default, that explicit user
    value wins over the adaptive profile.
    """

    out: Dict[str, Any] = dict(config or {})
    if args is None or not bool(getattr(args, "adaptive_profile", False)):
        return out

    try:
        profile, adaptive = _cached_adaptive_profile_config(
            str(dataset),
            int(getattr(args, "adaptive_profile_max_files", 80) or 80),
            int(getattr(args, "adaptive_profile_max_rows_per_file", 200000) or 200000),
            float(getattr(args, "confidence_level", 0.95) or 0.95),
            str(getattr(args, "evidence_budget", "medium") or "medium"),
            float(getattr(args, "history_weight", 0.0) or 0.0),
            str(getattr(args, "llm_risk_tolerance", "conservative") or "conservative"),
        )
    except Exception as exc:
        try:
            setattr(args, "_adaptive_profile_error", str(exc))
        except Exception:
            pass
        return out

    mapping = adaptive.to_basil_mapping()

    def arg_is_default(name: str, default: Any) -> bool:
        return getattr(args, name, default) == default

    def set_if_default(config_key: str, arg_name: str, default: Any, cast: Any = None) -> None:
        if config_key not in mapping or not arg_is_default(arg_name, default):
            return
        value = mapping[config_key]
        if cast is not None:
            value = cast(value)
        out[config_key] = value
        try:
            setattr(args, arg_name, value)
        except Exception:
            pass

    set_if_default("top_k", "task3_top_k", 12, int)
    set_if_default("bucket_size_s", "basil_bucket_size", 300, int)
    set_if_default("max_lag_s", "basil_max_lag", 900, int)
    set_if_default("severity_floor", "basil_severity_floor", 0.0, float)
    set_if_default("historical_top_k", "basil_historical_top_k", 5, int)
    set_if_default("candidate_reason_top_r", "basil_candidate_reason_top_r", 3, int)
    set_if_default("time_candidate_top_k", "time_candidate_top_k", 8, int)
    set_if_default("evidence_score_gap_threshold", "evidence_score_gap_threshold", 0.15, float)
    set_if_default("evidence_top1_score_floor", "evidence_top1_score_floor", 0.30, float)
    set_if_default("gate_skip_gap_threshold", "gate_skip_gap_threshold", 0.30, float)

    try:
        setattr(args, "_adaptive_profile", profile.to_dict())
        setattr(args, "_adaptive_basil_config", adaptive.to_dict())
    except Exception:
        pass
    return out


@lru_cache(maxsize=16)
def _cached_adaptive_profile_config(
    dataset: str,
    max_files: int,
    max_rows_per_file: int,
    confidence_level: float,
    evidence_budget: str,
    history_weight: float,
    llm_risk_tolerance: str,
) -> Any:
    profile = DatasetProfiler(
        str(dataset),
        max_files=int(max_files),
        max_rows_per_file=int(max_rows_per_file),
    ).build()
    adaptive = build_adaptive_basil_config(
        profile,
        AdaptiveKnobs(
            confidence_level=float(confidence_level),
            evidence_budget=str(evidence_budget),
            history_weight=float(history_weight),
            llm_risk_tolerance=str(llm_risk_tolerance),
        ),
    )
    return profile, adaptive
