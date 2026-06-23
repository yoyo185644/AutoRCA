from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from rca.common.config import load_dataset_config


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
