from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


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
