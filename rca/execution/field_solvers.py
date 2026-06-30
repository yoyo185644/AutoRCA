from __future__ import annotations

import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rca.execution.light_agent import choose_from_candidates
from rca.execution.telemetry_solver import (
    estimate_k_occurrence_datetimes_from_trace,
)
from rca.adapter.utils.reason_candidates import predict_reason_universal
from rca.adapter.utils.component_candidates import build_component_candidates
from rca.adapter.utils.time_candidates import (
    estimate_occurrence_datetime_from_trace,
    estimate_occurrence_datetime_candidates,
    select_time_candidate,
)
from rca.execution.fast_solver import (
    _norm_name,
    _parse_time_window,
    _to_tz_timestamp_ms,
    _trace_component_strength,
    _trace_spread_stats,
    evidence_supports_reason,
    infer_reason_with_evidence as _infer_reason_with_evidence_raw,
 )
from rca.index.config import basil_config_from_args, basil_enabled
from rca.index.extractor import get_cached_header, get_cached_metric_baseline
from rca.adapter.registry import get_dataset_adapter


_REASON_INFERENCE_CACHE: Dict[Tuple[str, str, str], Tuple[str, Dict[str, Any]]] = {}


def _get_reason_cache_key(metric_dir: str, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[str, str, str]:
    return (str(metric_dir), str(start.isoformat()), str(end.isoformat()))


def _load_metric_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    header = get_cached_header(path)
    usecols = ["timestamp", "cmdb_id", "name", "value"]
    if header and all(c in header for c in usecols):
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = _to_tz_timestamp_ms(df["timestamp"])
    return df


def _thresholds_from_baseline(baseline: pd.DataFrame, metric_path: str) -> pd.DataFrame:
    if baseline is None or baseline.empty:
        return pd.DataFrame(columns=["cmdb_id", "name", "p95"])
    file_name = os.path.basename(metric_path)
    b = baseline[baseline["_file"].astype(str) == str(file_name)].copy()
    if b.empty:
        return pd.DataFrame(columns=["cmdb_id", "name", "p95"])
    b = b.rename(columns={"component": "cmdb_id", "metric": "name"})
    for c in ["cmdb_id", "name"]:
        if c in b.columns:
            b[c] = b[c].astype(str)
    b["p95"] = pd.to_numeric(b.get("p95"), errors="coerce")
    b = b.dropna(subset=["p95"])
    if b.empty:
        return pd.DataFrame(columns=["cmdb_id", "name", "p95"])
    return b[["cmdb_id", "name", "p95"]]


def _breach_stats_with_baseline(
    *,
    metric_path: str,
    baseline: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    any_keywords: List[str],
    strong_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    empty = {"sum_ratio": 0.0, "count": 0, "distinct": 0, "strong_hit": False, "strong_count": 0, "top_metrics": []}
    if not os.path.exists(metric_path):
        return dict(empty)

    df = _load_metric_csv(metric_path)
    if df.empty:
        return dict(empty)
    if not all(c in df.columns for c in ["timestamp", "cmdb_id", "name", "value"]):
        return dict(empty)

    df = df.copy()
    df["cmdb_id"] = df["cmdb_id"].astype(str)
    df["name"] = df["name"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "cmdb_id", "name", "value"])
    if df.empty:
        return dict(empty)

    df["name_norm"] = df["name"].astype(str).map(_norm_name)
    any_norm = [_norm_name(k) for k in any_keywords if str(k).strip()]
    strong_norm = [_norm_name(k) for k in (strong_keywords or []) if str(k).strip()]

    mask = False
    for kw in any_norm:
        mask = mask | df["name_norm"].str.contains(kw, na=False)
    if not mask.any():
        return dict(empty)

    thresholds = _thresholds_from_baseline(baseline, metric_path)
    if thresholds.empty:
        return dict(empty)

    win = df[(df["timestamp"] >= start) & (df["timestamp"] <= end) & (mask)].copy()
    if win.empty:
        return dict(empty)
    win = win.merge(thresholds, on=["cmdb_id", "name"], how="left").dropna(subset=["p95"]).copy()
    if win.empty:
        return dict(empty)

    denom = win["p95"].abs().where(win["p95"].abs() > 1e-6, 1.0)
    win["breach_ratio"] = (win["value"] - win["p95"]) / denom
    win["breach_ratio"] = win["breach_ratio"].clip(lower=0.0, upper=10.0)
    win = win[win["breach_ratio"] > 0].copy()
    if win.empty:
        return dict(empty)

    strong_hit = False
    strong_count = 0
    if strong_norm:
        for k in strong_norm:
            if win["name_norm"].str.contains(k, na=False).any():
                strong_count += 1
        strong_hit = strong_count > 0

    top_metrics = (
        win.groupby("name", as_index=False)["breach_ratio"]
        .sum()
        .sort_values("breach_ratio", ascending=False)
        .head(10)["name"]
        .astype(str)
        .tolist()
    )
    return {
        "sum_ratio": float(win["breach_ratio"].sum()),
        "count": int(len(win)),
        "distinct": int(win["name"].nunique()),
        "strong_hit": bool(strong_hit),
        "strong_count": int(strong_count),
        "top_metrics": top_metrics,
    }


def infer_reason_with_evidence(instruction: str) -> Tuple[str, Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    t0 = time.time()
    metric_dir = ""
    cache_hit = False
    try:
        date_dir, start, end = _parse_time_window(instruction)
        metric_dir = os.path.join("dataset", "Telecom", "telemetry", date_dir, "metric")
        trace_path = os.path.join("dataset", "Telecom", "telemetry", date_dir, "trace", "trace_span.csv")
    except Exception:
        reason, ev = _infer_reason_with_evidence_raw(instruction)
        logger.info(
            "reason_inference completed | metric_dir=%s | cache_hit=%s | elapsed_ms=%d",
            str(metric_dir),
            bool(False),
            int((time.time() - t0) * 1000),
        )
        return reason, ev

    cache_key = _get_reason_cache_key(metric_dir, start, end)
    if cache_key in _REASON_INFERENCE_CACHE:
        cache_hit = True
        reason, ev = _REASON_INFERENCE_CACHE[cache_key]
        logger.info("reason_inference cache hit | metric_dir=%s", metric_dir)
        logger.info(
            "reason_inference completed | metric_dir=%s | cache_hit=%s | elapsed_ms=%d",
            str(metric_dir),
            bool(True),
            int((time.time() - t0) * 1000),
        )
        return reason, ev

    baseline = get_cached_metric_baseline(metric_dir)

    net_loss_kws = ["loss", "drop", "retrans", "timeout", "reset", "error"]
    net_delay_kws = ["tnsping_result_time", "tnsping", "ping", "rtt", "latency", "delay"]
    db_conn_kws = ["login_per_sec", "login", "tps_per_sec", "tps", "exec_per_sec", "exec", "sess_active", "session"]
    db_conn_strong = ["login_per_sec", "tps_per_sec", "exec_per_sec"]
    db_close_kws = ["on_off_state", "on_off", "logout", "disconnect", "hang", "shutdown", "restart", "close", "reset"]
    cpu_kws = ["proc_used_pct", "proc_user_used_pct", "cpu", "load", "iowait", "util"]

    net_path = os.path.join(metric_dir, "metric_node.csv")
    mw_path = os.path.join(metric_dir, "metric_middleware.csv")
    svc_path = os.path.join(metric_dir, "metric_service.csv")
    cont_path = os.path.join(metric_dir, "metric_container.csv")

    net_loss_n = _breach_stats_with_baseline(metric_path=net_path, baseline=baseline, start=start, end=end, any_keywords=net_loss_kws)
    net_delay_n = _breach_stats_with_baseline(metric_path=net_path, baseline=baseline, start=start, end=end, any_keywords=net_delay_kws)
    net_loss_m = _breach_stats_with_baseline(metric_path=mw_path, baseline=baseline, start=start, end=end, any_keywords=net_loss_kws)
    net_delay_m = _breach_stats_with_baseline(metric_path=mw_path, baseline=baseline, start=start, end=end, any_keywords=net_delay_kws)
    net_loss = max(
        [net_loss_n, net_loss_m],
        key=lambda d: (float(d.get("sum_ratio", 0.0)), int(d.get("distinct", 0)), int(d.get("count", 0))),
    )
    net_delay = max(
        [net_delay_n, net_delay_m],
        key=lambda d: (float(d.get("sum_ratio", 0.0)), int(d.get("distinct", 0)), int(d.get("count", 0))),
    )
    db_conn = _breach_stats_with_baseline(
        metric_path=svc_path,
        baseline=baseline,
        start=start,
        end=end,
        any_keywords=db_conn_kws,
        strong_keywords=db_conn_strong,
    )
    db_close = _breach_stats_with_baseline(metric_path=svc_path, baseline=baseline, start=start, end=end, any_keywords=db_close_kws)
    cpu_c = _breach_stats_with_baseline(metric_path=cont_path, baseline=baseline, start=start, end=end, any_keywords=cpu_kws)
    cpu_n = _breach_stats_with_baseline(metric_path=net_path, baseline=baseline, start=start, end=end, any_keywords=cpu_kws)

    def score(stat: Dict[str, Any], require_distinct: int = 1, weak_penalty: float = 1.0) -> float:
        if int(stat.get("distinct", 0)) < require_distinct:
            return 0.0
        s_sum = float(stat.get("sum_ratio", 0.0))
        s_cnt = float(stat.get("count", 0))
        s_dis = float(stat.get("distinct", 0))
        mean_ratio = s_sum / max(1.0, s_cnt)
        s = 2.0 * math.log1p(max(0.0, mean_ratio)) + 0.35 * math.log1p(max(0.0, s_cnt)) + 0.9 * math.log1p(max(0.0, s_dis))
        return float(s * weak_penalty)

    s_net_loss = max(score(net_loss_n, require_distinct=1), score(net_loss_m, require_distinct=1))
    s_net_delay = max(score(net_delay_n, require_distinct=1), score(net_delay_m, require_distinct=1))
    s_db_conn = score(db_conn, require_distinct=2, weak_penalty=1.0 if db_conn.get("strong_hit") else 0.25)
    s_db_close = score(db_close, require_distinct=2)
    s_cpu = max(score(cpu_c, require_distinct=1), score(cpu_n, require_distinct=1))

    scored = [
        ("network loss", s_net_loss, net_loss),
        ("network delay", s_net_delay, net_delay),
        ("db connection limit", s_db_conn, db_conn),
        ("db close", s_db_close, db_close),
        ("CPU fault", s_cpu, {"container": cpu_c, "node": cpu_n}),
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_reason = scored[0][0] if scored else ""
    best_score = float(scored[0][1]) if scored else 0.0
    second_score = float(scored[1][1]) if len(scored) > 1 else 0.0
    gap = best_score - second_score
    evidence: Dict[str, Any] = {
        "time_window": {"date_dir": date_dir, "start": start.strftime("%Y-%m-%d %H:%M:%S"), "end": end.strftime("%Y-%m-%d %H:%M:%S")},
        "scores": {r: float(s) for r, s, _ in scored},
        "gap": float(gap),
        "best": {"reason": best_reason, "score": float(best_score)},
        "stats": {
            "network_loss": net_loss,
            "network_delay": net_delay,
            "db_connection_limit": db_conn,
            "db_close": db_close,
            "cpu_container": cpu_c,
            "cpu_node": cpu_n,
        },
        "top_metrics": {
            "network_loss": net_loss.get("top_metrics", []),
            "network_delay": net_delay.get("top_metrics", []),
            "db_connection_limit": db_conn.get("top_metrics", []),
            "db_close": db_close.get("top_metrics", []),
            "cpu_container": cpu_c.get("top_metrics", []),
            "cpu_node": cpu_n.get("top_metrics", []),
        },
        "keyword_hits": {
            "network_delay": 0,
            "network_loss": 0,
            "db_connection_limit": int(db_conn.get("strong_count", 0) or 0),
            "db_close": 0,
            "cpu": 0,
        },
        "trace_spread": {},
    }

    def _count_hits(metrics: List[str], tokens: List[str]) -> int:
        blob = " ".join([str(x).lower() for x in metrics])
        return int(sum(1 for t in tokens if t and t in blob))

    evidence["keyword_hits"]["network_delay"] = _count_hits(evidence["top_metrics"]["network_delay"], ["tnsping", "ping", "rtt", "latency", "delay"])
    evidence["keyword_hits"]["network_loss"] = _count_hits(evidence["top_metrics"]["network_loss"], ["timeout", "retrans", "loss", "drop", "reset", "error"])
    evidence["keyword_hits"]["db_close"] = _count_hits(evidence["top_metrics"]["db_close"], ["close", "disconnect", "logout", "onoff", "shutdown", "restart"])
    evidence["keyword_hits"]["cpu"] = _count_hits(evidence["top_metrics"]["cpu_node"] + evidence["top_metrics"]["cpu_container"], ["cpu", "load", "iowait", "util"])

    evidence["trace_spread"] = _trace_spread_stats(trace_path, start, end)
    evidence["trace_component"] = _trace_component_strength(trace_path, start, end)

    spread = evidence["trace_spread"]
    spread_ratio = float(spread.get("affected_ratio", 0.0) or 0.0)
    hits = evidence["keyword_hits"]
    scores = evidence["scores"]
    tc = evidence["trace_component"]
    db_max = float(tc.get("db_max_ratio", 0.0) or 0.0)
    node_max = float(tc.get("node_max_ratio", 0.0) or 0.0)
    cont_max = float(tc.get("container_max_ratio", 0.0) or 0.0)
    top_type = str(tc.get("top_type") or "")
    if spread_ratio >= 0.35 and (int(hits.get("network_loss", 0)) > 0 or int(hits.get("network_delay", 0)) > 0):
        if int(hits.get("network_loss", 0)) >= int(hits.get("network_delay", 0)):
            scores["network loss"] = float(scores.get("network loss", 0.0) + 1.0)
        else:
            scores["network delay"] = float(scores.get("network delay", 0.0) + 1.0)
        scores["CPU fault"] = float(max(0.0, scores.get("CPU fault", 0.0) - 0.4))
        scores["db connection limit"] = float(max(0.0, scores.get("db connection limit", 0.0) - 0.3))
        scores["db close"] = float(max(0.0, scores.get("db close", 0.0) - 0.3))
    if spread_ratio <= 0.15 and float(scores.get("CPU fault", 0.0)) > 0:
        scores["CPU fault"] = float(scores.get("CPU fault", 0.0) + 0.2)
    if db_max >= 1.6 or top_type == "db":
        scores["db connection limit"] = float(scores.get("db connection limit", 0.0) + 0.4)
        scores["db close"] = float(scores.get("db close", 0.0) + 0.4)
    if db_max <= 1.25 and top_type in ("node", "container"):
        scores["db connection limit"] = float(max(0.0, scores.get("db connection limit", 0.0) - 0.6))
        scores["db close"] = float(max(0.0, scores.get("db close", 0.0) - 0.6))
    if (max(node_max, cont_max) >= 1.6) and spread_ratio <= 0.25 and int(hits.get("cpu", 0)) >= 1:
        scores["CPU fault"] = float(scores.get("CPU fault", 0.0) + 0.3)
        scores["network loss"] = float(max(0.0, scores.get("network loss", 0.0) - 0.2))
        scores["network delay"] = float(max(0.0, scores.get("network delay", 0.0) - 0.2))
    evidence["scores"] = scores
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if ranked:
        best_reason = ranked[0][0]
        evidence["best"] = {"reason": best_reason, "score": float(ranked[0][1])}
        evidence["gap"] = float(ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0))

    _REASON_INFERENCE_CACHE[cache_key] = (best_reason, evidence)
    logger.info("reason_inference cache miss | metric_dir=%s | stored", metric_dir)
    logger.info(
        "reason_inference completed | metric_dir=%s | cache_hit=%s | elapsed_ms=%d",
        str(metric_dir),
        bool(cache_hit),
        int((time.time() - t0) * 1000),
    )
    return best_reason, evidence


def solve_time(*, dataset: str, instruction: str, need_time: bool) -> Tuple[str, Optional[int], Any]:
    if not need_time:
        return "", None, None

    # scoring_points contains evaluation labels; time inference must not use it.
    # Reason-aware routing is supplied by indexed evidence / adapter taxonomy.
    reasons: List[str] = []

    try:
        # Direction 1+2+3: multi-signal fusion — trace onset/peak + reason-aware metric onset/peak
        candidates = estimate_occurrence_datetime_candidates(
            dataset=dataset,
            instruction=instruction,
            reasons=[],
            top_k=16,
        )
        # Direction 4: ±1min neighbors are already added inside _reason_metric_time_candidates
        # select_time_candidate applies consensus bonus + reason/component alignment scoring
        dt, _ = select_time_candidate(candidates, reason="")
        if not str(dt or "").strip():
            raise ValueError("empty dt")
    except Exception:
        # Fallback to legacy single-signal trace method
        dt, _ = estimate_occurrence_datetime_from_trace(dataset=dataset, instruction=instruction)

    try:
        center_time_s = int(pd.Timestamp(dt, tz="Asia/Shanghai").tz_convert("UTC").timestamp())
    except Exception:
        center_time_s = None
    return str(dt or ""), center_time_s, candidates if "candidates" in dir() else None


def _reason_candidates_from_evidence(evidence: Dict[str, Any]) -> List[str]:
    """Extract reason candidates from evidence regardless of format.

    Rich format (Telecom): evidence has a 'scores' dict keyed by reason label.
    Flat format (Market/Bank): evidence has a 'candidate_reasons' list.
    Falls back to an empty list; scoring_points contains evaluation labels.
    """
    scores = evidence.get("scores")
    if isinstance(scores, dict) and scores:
        return [str(k) for k in scores.keys() if str(k).strip()]
    cands = evidence.get("candidate_reasons")
    if isinstance(cands, list) and cands:
        return [str(c) for c in cands if str(c).strip()]
    return []


def solve_reason(
    *,
    spec: Any,
    dataset: str,
    instruction: str,
    args: Any,
    need_reason: bool,
) -> Tuple[str, str, bool, Any]:
    if not need_reason:
        return "", "", False, None

    # --- Step 1: build evidence (adapter-specific or universal) ---
    adapter = get_dataset_adapter(dataset)
    det_reason: str = ""
    ev2: Any = None
    if adapter:
        ar = adapter.solve_reason(
            spec=spec,
            instruction=instruction,
            scoring_points="",
            args=args,
            need_reason=need_reason,
        )
        if ar is not None:
            det_reason, _other, _ready, ev2 = ar

    if ev2 is None:
        det_reason, ev2 = predict_reason_universal(
            dataset=dataset,
            instruction=instruction,
            scoring_points="",
        )

    # --- Step 2: unified LLM rerank (same for all datasets) ---
    if (
        bool(getattr(args, "light_agent", False))
        and not bool(getattr(args, "deterministic_only", False))
        and isinstance(ev2, dict)
    ):
        candidates = _reason_candidates_from_evidence(ev2)
        if candidates:
            dec = choose_from_candidates(
                task_id=2,
                instruction=instruction,
                output_field="root cause reason",
                candidates=candidates,
                evidence=ev2,
                temperature=float(getattr(args, "light_agent_temperature", 0.0)),
            )
            if dec.chosen:
                det_reason = dec.chosen

    return "", str(det_reason or ""), True, ev2


def solve_component(
    *,
    spec: Any,
    dataset: str,
    instruction: str,
    args: Any,
    need_comp: bool,
    need_time: bool,
    need_reason: bool,
    center_time_s: Optional[int],
    det_reason: str,
    det_reason_ready: bool,
    det_reason_other: str,
    ev2: Any,
    light_agent_tasks: set[int],
    no_llm_fallback: bool,
) -> Tuple[str, Optional[str], Any, str]:
    if not need_comp:
        return "", None, None, str(det_reason or "")

    adapter = get_dataset_adapter(dataset)
    if adapter:
        _delta_cfg = adapter.get_component_delta_s(spec=spec, need_comp=need_comp, need_time=need_time, center_time_s=center_time_s)
        if isinstance(_delta_cfg, tuple) and len(_delta_cfg) == 2:
            delta_s, center_time_s_for_comp = int(_delta_cfg[0]), _delta_cfg[1]
        else:
            delta_s, center_time_s_for_comp = int(_delta_cfg), center_time_s
    else:
        delta_s, center_time_s_for_comp = 300, center_time_s

    top_k = int(getattr(args, "task3_top_k", 12))
    ev3 = build_component_candidates(
        dataset=dataset,
        instruction=instruction,
        top_k=top_k,
        center_time_s=center_time_s_for_comp,
        delta_s=delta_s,
        use_basil=basil_enabled(args),
        basil_config=basil_config_from_args(args, top_k, required_fields=getattr(spec, "required_fields", None), task_id=getattr(spec, "task_id", None)),
        basil_reasons=[x for x in [det_reason, det_reason_other] if str(x or "").strip()] if need_reason else [],
    )

    def _ctype(c: str) -> str:
        c = str(c or "")
        if c.startswith("db_"):
            return "db"
        if c.startswith(("os_", "node-")):
            return "node"
        if c.startswith(("docker_", "container")):
            return "container"
        if "-" in c:
            return "service"
        return "other"

    def _rtype(r: str) -> str:
        r = str(r or "")
        if r in ("db close", "db connection limit"):
            return "db"
        if r == "CPU fault":
            return "container"
        if r in ("network delay", "network loss"):
            return "node"
        return ""

    comp = ev3.top1
    det_reason_out = str(det_reason or "")
    if adapter and need_reason and det_reason_ready:
        fn = getattr(adapter, "solve_component_joint_scoring", None)
        if callable(fn):
            pair = fn(ev2=ev2, ev3=ev3, det_reason=det_reason, det_reason_ready=det_reason_ready)
            if pair is not None:
                det_reason_out, comp = pair

    top1_score = float(ev3.combined_scores.get(comp, 0.0))
    desired = ""
    desired_ratio = 0.0
    if det_reason_out in ("db close", "db connection limit"):
        desired = "db"
        desired_ratio = 0.62
    elif det_reason_out == "CPU fault":
        desired = "container"
        desired_ratio = 0.66
    elif det_reason_out in ("network delay", "network loss"):
        desired = "node"
        desired_ratio = 0.62
    if desired and top1_score > 1e-12:
        pool = [c for c in ev3.combined_scores.keys() if _ctype(c) == desired]
        if pool:
            best_c = max(pool, key=lambda c: float(ev3.combined_scores.get(c, 0.0)))
            best_s = float(ev3.combined_scores.get(best_c, 0.0))
            if best_s >= desired_ratio * top1_score:
                comp = best_c

    cand = list(ev3.candidates or [])
    filtered = [c for c in cand if desired and _ctype(c) == desired] if desired else []
    if adapter:
        allow_light_agent = adapter.should_use_light_agent(
            spec=spec, need_comp=need_comp, ev3_gap_norm=float(ev3.gap_norm),
            no_llm_fallback=no_llm_fallback, args=args,
        )
    else:
        allow_light_agent = (int(getattr(spec, "task_id", 0)) in light_agent_tasks)
    gap_thr = float(getattr(args, "task3_agent_gap", 0.15))
    use_agent = bool(args.light_agent) and (float(ev3.gap_norm) < gap_thr) and (not no_llm_fallback) and allow_light_agent
    if use_agent:
        use_cands = filtered if filtered else cand
        dec = choose_from_candidates(
            task_id=3,
            instruction=instruction,
            output_field="root cause component",
            candidates=use_cands,
            evidence=ev3.to_dict(),
            temperature=float(args.light_agent_temperature),
        )
        if dec.chosen:
            comp = dec.chosen

    dt_override = None
    if adapter:
        dt_override = adapter.get_time_dt_override(need_time=need_time, comp=comp, ev3=ev3)

    _ = det_reason_other
    return str(comp or ""), dt_override, ev3, str(det_reason_out or "")


def assemble_prediction(
    *,
    spec: Any,
    dataset: str,
    instruction: str,
    args: Any,
    need_time: bool,
    need_comp: bool,
    need_reason: bool,
    k_fail: int,
    root_dt: str,
    root_comp: str,
    root_reason: str,
    det_reason: str,
    det_reason_other: str,
    det_reason_ready: bool,
) -> dict:
    obj: dict = {"1": {}}
    if need_time and str(root_dt or "").strip():
        obj["1"]["root cause occurrence datetime"] = str(root_dt)
    if need_comp and str(root_comp or "").strip():
        obj["1"]["root cause component"] = str(root_comp)
    if need_reason and str(root_reason or "").strip():
        obj["1"]["root cause reason"] = str(root_reason)

    if int(k_fail) <= 1:
        return obj

    roots = []
    dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k_fail))
    reasons = []
    for i in range(int(k_fail)):
        r: dict = {}
        dt_i = dts[i] if i < len(dts) else dts[-1]
        if need_time:
            r["root cause occurrence datetime"] = dt_i
            try:
                ct_s = int(pd.Timestamp(dt_i, tz="Asia/Shanghai").tz_convert("UTC").timestamp())
            except Exception:
                ct_s = None
        else:
            ct_s = None
        if need_comp:
            top_k = int(getattr(args, "task3_top_k", 12))
            ev3_i = build_component_candidates(
                dataset=dataset,
                instruction=instruction,
                top_k=top_k,
                center_time_s=ct_s,
                use_basil=basil_enabled(args),
                basil_config=basil_config_from_args(args, top_k, required_fields=getattr(spec, "required_fields", None), task_id=getattr(spec, "task_id", None)),
                basil_reasons=[det_reason] if (need_reason and str(det_reason or "").strip()) else [],
            )
            comp_i = ev3_i.top1
            if any(x.get("root cause component") == comp_i for x in roots):
                ranked = sorted(
                    list(ev3_i.candidates or []),
                    key=lambda c: float(ev3_i.combined_scores.get(c, 0.0)),
                    reverse=True,
                )
                for c in ranked:
                    if not any(x.get("root cause component") == c for x in roots):
                        comp_i = c
                        break
            r["root cause component"] = comp_i
        if need_reason:
            if len(reasons) >= int(k_fail):
                r["root cause reason"] = reasons[i]
            elif dataset != "Telecom" and det_reason_ready:
                r["root cause reason"] = det_reason_other
            else:
                r["root cause reason"] = det_reason
        roots.append(r)

    obj = {}
    for j, r in enumerate(roots, start=1):
        rr = {}
        if "root cause occurrence datetime" in r:
            rr["root cause occurrence datetime"] = r["root cause occurrence datetime"]
        if "root cause component" in r:
            rr["root cause component"] = r["root cause component"]
        if "root cause reason" in r:
            rr["root cause reason"] = r["root cause reason"]
        obj[str(j)] = rr
    return obj
