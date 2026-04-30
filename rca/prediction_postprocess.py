import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from rca.telemetry_solver import canonicalize_component, estimate_k_occurrence_datetimes_from_logs, estimate_k_occurrence_datetimes_from_trace, required_root_count
from rca.time_candidates import estimate_occurrence_datetime_from_trace


def _safe_load_obj(prediction: str) -> Dict[str, Any]:
    s = str(prediction or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_dump_obj(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _need_fields(task_id: int):
    need_time = int(task_id) in (1, 4, 5, 7)
    need_comp = int(task_id) in (3, 5, 6, 7)
    need_reason = int(task_id) in (2, 4, 6, 7)
    return need_time, need_comp, need_reason


def _log_trace_times(dataset: str, instruction: str, k: int) -> List[str]:
    dts = estimate_k_occurrence_datetimes_from_logs(dataset=dataset, instruction=instruction, k=int(k))
    if not dts:
        dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k))
    return dts


def _single_time(dataset: str, instruction: str) -> str:
    dt, _ = estimate_occurrence_datetime_from_trace(dataset=dataset, instruction=instruction)
    return dt


_DT_FMT = "%Y-%m-%d %H:%M:%S"


def _normalize_datetime_str(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    s = s.replace("/", "-").replace("T", " ").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}$", s):
        s = s + ":00"
    has_tz = bool(re.search(r"(Z|[+-]\d{2}:?\d{2})$", s))
    try:
        if has_tz:
            ts = pd.to_datetime(s, errors="coerce", utc=True)
            if pd.isna(ts):
                return ""
            ts = ts.tz_convert("Asia/Shanghai")
        else:
            ts = pd.to_datetime(s, errors="coerce")
            if pd.isna(ts):
                return ""
            if getattr(ts, "tz", None) is None:
                ts = ts.tz_localize("Asia/Shanghai")
            else:
                ts = ts.tz_convert("Asia/Shanghai")
        return ts.strftime(_DT_FMT)
    except Exception:
        return ""


def normalize_prediction_for_eval(
    prediction: str,
    dataset: str,
    instruction: str,
    scoring_points: str,
    task_id: int,
) -> str:
    need_time, need_comp, need_reason = _need_fields(int(task_id))
    if not (need_time or need_comp or need_reason):
        return prediction

    obj = _safe_load_obj(prediction)
    if not obj:
        obj = {"1": {}}

    k = required_root_count(scoring_points)

    roots: List[Dict[str, Any]] = []
    for i in range(1, int(k) + 1):
        v = obj.get(str(i))
        if not isinstance(v, dict):
            v = obj.get("1") if isinstance(obj.get("1"), dict) else {}
        roots.append(dict(v))

    dts: Optional[List[str]] = None
    if need_time:
        if dataset.startswith("Market/") and int(k) >= 1:
            dts = _log_trace_times(dataset=dataset, instruction=instruction, k=int(k))
        else:
            if int(k) > 1:
                dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k))
            else:
                dts = [_single_time(dataset=dataset, instruction=instruction)]

    for i in range(int(k)):
        r = roots[i]
        if need_time:
            if dts:
                dt = dts[i] if i < len(dts) else dts[-1]
            else:
                dt = ""
            if not str(r.get("root cause occurrence datetime") or "").strip():
                r["root cause occurrence datetime"] = dt
        if need_comp:
            key2 = "root cause component".replace(" ", "_")
            v = r.get("root cause component") or r.get(key2) or ""
            if not str(r.get("root cause component") or "").strip():
                r["root cause component"] = str(v or "")
        if need_reason:
            key2 = "root cause reason".replace(" ", "_")
            v = r.get("root cause reason") or r.get(key2) or ""
            if not str(r.get("root cause reason") or "").strip():
                r["root cause reason"] = str(v or "")

        if dataset.startswith("Market/") and "root cause component" in r:
            r["root cause component"] = canonicalize_component(str(r.get("root cause component") or ""))

        roots[i] = r

    out: Dict[str, Any] = {}
    for i, r in enumerate(roots, start=1):
        rr: Dict[str, Any] = {}
        if "root cause occurrence datetime" in r:
            rr["root cause occurrence datetime"] = _normalize_datetime_str(r.get("root cause occurrence datetime") or "")
        if "root cause component" in r:
            rr["root cause component"] = r.get("root cause component") or ""
        if "root cause reason" in r:
            rr["root cause reason"] = r.get("root cause reason") or ""
        out[str(i)] = rr
    return _safe_dump_obj(out)
