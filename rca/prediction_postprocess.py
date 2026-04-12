import json
from typing import Any, Dict, List, Optional

from rca.market_solver import canonicalize_market_component, estimate_k_occurrence_datetimes_from_logs, estimate_k_occurrence_datetimes_from_trace, required_root_count
from rca.task_time_hybrid import estimate_occurrence_datetime_from_trace


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


def _market_times(dataset: str, instruction: str, k: int) -> List[str]:
    dts = estimate_k_occurrence_datetimes_from_logs(dataset=dataset, instruction=instruction, k=int(k))
    if not dts:
        dts = estimate_k_occurrence_datetimes_from_trace(dataset=dataset, instruction=instruction, k=int(k))
    return dts


def _single_time(dataset: str, instruction: str) -> str:
    dt, _ = estimate_occurrence_datetime_from_trace(dataset=dataset, instruction=instruction)
    return dt


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

    if dataset.startswith("Market/"):
        k = required_root_count(scoring_points)
    else:
        k = 1

    roots: List[Dict[str, Any]] = []
    for i in range(1, int(k) + 1):
        v = obj.get(str(i))
        if not isinstance(v, dict):
            v = obj.get("1") if isinstance(obj.get("1"), dict) else {}
        roots.append(dict(v))

    dts: Optional[List[str]] = None
    if need_time:
        if dataset.startswith("Market/") and int(k) > 1:
            dts = _market_times(dataset=dataset, instruction=instruction, k=int(k))
        elif dataset.startswith("Market/") and int(k) == 1:
            dts = _market_times(dataset=dataset, instruction=instruction, k=1)
        else:
            dts = [_single_time(dataset=dataset, instruction=instruction)]

    for i in range(int(k)):
        r = roots[i]
        if need_time:
            if dts:
                dt = dts[i] if i < len(dts) else dts[-1]
            else:
                dt = ""
            r.setdefault("root cause occurrence datetime", dt)
        if need_comp:
            r.setdefault("root cause component", r.get("root cause component") or r.get("root cause component".replace(" ", "_")) or "")
        if need_reason:
            r.setdefault("root cause reason", r.get("root cause reason") or r.get("root cause reason".replace(" ", "_")) or "")

        if dataset.startswith("Market/") and "root cause component" in r:
            r["root cause component"] = canonicalize_market_component(str(r.get("root cause component") or ""))

        roots[i] = r

    out: Dict[str, Any] = {}
    for i, r in enumerate(roots, start=1):
        rr: Dict[str, Any] = {}
        if "root cause occurrence datetime" in r:
            rr["root cause occurrence datetime"] = r.get("root cause occurrence datetime") or ""
        if "root cause component" in r:
            rr["root cause component"] = r.get("root cause component") or ""
        if "root cause reason" in r:
            rr["root cause reason"] = r.get("root cause reason") or ""
        out[str(i)] = rr
    return _safe_dump_obj(out)

