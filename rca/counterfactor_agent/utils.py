import json
import re
from typing import Any, Dict, List


def safe_json_loads(text: str, default: Any):
    if not text:
        return default

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.S | re.I)
    for block in fenced:
        try:
            return json.loads(block)
        except Exception:
            continue

    brace_match = re.search(r"(\{.*\})", text, flags=re.S)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except Exception:
            pass

    bracket_match = re.search(r"(\[.*\])", text, flags=re.S)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(1))
        except Exception:
            pass

    return default


def normalize_score(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v < 0:
            return 0.0
        if v > 1:
            return 1.0
        return v
    except Exception:
        return default


def ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def deduplicate_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        s = str(item)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def format_prediction_json(
    component: str,
    reason: str,
    confidence: float,
    reasoning: str,
    occurrence_datetime: str = "",
) -> str:
    obj: Dict[str, Any] = {"1": {}}
    if str(occurrence_datetime or "").strip():
        obj["1"]["root cause occurrence datetime"] = str(occurrence_datetime)
    obj["1"]["root cause component"] = str(component or "")
    obj["1"]["root cause reason"] = str(reason or "")
    return json.dumps(obj, ensure_ascii=False)


def format_prediction_json_multi(roots: List[Dict[str, Any]]) -> str:
    obj: Dict[str, Any] = {}
    for i, r in enumerate(list(roots or []), start=1):
        entry: Dict[str, Any] = {}
        if isinstance(r, dict):
            dt = str(r.get("occurrence_datetime", "") or r.get("root cause occurrence datetime", "") or "").strip()
            if dt:
                entry["root cause occurrence datetime"] = dt
            entry["root cause component"] = str(r.get("component", "") or r.get("root cause component", "") or "")
            entry["root cause reason"] = str(r.get("reason", "") or r.get("root cause reason", "") or "")
        else:
            entry["root cause component"] = str(r)
            entry["root cause reason"] = ""
        obj[str(i)] = entry
    return json.dumps(obj, ensure_ascii=False)


def normalize_prediction_json_for_eval(prediction_json: str, k: int = 1) -> str:
    obj = safe_json_loads(str(prediction_json or ""), default={})
    if not isinstance(obj, dict):
        obj = {}
    roots: List[Dict[str, Any]] = []
    kk = max(1, int(k) if int(k) >= 1 else 1)
    for i in range(1, kk + 1):
        v = obj.get(str(i))
        if not isinstance(v, dict):
            v = obj.get(i) if isinstance(obj.get(i), dict) else {}
        roots.append(
            {
                "occurrence_datetime": str(v.get("root cause occurrence datetime", "") or v.get("occurrence_datetime", "") or ""),
                "component": str(v.get("root cause component", "") or v.get("component", "") or ""),
                "reason": str(v.get("root cause reason", "") or v.get("reason", "") or ""),
            }
        )
    return format_prediction_json_multi(roots)
