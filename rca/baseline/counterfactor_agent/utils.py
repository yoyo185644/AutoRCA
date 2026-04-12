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
    obj["1"]["root cause component"] = str(component or "")
    obj["1"]["root cause reason"] = str(reason or "")
    if str(occurrence_datetime or "").strip():
        obj["1"]["root cause occurrence datetime"] = str(occurrence_datetime)
    return json.dumps(obj, ensure_ascii=False)
