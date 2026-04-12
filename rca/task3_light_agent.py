import json
from typing import Any, Dict, List, Optional

from rca.api_router import get_chat_completion


def infer_task3_component_with_agent(
    instruction: str,
    candidates: List[str],
    evidence: Dict[str, Any],
    temperature: float = 0.0,
) -> str:
    cands = [str(c) for c in candidates if str(c).strip()]
    if not cands:
        return ""
    sys = (
        "You are an RCA component selector.\n"
        "Select exactly ONE root cause component from the provided candidates.\n"
        "Return ONLY JSON: {\"root_cause_component\": \"...\"}.\n"
        "No other keys.\n"
    )
    user = {
        "instruction": str(instruction),
        "candidates": cands,
        "evidence": evidence,
        "guidance": [
            "Prefer candidates with both high metric anomaly score and high trace duration anomaly (if available).",
            "If trace evidence is empty, rely on metric evidence and peak time alignment with the time window.",
            "Choose conservatively; do not invent new components.",
        ],
    }
    resp = get_chat_completion(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        temperature=float(temperature),
    )
    try:
        obj = json.loads(resp)
        comp = obj.get("root_cause_component")
        if isinstance(comp, str) and comp in cands:
            return comp
    except Exception:
        pass
    for c in cands:
        if isinstance(resp, str) and c in resp:
            return c
    return ""

