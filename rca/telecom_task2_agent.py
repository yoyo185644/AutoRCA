import json
from typing import Any, Dict, List, Optional

from rca.api_router import get_chat_completion


_ALLOWED_REASONS = ["CPU fault", "network delay", "network loss", "db connection limit", "db close"]


def infer_task2_reason_with_agent(
    instruction: str,
    evidence: Dict[str, Any],
    allowed_reasons: Optional[List[str]] = None,
    temperature: float = 0.0,
) -> str:
    allowed = allowed_reasons or _ALLOWED_REASONS
    sys = (
        "You are an RCA classifier.\n"
        "Choose exactly one root cause reason from the allowed list.\n"
        "Return ONLY a JSON object with schema: {\"root_cause_reason\": \"...\"}.\n"
        "Do not output any other keys.\n"
    )
    user = {
        "instruction": str(instruction),
        "allowed_reasons": allowed,
        "evidence": evidence,
        "selection_guidance": [
            "Prefer the reason whose category has the largest anomaly evidence (sum_ratio/count/distinct).",
            "If network delay vs network loss is unclear, use presence of timeout/reset/retrans/loss/drop to pick loss; otherwise delay.",
            "If db evidence is weak (no strong KPI hit), avoid db connection limit.",
        ],
    }
    resp = get_chat_completion(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        temperature=float(temperature),
    )
    try:
        obj = json.loads(resp)
        reason = obj.get("root_cause_reason")
        if isinstance(reason, str) and reason in allowed:
            return reason
    except Exception:
        pass
    for r in allowed:
        if isinstance(resp, str) and r in resp:
            return r
    return ""

