import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rca.api_router import get_chat_completion


@dataclass(frozen=True)
class LightAgentDecision:
    task_id: int
    output_field: str
    chosen: str
    candidates: List[str] = field(default_factory=list)
    used: bool = False
    explanation: str = ""
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _task_guidance(task_id: int) -> List[str]:
    t = int(task_id)
    if t == 2:
        return [
            "Choose a reason supported by the strongest anomaly evidence.",
            "Prefer labels that match explicit keywords in top metrics (timeout/loss/retrans -> loss; ping/rtt/latency -> delay).",
            "Avoid choosing a label if evidence is absent or contradictory.",
        ]
    if t == 3:
        return [
            "Choose a component supported by both metric anomalies and trace duration anomalies when available.",
            "Prefer candidates with higher combined scores and reasonable peak-time alignment within the window.",
            "Do not invent new components; pick from candidates only.",
        ]
    return [
        "Choose conservatively from candidates only.",
        "Prefer the option with stronger evidence signals.",
    ]


def choose_from_candidates(
    task_id: int,
    instruction: str,
    output_field: str,
    candidates: Sequence[str],
    evidence: Dict[str, Any],
    temperature: float = 0.0,
) -> LightAgentDecision:
    cands = [str(c) for c in candidates if str(c).strip()]
    if not cands:
        return LightAgentDecision(task_id=int(task_id), output_field=str(output_field), chosen="", candidates=[], used=False)

    sys = (
        "You are a lightweight RCA selector.\n"
        "You must choose exactly one option from the provided candidates.\n"
        "Return ONLY a JSON object with schema: {\"value\": \"...\"}.\n"
        "No other keys. No markdown.\n"
    )
    user = {
        "task_id": int(task_id),
        "instruction": str(instruction),
        "output_field": str(output_field),
        "candidates": cands,
        "evidence": evidence,
        "guidance": _task_guidance(int(task_id)),
    }
    resp = get_chat_completion(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        temperature=float(temperature),
    )
    chosen = ""
    try:
        obj = json.loads(resp)
        v = obj.get("value")
        if isinstance(v, str) and v in cands:
            chosen = v
    except Exception:
        chosen = ""
    if not chosen:
        for c in cands:
            if isinstance(resp, str) and c in resp:
                chosen = c
                break
    return LightAgentDecision(
        task_id=int(task_id),
        output_field=str(output_field),
        chosen=str(chosen),
        candidates=cands,
        used=True,
        explanation="llm_choice",
        raw_response=resp if isinstance(resp, str) else str(resp),
    )

