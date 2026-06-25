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


# Maps each reason label to its (stats_key, component_type) entries and metric_keys
# in the rich evidence format produced by infer_reason_with_evidence.
_REASON_EVIDENCE_CONFIG: Dict[str, Dict[str, Any]] = {
    "network loss":        {"stat_keys": [("network_loss",        "node")],                          "metric_keys": ["network_loss"]},
    "network delay":       {"stat_keys": [("network_delay",       "node")],                          "metric_keys": ["network_delay"]},
    "db connection limit": {"stat_keys": [("db_connection_limit", "service")],                       "metric_keys": ["db_connection_limit"]},
    "db close":            {"stat_keys": [("db_close",            "service")],                       "metric_keys": ["db_close"]},
    "CPU fault":           {"stat_keys": [("cpu_container",       "container"), ("cpu_node", "node")], "metric_keys": ["cpu_container", "cpu_node"]},
}

# Keyword groups used to classify flat metric names into reason/component-type buckets.
_FLAT_REASON_KEYWORDS: Dict[str, Tuple[List[str], str]] = {
    "network loss":        (["loss", "drop", "retrans", "timeout", "reset", "error"], "node"),
    "network delay":       (["rtt", "latency", "delay", "ping", "tnsping"],           "node"),
    "db connection limit": (["login", "tps", "session", "conn", "exec", "sql"],       "service"),
    "db close":            (["on_off", "close", "disconnect", "logout", "shutdown"],   "service"),
    "CPU fault":           (["cpu", "load", "iowait", "util", "proc"],                "container"),
}


def _format_reason_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize evidence into a structured per-reason summary for LLM consumption.

    Handles two upstream formats:
      - Rich format: produced by infer_reason_with_evidence (Telecom adapter path).
        Has 'scores' dict, 'stats' dict with per-type breach stats, and 'top_metrics'
        dict keyed by reason type.
      - Flat format: produced by predict_reason_universal (Market / Bank path).
        Has a flat 'top_metrics' list and an aggregate 'keyword_hits' dict.

    Output always contains:
      per_reason_evidence  – per-reason breach evidence with component type info
      trace_spread_ratio   – fraction of components showing trace anomaly
      trace_top_component_type – component type with strongest trace anomaly (if known)

    No deterministic conclusion is included: the LLM judges from evidence directly.
    """
    scores = evidence.get("scores")
    stats = evidence.get("stats")
    per_type_metrics = evidence.get("top_metrics") if isinstance(evidence.get("top_metrics"), dict) else None

    per_reason: Dict[str, Any] = {}

    if scores and stats and per_type_metrics:
        # --- Rich format ---
        for reason, cfg in _REASON_EVIDENCE_CONFIG.items():
            best_breach = 0.0
            best_stat: Dict[str, Any] = {}
            comp_types: List[str] = []
            all_metrics: List[str] = []

            for stat_key, comp_type in cfg["stat_keys"]:
                s = dict(stats.get(stat_key) or {})
                breach = float(s.get("sum_ratio", 0.0))
                if breach > best_breach:
                    best_breach = breach
                    best_stat = s
                if breach > 0:
                    comp_types.append(comp_type)

            for mk in cfg["metric_keys"]:
                all_metrics.extend(list(per_type_metrics.get(mk) or [])[:5])

            per_reason[reason] = {
                "score":                    round(float((scores or {}).get(reason, 0.0)), 3),
                "top_metrics":              list(dict.fromkeys(all_metrics))[:5],
                "breach_count":             int(best_stat.get("count", 0)),
                "distinct_metrics":         int(best_stat.get("distinct", 0)),
                "anomalous_component_types": list(dict.fromkeys(comp_types)),
            }
    else:
        # --- Flat format ---
        flat_metrics: List[str] = evidence.get("top_metrics") if isinstance(evidence.get("top_metrics"), list) else []  # type: ignore[assignment]

        for reason, (keywords, comp_type) in _FLAT_REASON_KEYWORDS.items():
            matched = [m for m in flat_metrics if any(k in m.lower() for k in keywords)][:5]
            per_reason[reason] = {
                "top_metrics":              matched,
                "breach_count":             len(matched),
                "anomalous_component_types": [comp_type] if matched else [],
            }

    spread = dict(evidence.get("trace_spread") or {})
    tc = dict(evidence.get("trace_component") or {})
    return {
        "per_reason_evidence":    per_reason,
        "trace_spread_ratio":     round(float(spread.get("affected_ratio", 0.0)), 3),
        "trace_top_component_type": str(tc.get("top_type", "")),
    }


_GUIDANCE: List[str] = [
    "Choose from candidates only. Do not invent new values.",
    "If evidence contains per_reason_evidence, compare each candidate's score, breach_count, distinct_metrics, and top_metrics.",
    "anomalous_component_types shows where the anomaly lives: 'node' aligns with network reasons, 'service' with db reasons, 'container' with CPU fault.",
    "trace_spread_ratio > 0.3 suggests widespread impact, favouring network-level root causes over CPU or db.",
    "trace_top_component_type indicates which component type shows the strongest trace anomaly — prefer reasons that align with it.",
    "If evidence contains component scores or combined_scores, prefer candidates with higher scores and earlier anomaly onset.",
    "Prefer the candidate whose evidence best matches both metric type and component type. When uncertain, keep the highest-scored candidate.",
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

    formatted_ev = (
        _format_reason_evidence(evidence)
        if str(output_field).strip().lower() == "root cause reason"
        else evidence
    )

    sys = (
        "You are a lightweight RCA selector.\n"
        "You must choose exactly one option from the provided candidates.\n"
        "Return ONLY a JSON object with schema: {\"value\": \"...\"}.\n"
        "No other keys. No markdown.\n"
    )
    user = {
        "instruction": str(instruction),
        "output_field": str(output_field),
        "candidates": cands,
        "evidence": formatted_ev,
        "guidance": _GUIDANCE,
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


def estimate_token_count(text: str) -> int:
    s = str(text or "")
    if not s:
        return 0
    try:
        return max(1, int(len(s.encode("utf-8")) / 4))
    except Exception:
        return max(1, int(len(s) / 4))


def call_json_only(
    *,
    system: str,
    user: str,
    temperature: float = 0.0,
) -> Tuple[Dict[str, Any], str, int, int, List[Dict[str, str]]]:
    messages = [{"role": "system", "content": str(system)}, {"role": "user", "content": str(user)}]
    raw = get_chat_completion(messages=messages, temperature=float(temperature))
    raw_s = raw if isinstance(raw, str) else str(raw)
    obj: Dict[str, Any] = {}
    try:
        obj = json.loads(raw_s)
    except Exception:
        start = raw_s.find("{")
        end = raw_s.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw_s[start : end + 1])
            except Exception:
                obj = {}
    in_tokens = estimate_token_count(system) + estimate_token_count(user)
    out_tokens = estimate_token_count(raw_s)
    return obj if isinstance(obj, dict) else {}, raw_s, int(in_tokens), int(out_tokens), messages
