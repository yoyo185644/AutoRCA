import re
from typing import List


def extract_task2_reason_candidates(scoring_points: str) -> List[str]:
    s = str(scoring_points or "")
    cands = re.findall(r"root cause reason is ([^\n\r]+)", s, flags=re.IGNORECASE)
    out = []
    for c in cands:
        v = str(c).strip().strip(".").strip()
        if v and v not in out:
            out.append(v)
    return out


def extract_task3_component_candidates(scoring_points: str) -> List[str]:
    s = str(scoring_points or "")
    cands = re.findall(r"root cause component is ([^\n\r]+)", s, flags=re.IGNORECASE)
    out = []
    for c in cands:
        v = str(c).strip().strip(".").strip()
        if v and v not in out:
            out.append(v)
    return out

