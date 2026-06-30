from __future__ import annotations

import re
from typing import Tuple


def task_need_flags(task_id: int) -> Tuple[bool, bool, bool]:
    """Return (need_time, need_component, need_reason)."""
    t = int(task_id or 0)
    if t == 1:
        return True, False, False
    if t == 2:
        return False, False, True
    if t == 3:
        return False, True, False
    if t == 4:
        return True, False, True
    if t == 5:
        return True, True, False
    if t == 6:
        return False, True, True
    return True, True, True


def required_root_count_from_instruction(instruction: str) -> int:
    """Infer the number of root causes from the user-visible incident query."""
    text = str(instruction or "").lower()
    word_numbers = {
        "single": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for word, value in word_numbers.items():
        if re.search(rf"\b{word}(?:\s+\w+){{0,2}}\s+failures?\b", text):
            return int(value)
    for pattern in [
        r"\b(\d+)\s+failures?\b",
        r"number\s+of\s+failures?\D+(\d+)",
        r"failures?\s+recorded\D+(\d+)",
    ]:
        match = re.search(pattern, text)
        if match:
            try:
                return max(1, int(match.group(1)))
            except Exception:
                pass
    return 1
