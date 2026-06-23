from __future__ import annotations

import re


def norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def service_base(component: str) -> str:
    c = str(component or "").strip()
    c = re.sub(r"\.ts(?::\d+)?$", "", c)
    c = re.sub(r"-grpc$", "", c)
    m = re.match(r"^(.+)-\d+$", c)
    if m:
        return m.group(1)
    return c
