from __future__ import annotations

import copy
import os
import re
from functools import lru_cache
from typing import Any, Dict

import yaml


CONFIG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "datasets"))


def dataset_config_name(dataset: str) -> str:
    name = str(dataset or "").strip().lower()
    name = name.replace("/", "_").replace("-", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "default"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml_file(name: str) -> Dict[str, Any]:
    path = os.path.join(CONFIG_ROOT, f"{name}.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=64)
def load_dataset_config(dataset: str) -> Dict[str, Any]:
    requested = dataset_config_name(dataset)
    raw = _load_yaml_file(requested)
    if not raw and requested != "default":
        return load_dataset_config("default")

    parent_name = str(raw.get("inherits") or "default").strip()
    if not parent_name or parent_name == requested:
        parent_name = "default"
    parent = {} if requested == "default" else load_dataset_config(parent_name)
    return _deep_merge(parent, raw)
