from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, Optional, Type

from rca.common.config import load_dataset_config
from rca.adapter.base import DatasetAdapter


_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}


def register_adapter(name: str, cls: Type[DatasetAdapter]) -> None:
    _REGISTRY[str(name).strip().lower()] = cls


_INITIALIZED = False


def _lazy_register() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    _import_and_register("telecom", "rca.adapter.telecom", "TelecomAdapter")
    _import_and_register("market", "rca.adapter.market", "MarketAdapter")
    _import_and_register("bank", "rca.adapter.bank", "BankAdapter")


def _import_and_register(key: str, module: str, cls_name: str) -> None:
    try:
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            register_adapter(key, cls)
    except Exception:
        pass


@lru_cache(maxsize=64)
def get_dataset_adapter(dataset: str) -> Optional[DatasetAdapter]:
    _lazy_register()
    cfg = load_dataset_config(dataset)
    adapter_key = _adapter_key_from_config(cfg, dataset)
    if adapter_key:
        cls = _REGISTRY.get(adapter_key)
        if cls is not None:
            try:
                return cls(dataset=dataset, config=cfg)
            except Exception:
                pass
    key = _normalize_key(dataset)
    if key in _REGISTRY:
        try:
            return _REGISTRY[key](dataset=dataset, config=cfg)
        except Exception:
            pass
    return None


def _normalize_key(dataset: str) -> str:
    name = str(dataset or "").strip()
    name = name.replace("/", "_").replace("-", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return name


def _adapter_key_from_config(cfg: Dict[str, Any], dataset: str) -> Optional[str]:
    adapter_val = cfg.get("adapter", "")
    if not adapter_val or adapter_val == "existing":
        return None
    return str(adapter_val).strip().lower()


def adapter_for(dataset: str) -> Optional[DatasetAdapter]:
    return get_dataset_adapter(dataset)
