from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Callable, Iterable, List, Mapping, Sequence

import pandas as pd


class DataFrameDiskCache:
    """Small reusable persistent cache for derived telemetry frames."""

    def __init__(self, *, cache_dir: str, enabled: bool = True) -> None:
        self.cache_dir = str(cache_dir or "test/cache/basil_index/frames")
        self.enabled = bool(enabled)
        self._lock = threading.RLock()

    def configure(self, *, enabled: bool | None = None, cache_dir: str | None = None) -> None:
        with self._lock:
            if enabled is not None:
                self.enabled = bool(enabled)
            if cache_dir:
                self.cache_dir = str(cache_dir)

    def load_or_build(
        self,
        *,
        namespace: str,
        key_parts: Sequence[object],
        source_paths: Iterable[str],
        builder: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        if not self.enabled:
            return builder()

        paths = [str(p) for p in source_paths or []]
        fingerprint = self._fingerprint(paths)
        frame_path, meta_path = self._paths(namespace=namespace, key_parts=key_parts)
        with self._lock:
            if self._is_fresh(meta_path, fingerprint) and os.path.exists(frame_path):
                try:
                    return pd.read_pickle(frame_path, compression="gzip")
                except Exception:
                    pass

            frame = builder()
            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
            tmp_frame = f"{frame_path}.tmp"
            tmp_meta = f"{meta_path}.tmp"
            frame.to_pickle(tmp_frame, compression="gzip")
            with open(tmp_meta, "w", encoding="utf-8") as fh:
                json.dump({"fingerprint": fingerprint}, fh, ensure_ascii=False, sort_keys=True)
            os.replace(tmp_frame, frame_path)
            os.replace(tmp_meta, meta_path)
            return frame

    def _paths(self, *, namespace: str, key_parts: Sequence[object]) -> tuple[str, str]:
        safe_namespace = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(namespace))
        raw = json.dumps([str(x) for x in key_parts], ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        frame_path = os.path.join(self.cache_dir, f"{safe_namespace}_{digest}.pkl.gz")
        return frame_path, f"{frame_path}.meta.json"

    def _fingerprint(self, paths: Iterable[str]) -> List[Mapping[str, object]]:
        out = []
        for path in sorted(str(p) for p in paths or []):
            try:
                st = os.stat(path)
                out.append(
                    {
                        "path": os.path.abspath(path),
                        "mtime_ns": int(st.st_mtime_ns),
                        "size": int(st.st_size),
                    }
                )
            except OSError:
                out.append({"path": os.path.abspath(path), "missing": True})
        return out

    def _is_fresh(self, meta_path: str, fingerprint: List[Mapping[str, object]]) -> bool:
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            return meta.get("fingerprint") == fingerprint
        except Exception:
            return False
