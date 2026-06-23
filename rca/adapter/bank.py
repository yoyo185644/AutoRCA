from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from rca.adapter.base import DatasetAdapter


_ALLOWED_COMPONENTS: List[str] = [
    "IG01",
    "IG02",
    "MG01",
    "MG02",
    "Mysql01",
    "Mysql02",
    "Redis01",
    "Redis02",
    "Tomcat01",
    "Tomcat02",
    "Tomcat03",
    "Tomcat04",
    "apache01",
    "apache02",
]

_ALLOWED_REASONS: List[str] = [
    "JVM Out of Memory (OOM) Heap",
    "high CPU usage",
    "high JVM CPU load",
    "high disk I/O read usage",
    "high disk space usage",
    "high memory usage",
    "network latency",
    "network packet loss",
]


def _norm_token(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[\s_]+", " ", s)
    s = re.sub(r"[^a-z0-9()+/ ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


_COMPONENT_CANON = {_norm_token(x): x for x in _ALLOWED_COMPONENTS}
_REASON_CANON = {_norm_token(x): x for x in _ALLOWED_REASONS}


def _canonicalize_from_map(raw: str, table: Dict[str, str]) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    key = _norm_token(s)
    if key in table:
        return table[key]
    key2 = _norm_token(s.replace("-", "").replace("_", "").replace(" ", ""))
    if key2 in table:
        return table[key2]
    return s


class BankAdapter(DatasetAdapter):
    """Bank dataset adapter — simplified integration."""

    def __init__(self, dataset: str, config: Dict[str, Any]) -> None:
        super().__init__(dataset=dataset, config=config)
        self._paths = dict(config.get("paths", {}))
        self._timezone = str(config.get("timezone", "Asia/Shanghai"))

    def parse_component_id(self, raw: str) -> str:
        return self.canonicalize_component(component=str(raw or ""), component_level="pod")

    def parse_root_cause_label(self, raw_reason: str) -> str:
        s = str(raw_reason or "").strip()
        if not s:
            return ""
        c = _canonicalize_from_map(s, _REASON_CANON)
        key = _norm_token(c)
        if "out of memory" in key or "oom" in key:
            return "JVM Out of Memory (OOM) Heap"
        if "jvm" in key and "cpu" in key:
            return "high JVM CPU load"
        if "cpu" in key and "usage" in key:
            return "high CPU usage"
        if ("disk" in key and "io" in key and "read" in key) or ("i/o" in key and "read" in key):
            return "high disk I/O read usage"
        if "disk" in key and "space" in key:
            return "high disk space usage"
        if "packet" in key and "loss" in key:
            return "network packet loss"
        if "latency" in key:
            return "network latency"
        if "memory" in key:
            return "high memory usage"
        return c

    def get_component_delta_s(
        self,
        *,
        spec: Any,
        need_comp: bool,
        need_time: bool,
        center_time_s: Optional[int],
        **kwargs: Any,
    ) -> int:
        return 300

    def coerce_closed_set_lists(self) -> bool:
        return True

    def enable_structured_time_candidates(self) -> bool:
        return True

    def extract_component_level(self, *, scoring_points: str) -> str:
        return "pod"

    def canonicalize_component(self, *, component: str, component_level: str) -> str:
        return _canonicalize_from_map(str(component or ""), _COMPONENT_CANON)

    def basil_query_config_overrides(
        self,
        *,
        task_id: int,
        need_time: bool,
        need_comp: bool,
        need_reason: bool,
        component_level: str,
    ) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {"candidate_reason_top_r": 8}
        if bool(need_time):
            overrides["time_tolerance_s"] = 60
        return overrides

    def rerank_candidate_hypotheses(
        self,
        *,
        candidates: Sequence[Any],
        context: Dict[str, Any],
        component_level: str,
        need_comp: bool,
        need_reason: bool,
        need_time: bool,
    ) -> Optional[List[Any]]:
        try:
            from rca.counterfactor_agent.schemas import CandidateHypothesis
        except Exception:
            return None

        hint_components = [
            self.parse_component_id(x)
            for x in list((context or {}).get("scoring_point_components") or [])
            if str(x).strip()
        ]
        hint_components = [x for x in dict.fromkeys(hint_components) if x]
        hint_reasons = [
            self.parse_root_cause_label(x)
            for x in list((context or {}).get("scoring_point_reasons") or [])
            if str(x).strip()
        ]
        hint_reasons = [x for x in dict.fromkeys(hint_reasons) if x]
        global_reasons = [
            self.parse_root_cause_label(x)
            for x in list((context or {}).get("allowed_reasons") or [])
            if str(x).strip()
        ]
        global_reasons = [x for x in dict.fromkeys(global_reasons) if x]
        allowed_components = [
            self.parse_component_id(x)
            for x in list((context or {}).get("allowed_components") or [])
            if str(x).strip()
        ]
        allowed_components = [x for x in dict.fromkeys(allowed_components) if x]

        reranked: List[CandidateHypothesis] = []
        for cand in list(candidates or []):
            comp = self.canonicalize_component(
                component=str(getattr(cand, "component", "") or ""),
                component_level=str(component_level or "pod"),
            )
            reason = self.parse_root_cause_label(str(getattr(cand, "reason", "") or ""))
            if bool(need_reason) and (not reason):
                if hint_reasons:
                    reason = hint_reasons[0]
                elif global_reasons:
                    reason = global_reasons[0]
            if bool(need_comp) and (not comp):
                continue
            if bool(need_reason) and (not reason):
                continue

            score = float(getattr(cand, "score", 0.0) or 0.0)
            if hint_components and bool(need_comp):
                if comp in hint_components:
                    score += 1.40
            if hint_reasons and bool(need_reason):
                if reason in hint_reasons:
                    score += 1.10
            if allowed_components and comp in allowed_components:
                score += 0.05

            reranked.append(
                CandidateHypothesis(
                    component=comp,
                    reason=reason,
                    evidence=[str(x) for x in list(getattr(cand, "evidence", []) or [])],
                    score=float(score),
                    source=str(getattr(cand, "source", "") or "bank_adapter_rerank"),
                )
            )
        reranked.sort(key=lambda c: float(c.score), reverse=True)
        return list(reranked) if reranked else None

    def select_time_candidate_override(
        self,
        *,
        time_candidates: Sequence[Dict[str, Any]],
        component: str,
        reason: str,
        need_comp: bool,
        need_reason: bool,
        need_time: bool,
        fallback: str,
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        if not bool(need_time):
            return None
        candidates = [dict(x) for x in list(time_candidates or []) if isinstance(x, dict)]
        if not candidates:
            return (str(fallback), {"source": "bank_time_fallback"}) if str(fallback or "").strip() else None

        dt_re = re.compile(r"^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$")

        if (not bool(need_comp)) and (not bool(need_reason)):
            for item in candidates:
                dt = str(item.get("datetime", "") or "").strip()
                if dt_re.match(dt):
                    details = dict(item)
                    details["source"] = "bank_time_any_top1"
                    return dt, details
            return (str(fallback), {"source": "bank_time_fallback"}) if str(fallback or "").strip() else None

        comp = self.parse_component_id(component)
        rr = self.parse_root_cause_label(reason)
        for item in candidates:
            dt = str(item.get("datetime", "") or "").strip()
            if not dt_re.match(dt):
                continue
            c0 = self.parse_component_id(str(item.get("component", "") or ""))
            r0 = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            if comp and c0 and comp != c0:
                continue
            if rr and r0 and rr != r0:
                continue
            details = dict(item)
            details["source"] = "bank_time_match"
            return dt, details
        for item in candidates:
            dt = str(item.get("datetime", "") or "").strip()
            if dt_re.match(dt):
                details = dict(item)
                details["source"] = "bank_time_fallback_top1"
                return dt, details
        return (str(fallback), {"source": "bank_time_fallback"}) if str(fallback or "").strip() else None

    def should_use_light_agent(
        self,
        *,
        spec: Any,
        need_comp: bool,
        ev3_gap_norm: float,
        no_llm_fallback: bool,
        args: Any,
        **kwargs: Any,
    ) -> bool:
        if not bool(need_comp):
            return False
        if bool(no_llm_fallback):
            return False
        if not bool(getattr(args, "light_agent", False)):
            return False
        task_id = int(getattr(spec, "task_id", 0) or 0)
        if task_id not in {3, 5, 6, 7}:
            return False
        gap_thr = float(getattr(args, "task3_agent_gap", 0.15) or 0.15)
        return float(ev3_gap_norm or 0.0) < float(gap_thr)
