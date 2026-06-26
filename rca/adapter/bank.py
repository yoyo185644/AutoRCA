from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

    def basil_reason_taxonomy(self) -> List[str]:
        return list(_ALLOWED_REASONS)

    def component_semantic_type(self, component: str) -> str:
        c = self.parse_component_id(str(component or "")).lower()
        if c.startswith("mysql"):
            return "db"
        if c.startswith("redis"):
            return "cache"
        if c.startswith("tomcat"):
            return "app"
        if c.startswith("apache"):
            return "web"
        if c.startswith(("ig", "mg")):
            return "gateway"
        return ""

    def reason_semantic_type(self, reason: str) -> str:
        r = _norm_token(self.parse_root_cause_label(str(reason or "")))
        if "jvm" in r or "heap" in r:
            return "app"
        if "disk" in r:
            return "storage"
        if "network" in r or "latency" in r or "packet" in r:
            return "network"
        if "cpu" in r or "memory" in r:
            return ""
        return ""

    def reason_metric_keywords(self, reason: str) -> List[str]:
        r = _norm_token(self.parse_root_cause_label(str(reason or "")))
        if "cpu" in r:
            return ["cpu", "load", "util", "usage"]
        if "memory" in r or "oom" in r or "heap" in r:
            return ["mem", "memory", "heap", "oom", "jvm"]
        if "disk" in r and "read" in r:
            return ["disk", "read", "io", "i/o"]
        if "disk" in r and "space" in r:
            return ["disk", "space", "usage"]
        if "packet" in r and "loss" in r:
            return ["packet", "loss", "drop", "network"]
        if "latency" in r:
            return ["latency", "delay", "network"]
        return []

    def llm_reason_rules(self) -> List[str]:
        return [
            "JVM/heap reasons should align with application-server evidence such as Tomcat/JVM metrics.",
            "Disk reasons require disk/read/space metric evidence.",
            "Network reasons require packet/loss/latency network evidence.",
            "CPU and memory reasons are broad; require direct metric evidence before overriding deterministic top-1.",
        ]

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

    def constrain_llm_to_index_candidates(self) -> bool:
        return True

    def index_candidate_pool_k(
        self,
        *,
        task_id: int,
        k: int,
        default_pool_k: int,
        context: Dict[str, Any],
    ) -> int:
        # Bank has heterogeneous component channels (gateway/app/db/cache/web).
        # A top-5 global pool is usually dominated by IG/MG symptoms, so keep a
        # wider evidence pool and let the shared bounded reranker choose inside it.
        if int(task_id) in {1, 2, 3, 4, 5, 6, 7}:
            return max(int(default_pool_k), max(24, int(k) * 6))
        return int(default_pool_k)

    def refine_index_llm_gate(
        self,
        *,
        gate: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        task_id: int,
    ) -> Dict[str, Any]:
        out = dict(gate or {})
        reasons = [str(x) for x in list(out.get("reasons") or [])]
        need_time = bool((context or {}).get("need_time", False))
        need_comp = bool((context or {}).get("need_comp", False))
        need_reason = bool((context or {}).get("need_reason", False))

        rows = [dict(x) for x in list(candidates or []) if isinstance(x, Mapping)]
        comp_types = {
            self.component_semantic_type(str(x.get("component", "") or ""))
            for x in rows[:16]
            if str(x.get("component", "") or "").strip()
        }
        comp_types = {x for x in comp_types if x}
        reason_set = {
            self.parse_root_cause_label(str(x.get("reason", "") or ""))
            for x in rows[:16]
            if str(x.get("reason", "") or "").strip()
        }
        time_set = {
            str(x.get("datetime", "") or "")[:16]
            for x in rows[:16]
            if str(x.get("datetime", "") or "").strip()
        }

        if bool(out.get("need_llm")):
            if need_comp and len(comp_types) <= 1:
                out["need_llm"] = False
                reasons.append("bank_skip_llm_low_component_channel_diversity")
            if need_reason and len(reason_set) <= 1:
                out["need_llm"] = False
                reasons.append("bank_skip_llm_low_reason_diversity")
            if need_time and len(time_set) <= 1:
                out["need_llm"] = False
                reasons.append("bank_skip_llm_low_time_diversity")

        if bool(out.get("need_llm")):
            # Same bounded-adoption principle as other adapters: the LLM may
            # rerank evidence candidates, but should not freely override a
            # well-supported deterministic result.
            context["llm_change_min_confidence"] = max(
                self._safe_float(context.get("llm_change_min_confidence", 0.85)),
                0.88,
            )
            context["llm_max_score_drop_vs_det_top1"] = min(
                self._safe_float(context.get("llm_max_score_drop_vs_det_top1", 0.05)),
                0.04,
            )

        out["reasons"] = list(dict.fromkeys(reasons))
        out["bank_component_channel_count"] = int(len(comp_types))
        out["bank_reason_count"] = int(len(reason_set))
        out["bank_time_count"] = int(len(time_set))
        return out

    def extract_component_level(self) -> str:
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

    # rerank_candidate_hypotheses: uses base implementation.

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _reason_metric_support(self, *, component: str, reason: str, top_metrics: Sequence[str]) -> float:
        tokens = [str(x).lower() for x in self.reason_metric_keywords(reason)]
        if not tokens:
            return 0.0
        blob = " ".join(str(x or "").lower() for x in top_metrics or [])
        hits = sum(1 for tok in tokens if tok and tok in blob)
        return min(1.0, float(hits) / float(max(1, min(3, len(tokens)))))

    def expand_index_candidate_tuples(
        self,
        *,
        selected: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        task_id: int,
        k: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen_exact: set[Tuple[str, str, str]] = set()

        def add(item: Mapping[str, Any]) -> None:
            comp = self.parse_component_id(str(item.get("component", "") or ""))
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            dt = str(item.get("datetime", "") or "").strip()
            if not comp and not reason and not dt:
                return
            key = (comp, reason, dt)
            if key in seen_exact:
                return
            seen_exact.add(key)
            candidate = dict(item)
            candidate["component"] = comp
            candidate["reason"] = reason
            candidate["datetime"] = dt
            candidate["component_type"] = self.component_semantic_type(comp)
            candidate["score"] = self._safe_float(candidate.get("score", 0.0))
            rows.append(candidate)

        for item in list(selected or []):
            if isinstance(item, Mapping):
                add(item)
        for item in list((context or {}).get("candidate_tuples") or []):
            if isinstance(item, Mapping):
                add(item)

        if not rows:
            return []

        top_metrics = dict((context or {}).get("top_metrics", {}) or {})
        adjusted: List[Dict[str, Any]] = []
        for item in rows:
            comp = str(item.get("component", "") or "")
            ctype = str(item.get("component_type", "") or self.component_semantic_type(comp))
            reason = str(item.get("reason", "") or "")
            metrics = list(item.get("supporting_metrics", []) or top_metrics.get(comp, []) or [])
            score = self._safe_float(item.get("score", 0.0))
            symptomness = max(0.0, min(1.0, self._safe_float(item.get("symptomness", 0.0))))
            # Historical symptomness is an IDF-like discount. It is data-driven:
            # gateways that repeatedly appear as impacted components across
            # incidents should not always crowd out Tomcat/Redis/Mysql roots.
            if ctype == "gateway" and symptomness > 0:
                score *= max(0.35, 1.0 - symptomness)
            if reason:
                score += 0.12 * self._reason_metric_support(component=comp, reason=reason, top_metrics=metrics)
            item = dict(item)
            item["score"] = float(score)
            item["component_type"] = ctype
            if metrics and not item.get("supporting_metrics"):
                item["supporting_metrics"] = list(metrics)[:8]
            adjusted.append(item)

        adjusted.sort(key=lambda x: self._safe_float(x.get("score", 0.0)), reverse=True)

        limit = max(1, int(k))
        pool_limit = max(limit, 24)
        top_m_per_channel = 3
        selected_out: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()

        def keep(item: Mapping[str, Any]) -> None:
            if len(selected_out) >= pool_limit:
                return
            key = (
                str(item.get("component", "") or ""),
                str(item.get("reason", "") or ""),
                str(item.get("datetime", "") or ""),
            )
            if key in seen:
                return
            selected_out.append(dict(item))
            seen.add(key)

        for item in adjusted[:limit]:
            keep(item)

        by_channel: Dict[str, List[Dict[str, Any]]] = {}
        for item in adjusted:
            ch = str(item.get("component_type", "") or "other")
            by_channel.setdefault(ch, []).append(item)
        for ch in ("gateway", "app", "db", "cache", "web"):
            for item in by_channel.get(ch, [])[:top_m_per_channel]:
                keep(item)

        for item in adjusted:
            keep(item)
            if len(selected_out) >= pool_limit:
                break
        return selected_out

    def _hint_bonus(
        self,
        *,
        comp: str,
        reason: str,
        hint_components: List[str],
        hint_reasons: List[str],
        need_comp: bool,
        need_reason: bool,
    ) -> float:
        """Bank uses stronger hint bonuses than base default (1.40 comp / 1.10 reason)."""
        bonus = 0.0
        if need_comp and hint_components and comp in hint_components:
            bonus += 1.40
        if need_reason and hint_reasons and reason in hint_reasons:
            bonus += 1.10
        return bonus

    # select_time_candidate_override: uses base score-based implementation.
    # Bank's exact comp+reason match is expressed as score bonuses in _score_time_candidate.

    def _score_time_candidate(self, item: Dict[str, Any], comp: str, reason: str) -> float:
        score = float(item.get("score", 0.0) or 0.0)
        c0 = self.parse_component_id(str(item.get("component", "") or ""))
        r0 = self.parse_root_cause_label(str(item.get("reason", "") or ""))
        if comp and c0:
            score += 1.0 if c0 == comp else -1.0
        if reason and r0:
            score += 0.5 if r0 == reason else -0.5
        return score

    # _component_prior: base class computes data-driven lift(type) from query.csv
    # + telemetry. Bank-specific component_semantic_type() ensures correct typing.

    def supplement_candidate_pool(
        self,
        existing_tuples: List[Dict[str, Any]],
        context: Dict[str, Any],
        *,
        date_dir: str = "",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Bank: metric-scan supplement (base) + closed-set injection.

        Because Bank has only 14 known components, we inject any component
        from _ALLOWED_COMPONENTS that is still absent after the metric scan.
        This guarantees 100 % component recall before reranking.
        """
        # Step 1: metric anomaly scan (same as base)
        supplements = super().supplement_candidate_pool(
            existing_tuples, context, date_dir=date_dir, **kwargs
        )

        # Step 2: closed-set coverage — inject remaining allowed components
        covered: set = {
            self.parse_component_id(str(t.get("component", "") or ""))
            for t in list(existing_tuples) + list(supplements)
            if str(t.get("component", "") or "").strip()
        }

        pool_reasons: List[str] = list(
            dict.fromkeys(
                str(t.get("reason", "") or "").strip()
                for t in existing_tuples
                if str(t.get("reason", "") or "").strip()
            )
        )
        pair_reasons: List[str] = pool_reasons[:2] if pool_reasons else [""]

        for comp in _ALLOWED_COMPONENTS:
            comp_id = self.parse_component_id(comp)
            if comp_id in covered:
                continue
            for reason in pair_reasons:
                supplements.append(
                    {
                        "component": comp_id,
                        "reason": reason,
                        "datetime": "",
                        "score": 0.10,   # very low; only wins if reranking strongly promotes it
                        "source": "closed_set_supplement",
                        "supporting_metrics": [],
                        "component_type": "",
                        "first_anomaly_time": "",
                        "symptomness": 0.0,
                    }
                )
            covered.add(comp_id)

        return supplements

    # should_use_light_agent: base default covers {3,5,6,7} — no override needed
