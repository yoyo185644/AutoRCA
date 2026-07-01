from __future__ import annotations

import os
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

    def basil_reason_taxonomy(self) -> List[str]:
        return list(_ALLOWED_REASONS)

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

    def prefer_local_first_rootness(self) -> bool:
        return True

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
            return ["packet", "loss", "drop", "network", "fin-wait", "close-wait", "reset", "timeout"]
        if "latency" in r:
            return ["latency", "delay", "network", "mrt", "rtt", "response"]
        return []

    def reason_compatible_component_types(self, reason: str) -> List[str]:
        r = _norm_token(self.parse_root_cause_label(str(reason or "")))
        if "jvm" in r or "heap" in r:
            return ["app"]
        if "disk" in r:
            return ["db", "web", "app"]
        if "network" in r or "latency" in r or "packet" in r:
            return ["gateway", "web", "app"]
        if "cpu" in r or "memory" in r:
            return ["app", "db", "cache", "web", "gateway"]
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
        c = _canonicalize_from_map(str(component or ""), _COMPONENT_CANON)
        return c if c in _ALLOWED_COMPONENTS else ""

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

    def augment_index_reasons(
        self,
        *,
        reasons: Sequence[str],
        result: Any,
        instruction: str,
        task_id: int,
        args: Any,
        need_reason: bool,
    ) -> Tuple[List[str], Dict[str, float], Dict[str, Any]]:
        ordered, scores, evidence = super().augment_index_reasons(
            reasons=reasons,
            result=result,
            instruction=instruction,
            task_id=task_id,
            args=args,
            need_reason=need_reason,
        )
        if not need_reason:
            return ordered, scores, evidence

        log_scores = self._log_reason_scores(instruction=instruction)
        if not log_scores:
            log_scores = {}

        taxonomy = [self.parse_root_cause_label(x) for x in self.basil_reason_taxonomy()]
        taxonomy = [x for x in dict.fromkeys(taxonomy) if x]
        consensus_scores = self._reason_consensus_scores_from_index(result=result)
        merged: Dict[str, float] = {}
        for reason in taxonomy:
            # Metric/index evidence remains primary; log evidence is an
            # independent telemetry signal that helps disambiguate OOM, SQL
            # slow-query, and network/error cases without reading labels.
            merged[reason] = max(
                self._safe_float(scores.get(reason, 0.0)),
                0.90 * self._safe_float(consensus_scores.get(reason, 0.0)),
                0.85 * self._safe_float(log_scores.get(reason, 0.0)),
            )
        mx = max([abs(v) for v in merged.values()] + [0.0])
        norm = {k: (max(0.0, float(v) / mx) if mx > 1e-12 else 0.0) for k, v in merged.items()}
        ordered2 = [r for r, _ in sorted(norm.items(), key=lambda kv: float(kv[1]), reverse=True)]
        evidence2 = dict(evidence or {})
        evidence2["bank_log_reason_scores"] = dict(log_scores)
        evidence2["bank_reason_consensus_scores"] = dict(consensus_scores)
        return ordered2, norm, evidence2

    def _reason_consensus_scores_from_index(self, *, result: Any) -> Dict[str, float]:
        comp_ev = getattr(result, "component_evidence", None)
        top_metrics = dict(getattr(comp_ev, "top_metrics", {}) or {})
        comp_scores = dict(getattr(comp_ev, "combined_scores", {}) or {})
        if not top_metrics:
            top_metrics = dict(getattr(result, "top_metrics", {}) or {})
        if not comp_scores:
            comp_scores = dict(getattr(result, "combined_scores", {}) or {})
        taxonomy = [self.parse_root_cause_label(x) for x in self.basil_reason_taxonomy()]
        taxonomy = [x for x in dict.fromkeys(taxonomy) if x]
        if not taxonomy or not top_metrics:
            return {}

        max_comp = max([abs(self._safe_float(v)) for v in comp_scores.values()] + [1.0])
        raw: Dict[str, float] = {r: 0.0 for r in taxonomy}
        hits_by_reason: Dict[str, int] = {r: 0 for r in taxonomy}
        for rank, (comp, _score) in enumerate(
            sorted(comp_scores.items(), key=lambda kv: self._safe_float(kv[1]), reverse=True)[:14],
            start=1,
        ):
            comp = self.parse_component_id(str(comp or ""))
            metrics = [str(x) for x in list(top_metrics.get(comp, []) or []) if str(x).strip()]
            if not comp or not metrics:
                continue
            comp_norm = self._safe_float(comp_scores.get(comp, 0.0)) / max(max_comp, 1e-9)
            ctype = self.component_semantic_type(comp)
            rank_weight = 1.0 / (1.0 + 0.20 * float(rank - 1))
            for reason in taxonomy:
                support = self._reason_tie_support(component=comp, reason=reason, top_metrics=metrics)
                if support <= 0.0:
                    continue
                expected = set(self.reason_compatible_component_types(reason) or [])
                if expected and ctype:
                    support *= 1.12 if ctype in expected else 0.55
                raw[reason] += max(0.10, comp_norm) * rank_weight * min(1.0, support)
                hits_by_reason[reason] += 1

        for reason, hits in hits_by_reason.items():
            if hits >= 2:
                raw[reason] *= 1.0 + min(0.30, 0.08 * float(hits - 1))
            elif hits <= 0:
                raw[reason] = 0.0
        mx = max([abs(v) for v in raw.values()] + [0.0])
        if mx <= 1e-12:
            return raw
        return {k: max(0.0, float(v) / mx) for k, v in raw.items()}

    def augment_index_time_candidates(
        self,
        *,
        time_candidates: Sequence[Dict[str, Any]],
        result: Any,
        instruction: str,
        task_id: int,
        args: Any,
        need_time: bool,
        component_scores: Dict[str, float],
        component_first_times: Dict[str, str],
        component_peak_times: Dict[str, str],
        context_overrides: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Bank trace files are large and trace-onset candidates tend to be
        # global symptoms.  Keep the indexed component times and add
        # telemetry-only metric change-points, avoiding the trace-heavy base
        # augmentation path used by datasets where trace is cleaner.
        merged = [dict(x) for x in list(time_candidates or []) if isinstance(x, Mapping)]
        if not need_time:
            return merged
        max_comp = max([abs(self._safe_float(v)) for v in dict(component_scores or {}).values()] + [1.0])
        for comp, dt in dict(component_first_times or {}).items():
            if not str(dt or "").strip():
                continue
            score = 0.60 + 0.20 * (self._safe_float(dict(component_scores or {}).get(comp, 0.0)) / max(max_comp, 1e-9))
            merged.append({"datetime": str(dt), "source": "component_metric_first", "score": score, "component": str(comp), "reason": ""})
        for comp, dt in dict(component_peak_times or {}).items():
            if not str(dt or "").strip():
                continue
            score = 0.52 + 0.16 * (self._safe_float(dict(component_scores or {}).get(comp, 0.0)) / max(max_comp, 1e-9))
            merged.append({"datetime": str(dt), "source": "component_metric_peak", "score": score, "component": str(comp), "reason": ""})
        merged.extend(self._metric_change_point_time_candidates(instruction=instruction))
        return self._filter_time_candidates_to_query_window(merged, instruction=instruction)

    # rerank_candidate_hypotheses: uses base implementation.

    def _postprocess_reranked(self, candidates: List[Any]) -> List[Any]:
        ordered = list(candidates or [])
        if len(ordered) < 2:
            return ordered
        ordered = self._promote_component_only_app_challenger(ordered)
        ordered = self._promote_component_time_challenger(ordered)
        ordered = self._promote_same_reason_local_challenger(ordered)
        ordered = self._promote_same_component_reason_challenger(ordered)
        return ordered

    def _candidate_features(self, cand: Any) -> Dict[str, float]:
        raw = dict(getattr(cand, "rootness_features", {}) or {})
        raw.update(dict(getattr(cand, "metric_ownership_details", {}) or {}))
        for key in (
            "metric_ownership",
            "metric_score",
            "ownership",
            "locality",
            "persistence",
            "instance_owned_score",
            "sibling_contrast",
            "symptomness",
            "causal_source_score",
            "outgoing_propagation",
            "incoming_symptomness",
        ):
            if hasattr(cand, key):
                raw[key] = getattr(cand, key)
        return {str(k): self._safe_float(v) for k, v in raw.items()}

    def _candidate_owned_signal(self, cand: Any) -> float:
        f = self._candidate_features(cand)
        return max(
            self._safe_float(f.get("instance_owned_score", 0.0)),
            self._safe_float(f.get("metric_ownership", 0.0)),
            self._safe_float(f.get("ownership", 0.0)),
            self._safe_float(f.get("locality", 0.0)),
            self._safe_float(f.get("sibling_contrast", 0.0)),
            0.5 * self._safe_float(f.get("local_change", 0.0)),
        )

    def _move_candidate_to_front(self, ordered: List[Any], idx: int, reason: str) -> List[Any]:
        if idx <= 0 or idx >= len(ordered):
            return ordered
        cand = ordered.pop(idx)
        try:
            old_ev = [str(x) for x in list(getattr(cand, "evidence", []) or [])]
            old_ev.append(reason)
            cand.evidence = list(dict.fromkeys(old_ev))
        except Exception:
            pass
        ordered.insert(0, cand)
        return ordered

    def _promote_component_only_app_challenger(self, ordered: List[Any]) -> List[Any]:
        top = ordered[0]
        if str(getattr(top, "datetime", "") or "").strip():
            return ordered
        if self.parse_root_cause_label(str(getattr(top, "reason", "") or "")):
            return ordered
        top_comp = str(getattr(top, "component", "") or "")
        if self.component_semantic_type(top_comp) != "gateway":
            return ordered
        top_features = dict(getattr(top, "rootness_features", {}) or {})
        top_owned = self._safe_float(top_features.get("instance_owned_score", 0.0))
        if top_owned >= 0.08:
            return ordered
        top_score = self._safe_float(getattr(top, "score", 0.0))
        for idx, cand in enumerate(ordered[1:8], start=1):
            comp = str(getattr(cand, "component", "") or "")
            if self.component_semantic_type(comp) != "app":
                continue
            features = dict(getattr(cand, "rootness_features", {}) or {})
            owned = self._safe_float(features.get("instance_owned_score", 0.0))
            if owned < max(0.10, top_owned * 1.8):
                continue
            if self._safe_float(getattr(cand, "score", 0.0)) < top_score - 0.40:
                continue
            return self._move_candidate_to_front(ordered, idx, "bank_component_only_app_challenger")
        return ordered

    def _promote_component_time_challenger(self, ordered: List[Any]) -> List[Any]:
        """For component+time candidates, avoid weak gateway symptoms.

        This path requires a datetime, so component-only task3 is untouched.
        It only changes the top result when a non-gateway challenger has much
        stronger component-owned evidence and is still score-competitive.
        """
        top = ordered[0]
        if not str(getattr(top, "datetime", "") or "").strip():
            return ordered
        top_comp = str(getattr(top, "component", "") or "")
        if self.component_semantic_type(top_comp) != "gateway":
            return ordered
        top_owned = self._candidate_owned_signal(top)
        if top_owned >= 0.10:
            return ordered
        top_score = self._safe_float(getattr(top, "score", 0.0))
        for idx, cand in enumerate(ordered[1:10], start=1):
            comp = str(getattr(cand, "component", "") or "")
            ctype = self.component_semantic_type(comp)
            if ctype not in {"app", "db", "cache", "web"}:
                continue
            owned = self._candidate_owned_signal(cand)
            if owned < max(0.12, top_owned * 1.7):
                continue
            if self._safe_float(getattr(cand, "score", 0.0)) < top_score - 0.22:
                continue
            return self._move_candidate_to_front(ordered, idx, "bank_component_time_owned_challenger")
        return ordered

    def _promote_same_reason_local_challenger(self, ordered: List[Any]) -> List[Any]:
        """For component+reason tasks, keep the reason but choose the more local root."""
        top = ordered[0]
        reason = self.parse_root_cause_label(str(getattr(top, "reason", "") or ""))
        if not reason:
            return ordered
        top_comp = str(getattr(top, "component", "") or "")
        top_type = self.component_semantic_type(top_comp)
        top_owned = self._candidate_owned_signal(top)
        top_score = self._safe_float(getattr(top, "score", 0.0))
        for idx, cand in enumerate(ordered[1:10], start=1):
            if self.parse_root_cause_label(str(getattr(cand, "reason", "") or "")) != reason:
                continue
            comp = str(getattr(cand, "component", "") or "")
            ctype = self.component_semantic_type(comp)
            if not ctype or ctype == top_type:
                continue
            owned = self._candidate_owned_signal(cand)
            score = self._safe_float(getattr(cand, "score", 0.0))
            if top_type == "gateway":
                if owned < max(0.10, top_owned * 1.45):
                    continue
                if score < top_score - 0.12:
                    continue
                return self._move_candidate_to_front(ordered, idx, "bank_same_reason_owned_challenger")
            if owned >= max(0.18, top_owned * 1.8) and score >= top_score - 0.06:
                return self._move_candidate_to_front(ordered, idx, "bank_same_reason_stronger_local_evidence")
        return ordered

    def _promote_same_component_reason_challenger(self, ordered: List[Any]) -> List[Any]:
        """For a fixed component, switch reason only when evidence clearly favors it."""
        top = ordered[0]
        comp = str(getattr(top, "component", "") or "")
        reason = self.parse_root_cause_label(str(getattr(top, "reason", "") or ""))
        if not comp or not reason:
            return ordered
        top_metrics = [str(x) for x in list(getattr(top, "evidence", []) or [])]
        top_support = self._reason_tie_support(component=comp, reason=reason, top_metrics=top_metrics)
        top_score = self._safe_float(getattr(top, "score", 0.0))
        for idx, cand in enumerate(ordered[1:10], start=1):
            if str(getattr(cand, "component", "") or "") != comp:
                continue
            cand_reason = self.parse_root_cause_label(str(getattr(cand, "reason", "") or ""))
            if not cand_reason or cand_reason == reason:
                continue
            metrics = [str(x) for x in list(getattr(cand, "evidence", []) or [])]
            support = self._reason_tie_support(component=comp, reason=cand_reason, top_metrics=metrics)
            score = self._safe_float(getattr(cand, "score", 0.0))
            if support < max(0.34, top_support + 0.10):
                continue
            if score < top_score - 0.08:
                continue
            return self._move_candidate_to_front(ordered, idx, "bank_same_component_reason_evidence")
        return ordered

    def postprocess_output_tuples(
        self,
        tuples: Sequence[Dict[str, Any]],
        *,
        context: Dict[str, Any],
        need_comp: bool,
        need_reason: bool,
        need_time: bool,
    ) -> List[Dict[str, Any]]:
        ordered = [dict(x) for x in list(tuples or []) if isinstance(x, Mapping)]
        if need_time and need_comp and not need_reason and len(ordered) > 1:
            ordered = self._postprocess_component_time_tuples(ordered)
        if need_comp and need_reason and not need_time and len(ordered) > 1:
            ordered = self._postprocess_component_reason_tuples(ordered)
        if need_time and need_reason and not need_comp and len(ordered) > 1:
            ordered = self._postprocess_reason_time_tuples(ordered, context=context)
        if not (need_time and not need_comp and not need_reason):
            return ordered
        if len(ordered) < 2:
            return ordered
        ordered = self._postprocess_time_only_tuples(ordered)
        diverse: List[Dict[str, Any]] = []
        used_clusters: set[str] = set()
        for item in ordered:
            cluster = self._time_cluster(str(item.get("datetime", "") or ""))
            if cluster and cluster in used_clusters:
                continue
            if cluster:
                used_clusters.add(cluster)
            diverse.append(item)
        if len(diverse) >= len(ordered):
            return diverse
        seen = {
            (
                str(x.get("datetime", "") or ""),
                str(x.get("component", "") or ""),
                str(x.get("reason", "") or ""),
            )
            for x in diverse
        }
        for item in ordered:
            key = (
                str(item.get("datetime", "") or ""),
                str(item.get("component", "") or ""),
                str(item.get("reason", "") or ""),
            )
            if key in seen:
                continue
            diverse.append(item)
            seen.add(key)
            if len(diverse) >= len(ordered):
                break
        return diverse

    def _postprocess_time_only_tuples(self, ordered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(ordered) < 2:
            return ordered
        top = ordered[0]
        top_score = self._safe_float(top.get("score", 0.0))
        top_source = str(top.get("source", "") or "").lower()
        if "component_metric_first" in top_source:
            for idx, item in enumerate(ordered[1:10], start=1):
                source = str(item.get("source", "") or "").lower()
                if "metric_change_point" not in source:
                    continue
                if self._safe_float(item.get("score", 0.0)) < top_score - 0.035:
                    continue
                chosen = ordered.pop(idx)
                chosen["source"] = str(chosen.get("source", "") or "") + "|bank_time_only_change_point_tie"
                ordered.insert(0, chosen)
                top = chosen
                top_score = self._safe_float(top.get("score", 0.0))
                top_source = str(top.get("source", "") or "").lower()
                break
        top_boundary = self._is_boundary_time(str(top.get("datetime", "") or ""))
        top_symptom = self._safe_float(top.get("symptomness", 0.0)) >= 0.80 or "basil_hypothesis" in top_source
        if not (top_boundary or top_symptom):
            return ordered

        best_idx = -1
        best_score = -1.0
        for idx, item in enumerate(ordered[1:12], start=1):
            dt = str(item.get("datetime", "") or "")
            if not dt:
                continue
            source = str(item.get("source", "") or "").lower()
            non_boundary = not self._is_boundary_time(dt)
            local_change = "metric_change_point" in source or "component_metric" in source
            if not (non_boundary or local_change):
                continue
            if self._safe_float(item.get("score", 0.0)) < top_score - 0.36:
                continue
            owned = max(
                self._safe_float(item.get("instance_owned_score", 0.0)),
                self._safe_float(item.get("metric_ownership", 0.0)),
                self._safe_float(item.get("sibling_contrast", 0.0)),
            )
            score = self._safe_float(item.get("score", 0.0)) + 0.10 * owned
            if local_change:
                score += 0.06
            if non_boundary:
                score += 0.04
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx > 0:
            chosen = ordered.pop(best_idx)
            chosen["source"] = str(chosen.get("source", "") or "") + "|bank_time_only_local_change"
            ordered.insert(0, chosen)
        return ordered

    def _postprocess_reason_time_tuples(self, ordered: List[Dict[str, Any]], *, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not ordered:
            return ordered
        top = ordered[0]
        top_score = self._safe_float(top.get("score", 0.0))
        top_boundary = self._is_boundary_time(str(top.get("datetime", "") or ""))
        top_reason = self.parse_root_cause_label(str(top.get("reason", "") or ""))
        reason_scores = {self.parse_root_cause_label(str(k)): self._safe_float(v) for k, v in dict((context or {}).get("reason_scores", {}) or {}).items()}
        log_scores = self._log_reason_scores_for_context(context or {})

        def reason_strength(item: Mapping[str, Any]) -> float:
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            comp = self.parse_component_id(str(item.get("component", "") or ""))
            metrics = [str(x) for x in list(item.get("supporting_metrics") or item.get("evidence") or [])]
            return max(
                self._safe_float(reason_scores.get(reason, 0.0)),
                self._safe_float(log_scores.get(reason, 0.0)),
                self._reason_tie_support(component=comp, reason=reason, top_metrics=metrics),
            )

        top_strength = reason_strength(top)

        for idx, item in enumerate(ordered[1:12], start=1):
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            if not reason:
                continue
            score = self._safe_float(item.get("score", 0.0))
            source = str(item.get("source", "") or "").lower()
            boundary = self._is_boundary_time(str(item.get("datetime", "") or ""))
            surface = ("indexed_reason_surface" in source) or not str(item.get("component", "") or "").strip()
            same_reason = bool(top_reason and reason == top_reason)
            same_dt = str(item.get("datetime", "") or "") == str(top.get("datetime", "") or "")
            same_cluster = self._time_cluster(str(item.get("datetime", "") or "")) == self._time_cluster(str(top.get("datetime", "") or ""))
            strength = reason_strength(item)
            if (same_dt or same_cluster) and strength >= max(0.20, top_strength + 0.08) and score >= top_score - 0.22:
                chosen = ordered.pop(idx)
                chosen["source"] = str(chosen.get("source", "") or "") + "|bank_reason_time_stronger_reason_evidence"
                ordered.insert(0, chosen)
                top = chosen
                top_score = self._safe_float(top.get("score", 0.0))
                top_boundary = self._is_boundary_time(str(top.get("datetime", "") or ""))
                top_reason = self.parse_root_cause_label(str(top.get("reason", "") or ""))
                top_strength = reason_strength(top)
                break
            if same_reason and top_boundary and not boundary and score >= top_score - 0.20:
                chosen = ordered.pop(idx)
                chosen["source"] = str(chosen.get("source", "") or "") + "|bank_reason_time_non_boundary"
                ordered.insert(0, chosen)
                break
            if surface and not boundary and score >= top_score - 0.16:
                chosen = ordered.pop(idx)
                chosen["source"] = str(chosen.get("source", "") or "") + "|bank_reason_time_surface"
                ordered.insert(0, chosen)
                break

        first = ordered[0]
        first_reason = self.parse_root_cause_label(str(first.get("reason", "") or ""))
        first_comp = self.parse_component_id(str(first.get("component", "") or ""))
        first_metrics = [str(x) for x in list(first.get("supporting_metrics") or first.get("evidence") or [])]
        if not first_metrics:
            first_dt = str(first.get("datetime", "") or "")
            first_score = self._safe_float(first.get("score", 0.0))
            for item in ordered[1:10]:
                if str(item.get("datetime", "") or "") != first_dt:
                    continue
                metrics = [str(x) for x in list(item.get("supporting_metrics") or item.get("evidence") or [])]
                if metrics and self._safe_float(item.get("score", 0.0)) >= first_score - 0.12:
                    first_metrics = metrics
                    break

        if first_reason == "high JVM CPU load" and self.component_semantic_type(first_comp) == "gateway":
            # JVM CPU is not a gateway-owned failure mode.  For reason-time
            # queries, keep the time evidence but use the dataset taxonomy's
            # non-JVM CPU reason when CPU evidence is competitive.
            cpu_strength = max(
                self._safe_float(reason_scores.get("high CPU usage", 0.0)),
                self._safe_float(log_scores.get("high CPU usage", 0.0)),
            )
            if cpu_strength >= max(0.10, 0.70 * reason_strength(first)):
                first["reason"] = "high CPU usage"
                first["source"] = str(first.get("source", "") or "") + "|bank_gateway_jvm_cpu_to_cpu"
                ordered[0] = first
                first_reason = "high CPU usage"

        metric_reason = self._dominant_reason_from_metrics(first_metrics)
        if metric_reason and metric_reason != first_reason:
            current_strength = self._reason_tie_support(component=first_comp, reason=first_reason, top_metrics=first_metrics)
            metric_strength = self._reason_tie_support(component=first_comp, reason=metric_reason, top_metrics=first_metrics)
            if metric_strength >= max(0.38, current_strength + 0.08):
                first["reason"] = metric_reason
                first["source"] = str(first.get("source", "") or "") + "|bank_reason_time_metric_reason"
                if first_metrics and not first.get("supporting_metrics"):
                    first["supporting_metrics"] = list(first_metrics)[:8]
                ordered[0] = first

        diverse: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        used_clusters: set[str] = set()
        used_reasons: set[str] = set()
        for item in ordered:
            cluster = self._time_cluster(str(item.get("datetime", "") or ""))
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            if cluster and reason and (cluster in used_clusters or reason in used_reasons):
                deferred.append(item)
                continue
            diverse.append(item)
            if cluster:
                used_clusters.add(cluster)
            if reason:
                used_reasons.add(reason)
        diverse.extend(deferred)
        return diverse

    def _postprocess_component_reason_tuples(self, ordered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Final Bank-only component/reason tuple guard.

        The generic ranker can keep a gateway symptom or a weak reason tie at
        rank 1.  At this final tuple layer we only promote close challengers
        when they are supported by component-local metrics.
        """
        if not ordered:
            return ordered
        top = ordered[0]
        top_score = self._safe_float(top.get("score", 0.0))
        top_comp = self.parse_component_id(str(top.get("component", "") or ""))
        top_reason = self.parse_root_cause_label(str(top.get("reason", "") or ""))
        top_type = self.component_semantic_type(top_comp)
        top_metrics = [str(x) for x in list(top.get("supporting_metrics") or top.get("evidence") or [])]
        top_support = self._reason_tie_support(component=top_comp, reason=top_reason, top_metrics=top_metrics)

        for idx, item in enumerate(ordered[1:14], start=1):
            comp = self.parse_component_id(str(item.get("component", "") or ""))
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            if not comp or not reason:
                continue
            score = self._safe_float(item.get("score", 0.0))
            if score < top_score - 0.14:
                continue
            ctype = self.component_semantic_type(comp)
            metrics = [str(x) for x in list(item.get("supporting_metrics") or item.get("evidence") or [])]
            support = self._reason_tie_support(component=comp, reason=reason, top_metrics=metrics)

            if comp == top_comp and reason != top_reason:
                if support >= max(0.34, top_support + 0.10):
                    chosen = ordered.pop(idx)
                    chosen["source"] = str(chosen.get("source", "") or "") + "|bank_final_same_component_reason"
                    ordered.insert(0, chosen)
                    break
                continue

        diverse: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        used_pairs: set[Tuple[str, str]] = set()
        for item in ordered:
            comp = self.parse_component_id(str(item.get("component", "") or ""))
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            key = (comp, reason)
            if comp and reason and key in used_pairs:
                deferred.append(item)
                continue
            diverse.append(item)
            if comp and reason:
                used_pairs.add(key)
        diverse.extend(deferred)
        return diverse

    def _reason_tie_support(self, *, component: str, reason: str, top_metrics: Sequence[str]) -> float:
        base = self._reason_metric_support(component=component, reason=reason, top_metrics=top_metrics)
        blob = " ".join(str(x or "").lower() for x in list(top_metrics or []))
        r = _norm_token(self.parse_root_cause_label(str(reason or "")))
        bonus = 0.0
        if "high cpu usage" in r and any(tok in blob for tok in ("cpu_cpu", "cpuload", "singlecpu", "cpuutil", "cpuusertime")):
            bonus += 0.28
        if "high jvm cpu" in r and any(tok in blob for tok in ("jvm_cpu", "jvm_jvm_cpuload", "gc", "parnew")):
            bonus += 0.28
        if "memory" in r and any(tok in blob for tok in ("memory", "heap", "mem")):
            bonus += 0.18
        if "disk i/o read" in r and any(tok in blob for tok in ("dskread", "dskrtps", "query_time", "read")):
            bonus += 0.28
        if "network latency" in r and any(tok in blob for tok in ("latency", "delay", "mrt", "rtt", "fin-wait")):
            bonus += 0.24
        if "network packet loss" in r and any(tok in blob for tok in ("packet", "loss", "reset", "timeout", "fin-wait")):
            bonus += 0.22
        return min(1.0, base + bonus)

    def _dominant_reason_from_metrics(self, metrics: Sequence[str]) -> str:
        blob = " ".join(str(x or "").lower() for x in list(metrics or []) if str(x or "").strip())
        if not blob:
            return ""
        counts = {
            "high disk I/O read usage": sum(blob.count(tok) for tok in ("dskread", "dskrtps", "dskreadwrite", "query_time", "lock_time", "handler_read", "select")),
            "high disk space usage": sum(blob.count(tok) for tok in ("filesystem", "space", "capacity", "usedpercent", "diskused")),
            "high memory usage": sum(blob.count(tok) for tok in ("memory", "mem", "heapmemory", "used_memory", "nocachemem", "usermem")),
            "network latency": sum(blob.count(tok) for tok in ("latency", "delay", "mrt", "rtt", "fin-wait", "close-wait")),
            "network packet loss": sum(blob.count(tok) for tok in ("packet", "loss", "reset", "timeout", "refused", "broken")),
            "high CPU usage": sum(blob.count(tok) for tok in ("cpuutil", "cpuload", "singlecpu", "cpuusertime", "cpusystime")),
            "high JVM CPU load": sum(blob.count(tok) for tok in ("jvm_cpu", "jvm_jvm_cpuload", "parnew", "full gc")),
            "JVM Out of Memory (OOM) Heap": sum(blob.count(tok) for tok in ("outofmemory", "oom", "heapmemoryusage", "heapmemoryused")),
        }
        reason, score = max(counts.items(), key=lambda kv: kv[1])
        if score <= 0:
            return ""
        sorted_scores = sorted(counts.values(), reverse=True)
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0
        if score < max(2, second + 1):
            return ""
        return reason

    def _postprocess_component_time_tuples(self, ordered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not ordered:
            return ordered
        top = ordered[0]
        top_score = self._safe_float(top.get("score", 0.0))
        top_comp = self.parse_component_id(str(top.get("component", "") or ""))
        top_type = self.component_semantic_type(top_comp)
        top_boundary = self._is_boundary_time(str(top.get("datetime", "") or ""))

        for idx, item in enumerate(ordered[1:10], start=1):
            comp = self.parse_component_id(str(item.get("component", "") or ""))
            ctype = self.component_semantic_type(comp)
            if not comp or comp == top_comp:
                continue
            score = self._safe_float(item.get("score", 0.0))
            boundary = self._is_boundary_time(str(item.get("datetime", "") or ""))
            close = score >= top_score - (0.14 if top_type == "gateway" else 0.08)
            if not close:
                continue
            if top_type == "gateway" and ctype in {"app", "db", "cache", "web"}:
                chosen = ordered.pop(idx)
                chosen["source"] = str(chosen.get("source", "") or "") + "|bank_final_component_time_local"
                ordered.insert(0, chosen)
                break
            if top_boundary and not boundary:
                chosen = ordered.pop(idx)
                chosen["source"] = str(chosen.get("source", "") or "") + "|bank_final_component_time_non_boundary"
                ordered.insert(0, chosen)
                break

        diverse: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        used_components: set[str] = set()
        used_clusters: set[str] = set()
        for item in ordered:
            comp = self.parse_component_id(str(item.get("component", "") or ""))
            cluster = self._time_cluster(str(item.get("datetime", "") or ""))
            if (comp and comp in used_components) or (cluster and cluster in used_clusters):
                deferred.append(item)
                continue
            diverse.append(item)
            if comp:
                used_components.add(comp)
            if cluster:
                used_clusters.add(cluster)
        diverse.extend(deferred)
        return diverse

    @staticmethod
    def _time_cluster(dt: str) -> str:
        text = str(dt or "").strip()
        m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}):(\d{2}):", text)
        if not m:
            return text
        try:
            minute = int(m.group(2))
        except Exception:
            minute = 0
        return f"{m.group(1)}:{(minute // 5) * 5:02d}"

    @staticmethod
    def _is_boundary_time(dt: str) -> bool:
        text = str(dt or "").strip()
        m = re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:(\d{2}):", text)
        if not m:
            return False
        try:
            minute = int(m.group(1))
        except Exception:
            return False
        return minute in {0, 30}

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

    def metric_rank_evidence(self, *, component: str, metrics: Sequence[str]) -> Dict[str, float]:
        ctype = self.component_semantic_type(str(component or ""))
        rows = [str(m or "").lower() for m in list(metrics or []) if str(m or "").strip()]
        if not rows:
            return {"metric_score": 0.0, "ownership": 0.0, "symptomness": 0.0}

        owned_tokens = {
            "app": ("tomcat", "jvm", "heap", "thread", "request"),
            "db": ("mysql", "innodb", "sql", "query", "threadsrunning", "trx", "handler"),
            "cache": ("redis", "used_memory", "instantaneous", "mem_fragmentation"),
            "web": ("apache", "httpd", "http", "request"),
            "gateway": ("network", "net", "tcp", "ens160", "packet", "latency"),
        }.get(ctype, ())
        generic_tokens = (
            "trace_duration",
            "trace",
            "oslinux",
            "localdisk",
            "cpu_cpu",
            "filesystem",
            "zabbix",
            "ntp",
        )

        owned = 0
        generic = 0
        for metric in rows:
            if any(tok in metric for tok in owned_tokens):
                owned += 1
            if any(tok in metric for tok in generic_tokens):
                generic += 1
        n = max(1, len(rows))
        ownership = float(owned) / float(n)
        metric_score = min(1.0, float(owned) / 3.0)
        symptomness = 0.0
        if generic:
            symptomness = float(generic) / float(n)
        if ownership > 0:
            symptomness *= max(0.0, 1.0 - ownership)
        return {
            "metric_score": float(metric_score),
            "ownership": float(ownership),
            "symptomness": float(max(0.0, min(1.0, symptomness))),
        }

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
        log_ev_map = self._log_component_evidence_map(context=context)
        adjusted: List[Dict[str, Any]] = []
        for item in rows:
            comp = str(item.get("component", "") or "")
            ctype = str(item.get("component_type", "") or self.component_semantic_type(comp))
            reason = str(item.get("reason", "") or "")
            metrics = list(item.get("supporting_metrics", []) or top_metrics.get(comp, []) or [])
            local_ev = self.component_local_metric_evidence(component=comp, context=context)
            local_metrics = [str(x) for x in list(local_ev.get("top_metrics", []) or []) if str(x).strip()]
            if local_metrics and not metrics:
                metrics = local_metrics
            latent_ev: Dict[str, Any] = {}
            scoring_reason = reason
            if not scoring_reason and metrics:
                latent_ev = self.latent_reason_metric_score(component=comp, metrics=metrics)
                scoring_reason = self.parse_root_cause_label(str(latent_ev.get("reason", "") or ""))
            if int(task_id) == 4 and reason and metrics:
                dominant_reason = self._dominant_reason_from_metrics(metrics)
                if dominant_reason and dominant_reason != reason:
                    if dominant_reason == "high disk I/O read usage":
                        reason = dominant_reason
                    elif dominant_reason == "high memory usage" and ctype == "cache":
                        reason = dominant_reason
                    elif reason == "high JVM CPU load" and ctype == "gateway" and dominant_reason == "high CPU usage":
                        reason = dominant_reason
            rank_ev = self.metric_rank_evidence(component=comp, metrics=metrics)
            log_ev = dict(log_ev_map.get(comp, {}) or {})
            symptomness = max(0.0, min(1.0, self._safe_float(item.get("symptomness", 0.0))))
            if rank_ev.get("symptomness", 0.0):
                symptomness = max(symptomness, self._safe_float(rank_ev.get("symptomness", 0.0)))
            if local_ev.get("symptomness", 0.0):
                symptomness = max(symptomness, self._safe_float(local_ev.get("symptomness", 0.0)))
            log_score = self._safe_float(log_ev.get("log_score", 0.0))
            log_ownership = self._safe_float(log_ev.get("ownership", 0.0))
            if log_score > 0.0:
                metrics = list(dict.fromkeys(list(metrics or []) + [str(x) for x in list(log_ev.get("top_logs", []) or [])]))[:8]
            score = self.component_evidence_score(
                component=comp,
                base_score=self._safe_float(item.get("score", 0.0)),
                context=context,
                metric_score=max(
                    self._safe_float(rank_ev.get("metric_score", 0.0)),
                    self._safe_float(local_ev.get("metric_score", 0.0)),
                    0.65 * log_score,
                ),
                ownership_score=max(
                    self._safe_float(rank_ev.get("ownership", 0.0)),
                    self._safe_float(local_ev.get("ownership", 0.0)),
                    0.50 * log_ownership,
                ),
                symptomness=symptomness,
                locality=self._safe_float(local_ev.get("locality", 0.0)),
                persistence=max(self._safe_float(local_ev.get("persistence", 0.0)), 0.50 * log_score),
                instance_owned_score=self._safe_float(local_ev.get("instance_owned_score", 0.0)),
                sibling_contrast=self._safe_float(local_ev.get("sibling_contrast", 0.0)),
            )
            rootness_features = self._component_rootness_features(
                component=comp,
                base_score=self._safe_float(item.get("score", 0.0)),
                context=context,
                metric_score=max(
                    self._safe_float(rank_ev.get("metric_score", 0.0)),
                    self._safe_float(local_ev.get("metric_score", 0.0)),
                    0.65 * log_score,
                ),
                ownership_score=max(
                    self._safe_float(rank_ev.get("ownership", 0.0)),
                    self._safe_float(local_ev.get("ownership", 0.0)),
                    0.50 * log_ownership,
                ),
                symptomness=symptomness,
                locality=self._safe_float(local_ev.get("locality", 0.0)),
                persistence=max(self._safe_float(local_ev.get("persistence", 0.0)), 0.50 * log_score),
                instance_owned_score=self._safe_float(local_ev.get("instance_owned_score", 0.0)),
                sibling_contrast=self._safe_float(local_ev.get("sibling_contrast", 0.0)),
            )
            if scoring_reason:
                reason_support = self._reason_metric_support(component=comp, reason=scoring_reason, top_metrics=metrics)
                score += 0.12 * reason_support
                if not reason:
                    # Component-only and component-time tasks do not output a
                    # reason, but latent reason evidence helps separate true
                    # local roots from shared gateway/OS symptoms.
                    score += 0.08 * self._safe_float(latent_ev.get("score", 0.0))
                    expected_types = set(self.reason_compatible_component_types(scoring_reason) or [])
                    if expected_types and ctype in expected_types:
                        score += 0.05 * reason_support
                    elif expected_types and ctype and ctype not in expected_types:
                        score -= 0.05 * reason_support
            item = dict(item)
            item["reason"] = reason
            if latent_ev and not item.get("latent_reason"):
                item["latent_reason"] = str(scoring_reason or "")
                item["latent_reason_score"] = float(self._safe_float(latent_ev.get("score", 0.0)))
            item["score"] = float(score)
            item["component_type"] = ctype
            item["metric_ownership"] = float(rank_ev.get("ownership", 0.0) or 0.0)
            item["metric_ownership_details"] = dict(rank_ev)
            if log_ev:
                item["log_component_evidence"] = dict(log_ev)
            item["rootness_features"] = dict(rootness_features)
            if metrics and not item.get("supporting_metrics"):
                item["supporting_metrics"] = list(metrics)[:8]
            adjusted.append(item)

        adjusted.sort(key=lambda x: self._safe_float(x.get("score", 0.0)), reverse=True)
        limit = max(1, int(k))
        if int(task_id) in {5, 7}:
            pool_limit = max(limit, 56)
        elif int(task_id) == 4:
            pool_limit = max(limit, 40)
        else:
            pool_limit = max(limit, 24)
        top_m_per_channel = 4 if int(task_id) in {5, 7} else 3
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

        if int(task_id) in {5, 7}:
            by_component: Dict[str, List[Dict[str, Any]]] = {}
            for item in adjusted:
                comp = str(item.get("component", "") or "")
                if not comp or not str(item.get("datetime", "") or "").strip():
                    continue
                by_component.setdefault(comp, []).append(item)
            for comp in sorted(by_component):
                for item in by_component.get(comp, [])[:2]:
                    keep(item)

        if int(task_id) == 7:
            for item in self._recombine_task7_dimension_candidates(adjusted, context=context):
                keep(item)

        if int(task_id) in {4, 7}:
            by_reason: Dict[str, List[Dict[str, Any]]] = {}
            for item in adjusted:
                reason0 = str(item.get("reason", "") or "")
                if not reason0 or not str(item.get("datetime", "") or "").strip():
                    continue
                by_reason.setdefault(reason0, []).append(item)
            for reason0 in sorted(by_reason):
                for item in by_reason.get(reason0, [])[:2]:
                    keep(item)

        for item in adjusted:
            keep(item)
            if len(selected_out) >= pool_limit:
                break
        return selected_out

    def _recombine_task7_dimension_candidates(
        self,
        adjusted: Sequence[Dict[str, Any]],
        *,
        context: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """Task7 needs exact component/reason/time tuples.

        The index often surfaces the right dimensions separately: a candidate
        may have the right component+reason but a shifted time, while another
        has the right time but a symptom component.  This telemetry-only
        recombination keeps bounded cross-products of already surfaced
        dimensions so the final ranker has a chance to choose a complete tuple.
        """
        rows = [dict(x) for x in list(adjusted or []) if isinstance(x, Mapping)]
        if not rows:
            return []

        def score_of(item: Mapping[str, Any]) -> float:
            return self._safe_float(item.get("score", 0.0))

        comps: List[Tuple[str, float, List[str], str]] = []
        seen_comp: set[str] = set()
        for item in sorted(rows, key=score_of, reverse=True):
            comp = self.parse_component_id(str(item.get("component", "") or ""))
            if not comp or comp in seen_comp:
                continue
            seen_comp.add(comp)
            comps.append(
                (
                    comp,
                    score_of(item),
                    [str(x) for x in list(item.get("supporting_metrics") or [])],
                    str(item.get("component_type", "") or self.component_semantic_type(comp)),
                )
            )
            if len(comps) >= 8:
                break

        reasons: List[Tuple[str, float]] = []
        seen_reason: set[str] = set()
        reason_scores = {
            self.parse_root_cause_label(str(k)): self._safe_float(v)
            for k, v in dict((context or {}).get("reason_scores", {}) or {}).items()
        }
        for item in sorted(rows, key=score_of, reverse=True):
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            if not reason or reason in seen_reason:
                continue
            seen_reason.add(reason)
            reasons.append((reason, max(score_of(item), self._safe_float(reason_scores.get(reason, 0.0)))))
            if len(reasons) >= 5:
                break
        for reason, value in sorted(reason_scores.items(), key=lambda kv: kv[1], reverse=True):
            if reason and reason not in seen_reason:
                seen_reason.add(reason)
                reasons.append((reason, value))
            if len(reasons) >= 5:
                break

        times: List[Tuple[str, float, str]] = []
        seen_time: set[str] = set()
        for item in sorted(rows, key=score_of, reverse=True):
            dt = str(item.get("datetime", "") or "").strip()
            if not dt or dt in seen_time:
                continue
            seen_time.add(dt)
            times.append((dt, score_of(item), str(item.get("source", "") or "")))
            if len(times) >= 8:
                break

        out: List[Dict[str, Any]] = []
        if not comps or not reasons or not times:
            return out
        for comp, comp_score, metrics, ctype in comps:
            for reason, reason_score in reasons:
                expected = set(self.reason_compatible_component_types(reason) or [])
                compat = 1.0
                if expected and ctype:
                    compat = 1.05 if ctype in expected else 0.80
                support = self._reason_tie_support(component=comp, reason=reason, top_metrics=metrics)
                for dt, time_score, source in times[:6]:
                    score = 0.46 * comp_score + 0.30 * reason_score + 0.24 * time_score
                    score = score * compat + 0.10 * support
                    out.append(
                        {
                            "component": comp,
                            "reason": reason,
                            "datetime": dt,
                            "score": float(score),
                            "source": f"bank_task7_dimension_recombine:{source}",
                            "component_type": ctype,
                            "supporting_metrics": list(metrics)[:8],
                        }
                    )
        out.sort(key=lambda x: self._safe_float(x.get("score", 0.0)), reverse=True)
        return out[:32]

    def _log_component_evidence_map(self, *, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Window-local component evidence from Bank service logs.

        Counts are compared with a same-length pre-window baseline, so common
        background GC/slow logs do not automatically dominate unless they spike
        inside the incident window.
        """
        import pandas as pd

        dataset = str((context or {}).get("dataset", "") or self.dataset)
        date_dir = str((context or {}).get("date_dir", "") or "")
        start_local = str((context or {}).get("query_start_local", "") or "")
        end_local = str((context or {}).get("query_end_local", "") or "")
        if not dataset or not date_dir or not start_local or not end_local:
            return {}
        cache_key = (dataset, date_dir, start_local, end_local)
        cache = getattr(self, "_bank_log_component_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_bank_log_component_cache", cache)
        if cache_key in cache:
            return dict(cache[cache_key])
        try:
            tz = str(self.config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
            start_s = int(pd.Timestamp(start_local, tz=tz).tz_convert("UTC").timestamp())
            end_s = int(pd.Timestamp(end_local, tz=tz).tz_convert("UTC").timestamp())
        except Exception:
            cache[cache_key] = {}
            return {}
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        window_s = max(60, int(end_s - start_s))
        before_start_s = int(start_s - window_s)
        log_path = os.path.join("dataset", dataset, "telemetry", date_dir, "log", "log_service.csv")
        if not os.path.exists(log_path):
            cache[cache_key] = {}
            return {}

        win_counts: Dict[str, float] = {}
        before_counts: Dict[str, float] = {}
        log_names: Dict[str, Dict[str, float]] = {}
        try:
            chunks = pd.read_csv(
                log_path,
                usecols=lambda c: c in {"timestamp", "cmdb_id", "log_name"},
                chunksize=200_000,
                low_memory=False,
            )
            for chunk in chunks:
                if "timestamp" not in chunk.columns or "cmdb_id" not in chunk.columns:
                    continue
                ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
                use = chunk[(ts >= before_start_s) & (ts <= end_s)].copy()
                if use.empty:
                    continue
                use["component"] = use["cmdb_id"].map(lambda x: self.parse_component_id(str(x)))
                use = use[use["component"].astype(str).str.len() > 0].copy()
                if use.empty:
                    continue
                use["timestamp_num"] = pd.to_numeric(use["timestamp"], errors="coerce")
                before = use[(use["timestamp_num"] >= before_start_s) & (use["timestamp_num"] < start_s)]
                win = use[(use["timestamp_num"] >= start_s) & (use["timestamp_num"] <= end_s)]
                for comp, value in before.groupby("component").size().items():
                    before_counts[str(comp)] = before_counts.get(str(comp), 0.0) + float(value)
                for comp, value in win.groupby("component").size().items():
                    win_counts[str(comp)] = win_counts.get(str(comp), 0.0) + float(value)
                if "log_name" in win.columns and not win.empty:
                    for (comp, name), value in win.groupby(["component", "log_name"]).size().items():
                        log_names.setdefault(str(comp), {})
                        log_names[str(comp)][str(name)] = log_names[str(comp)].get(str(name), 0.0) + float(value)
        except Exception:
            cache[cache_key] = {}
            return {}

        comps = set(win_counts) | set(before_counts)
        if not comps:
            cache[cache_key] = {}
            return {}
        raw: Dict[str, float] = {}
        for comp in comps:
            win_rate = float(win_counts.get(comp, 0.0)) / float(max(1, window_s // 60))
            before_rate = float(before_counts.get(comp, 0.0)) / float(max(1, window_s // 60))
            raw[comp] = max(0.0, win_rate - before_rate)
        mx = max([abs(v) for v in raw.values()] + [0.0])
        if mx <= 1e-12:
            cache[cache_key] = {}
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for comp, value in raw.items():
            if value <= 0:
                continue
            top_logs = [
                f"log:{name}"
                for name, _cnt in sorted(log_names.get(comp, {}).items(), key=lambda kv: kv[1], reverse=True)[:3]
            ]
            score = max(0.0, min(1.0, float(value) / mx))
            out[comp] = {"log_score": score, "ownership": score, "top_logs": top_logs}
        cache[cache_key] = dict(out)
        return out

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
        return 0.0

    # select_time_candidate_override: uses base score-based implementation.
    # Bank's exact comp+reason match is expressed as score bonuses in _score_time_candidate.

    def _score_time_candidate(self, item: Dict[str, Any], comp: str, reason: str) -> float:
        score = float(item.get("score", 0.0) or 0.0)
        source = str(item.get("source", "") or "").lower()
        if "metric_change_point" in source:
            score += 0.42
        if "window" in source or "fallback" in source:
            score -= 0.25
        if "trace" in source and not comp:
            score -= 0.12
        c0 = self.parse_component_id(str(item.get("component", "") or ""))
        r0 = self.parse_root_cause_label(str(item.get("reason", "") or ""))
        if comp and c0:
            score += 1.0 if c0 == comp else -1.0
        if reason and r0:
            score += 0.5 if r0 == reason else -0.5
        return score

    def _filter_time_candidates_to_query_window(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        instruction: str,
        tolerance_s: int = 90,
    ) -> List[Dict[str, Any]]:
        try:
            import pandas as pd
            from rca.adapter.utils.component_candidates import parse_time_window_generic

            _date, start, end = parse_time_window_generic(str(instruction or ""))
            tz = str(self.config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
            start_s = int(start.tz_convert("UTC").timestamp()) - int(tolerance_s)
            end_s = int(end.tz_convert("UTC").timestamp()) + int(tolerance_s)
        except Exception:
            return [dict(x) for x in list(candidates or []) if isinstance(x, Mapping)]

        out: List[Dict[str, Any]] = []
        best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for item in [dict(x) for x in list(candidates or []) if isinstance(x, Mapping)]:
            dt = str(item.get("datetime", "") or "").strip()
            if not dt:
                continue
            try:
                ts = int(pd.Timestamp(dt, tz=tz).tz_convert("UTC").timestamp())
            except Exception:
                continue
            if ts < start_s or ts > end_s:
                continue
            key = (dt, str(item.get("component", "") or ""), str(item.get("source", "") or ""))
            if key not in best or self._safe_float(item.get("score", 0.0)) > self._safe_float(best[key].get("score", 0.0)):
                best[key] = item
        ranked = sorted(best.values(), key=lambda x: self._safe_float(x.get("score", 0.0)), reverse=True)
        diverse: List[Dict[str, Any]] = []
        seen_keys: set[Tuple[str, str]] = set()
        for item in ranked:
            comp = self.parse_component_id(str(item.get("component", "") or ""))
            source = str(item.get("source", "") or "")
            key = (comp, source)
            if comp and key in seen_keys:
                continue
            diverse.append(item)
            if comp:
                seen_keys.add(key)
            if len(diverse) >= 32:
                break
        seen_dt = {(str(x.get("datetime", "") or ""), str(x.get("component", "") or ""), str(x.get("source", "") or "")) for x in diverse}
        for item in ranked:
            key = (str(item.get("datetime", "") or ""), str(item.get("component", "") or ""), str(item.get("source", "") or ""))
            if key in seen_dt:
                continue
            diverse.append(item)
            seen_dt.add(key)
            if len(diverse) >= 48:
                break
        return diverse

    def _metric_change_point_time_candidates(self, *, instruction: str) -> List[Dict[str, Any]]:
        """Telemetry-only time candidates from window-local before/after change.

        This is a candidate-recall supplement for time-only and time-bearing
        tasks.  It does not use query labels or record history; it only asks
        which minute best separates weak-before from persistent-strong-after
        anomaly evidence in the incident window.
        """
        import pandas as pd
        from rca.adapter.utils.component_candidates import _normalize_metric_file, parse_time_window_generic
        from rca.common.io import resolve_date_dir

        try:
            date_dir, start, end = parse_time_window_generic(str(instruction or ""))
            date_dir = resolve_date_dir(self.dataset, date_dir)
            start_s = int(start.tz_convert("UTC").timestamp())
            end_s = int(end.tz_convert("UTC").timestamp())
        except Exception:
            return []
        metric_dir = os.path.join("dataset", self.dataset, "telemetry", date_dir, "metric")
        if not os.path.isdir(metric_dir):
            return []

        minute_scores: Dict[int, float] = {}
        minute_components: Dict[int, Dict[str, float]] = {}
        for name in os.listdir(metric_dir):
            if not str(name).endswith(".csv"):
                continue
            path = os.path.join(metric_dir, name)
            try:
                long = _normalize_metric_file(path)
            except Exception:
                continue
            if long.empty:
                continue
            try:
                p95 = (
                    long.groupby(["component", "metric"], as_index=False)["value"]
                    .quantile(0.95)
                    .rename(columns={"value": "p95"})
                )
                win = long[(long["ts_s"] >= start_s) & (long["ts_s"] <= end_s)].copy()
                if win.empty:
                    continue
                win = win.merge(p95, on=["component", "metric"], how="left").dropna(subset=["p95"]).copy()
                denom = win["p95"].abs().where(win["p95"].abs() > 1e-6, 1.0)
                win["breach_ratio"] = ((win["value"] - win["p95"]) / denom).clip(lower=0.0, upper=10.0)
                win = win[win["breach_ratio"] > 0].copy()
                if win.empty:
                    continue
                win["component"] = win["component"].map(lambda x: self.parse_component_id(str(x)))
                win = win[win["component"].astype(str).str.len() > 0].copy()
                win["minute"] = (win["ts_s"].astype(int) // 60) * 60
            except Exception:
                continue
            by_min = win.groupby("minute")["breach_ratio"].sum()
            for minute, value in by_min.items():
                m = int(minute)
                minute_scores[m] = minute_scores.get(m, 0.0) + float(value)
            by_comp_min = win.groupby(["minute", "component"])["breach_ratio"].sum()
            for (minute, comp), value in by_comp_min.items():
                m = int(minute)
                minute_components.setdefault(m, {})
                c = str(comp)
                minute_components[m][c] = minute_components[m].get(c, 0.0) + float(value)

        if not minute_scores:
            return []
        minutes = sorted(minute_scores)
        if len(minutes) < 2:
            return []
        values = [float(minute_scores[m]) for m in minutes]
        mx = max(values) if values else 0.0
        if mx <= 1e-9:
            return []

        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, minute in enumerate(minutes):
            before = values[max(0, idx - 3):idx]
            after = values[idx:min(len(values), idx + 4)]
            if not after:
                continue
            before_mean = sum(before) / float(max(1, len(before))) if before else 0.0
            after_mean = sum(after) / float(max(1, len(after)))
            contrast = max(0.0, after_mean - before_mean)
            persistence = sum(1 for v in after if v >= 0.35 * mx) / float(max(1, len(after)))
            local_components = minute_components.get(minute, {})
            top_comp = ""
            if local_components:
                top_comp = max(local_components, key=lambda c: local_components[c])
            score = 0.70 * (contrast / mx) + 0.30 * persistence
            if score <= 0.0:
                continue
            scored.append((float(score), int(minute), {"contrast": float(contrast), "persistence": float(persistence), "component": top_comp}))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        seen_minutes: set[int] = set()
        for score, minute, ev in scored:
            if any(abs(minute - old) <= 60 for old in seen_minutes):
                continue
            seen_minutes.add(minute)
            dt = pd.to_datetime(int(minute), unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
            out.append(
                {
                    "datetime": dt,
                    "source": "metric_change_point",
                    "score": 0.82 + min(0.16, 0.16 * float(score)),
                    "component": str(ev.get("component", "") or ""),
                    "reason": "",
                    "evidence": dict(ev),
                }
            )
            if len(out) >= 6:
                break
        return out

    def _log_reason_scores(self, *, instruction: str) -> Dict[str, float]:
        import pandas as pd
        from rca.adapter.utils.component_candidates import parse_time_window_generic
        from rca.common.io import resolve_date_dir

        try:
            date_dir, start, end = parse_time_window_generic(str(instruction or ""))
            date_dir = resolve_date_dir(self.dataset, date_dir)
            start_s = int(start.tz_convert("UTC").timestamp())
            end_s = int(end.tz_convert("UTC").timestamp())
        except Exception:
            return {}
        log_path = os.path.join("dataset", self.dataset, "telemetry", date_dir, "log", "log_service.csv")
        if not os.path.exists(log_path):
            return {}

        reason_tokens = {
            "JVM Out of Memory (OOM) Heap": ["outofmemory", "oom", "java heap space", "gc overhead", "heap"],
            "high JVM CPU load": ["gc", "full gc", "jvm", "parnew", "user=", "real="],
            "high CPU usage": ["cpu", "load", "busy"],
            "high disk I/O read usage": ["slow", "query_time", "lock_time", "read", "select"],
            "high disk space usage": ["no space", "disk", "filesystem", "capacity"],
            "high memory usage": ["memory", "mem", "heap", "gc"],
            "network latency": ["timeout", "timed out", "latency", "delay", "connect"],
            "network packet loss": ["reset", "broken pipe", "connection refused", "connection reset", "packet", "loss", "timeout"],
        }
        raw_scores = {r: 0.0 for r in self.basil_reason_taxonomy()}
        usecols = ["timestamp", "log_name", "value"]
        try:
            chunks = pd.read_csv(log_path, usecols=lambda c: c in set(usecols), chunksize=200_000, low_memory=False)
            for chunk in chunks:
                if "timestamp" not in chunk.columns:
                    continue
                ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
                chunk = chunk[(ts >= start_s) & (ts <= end_s)].copy()
                if chunk.empty:
                    continue
                text = (
                    chunk.get("log_name", "").astype(str).str.lower()
                    + " "
                    + chunk.get("value", "").astype(str).str.lower()
                )
                text = text.str.replace(r"[^a-z0-9=._/ -]+", " ", regex=True)
                joined = "\n".join(text.head(5000).tolist())
                for reason, tokens in reason_tokens.items():
                    score = 0.0
                    for tok in tokens:
                        if tok in joined:
                            score += 1.0
                    raw_scores[reason] = raw_scores.get(reason, 0.0) + score
        except Exception:
            return {}
        mx = max([abs(v) for v in raw_scores.values()] + [0.0])
        if mx <= 1e-12:
            return {}
        return {k: max(0.0, float(v) / mx) for k, v in raw_scores.items()}

    def _log_reason_scores_for_context(self, context: Mapping[str, Any]) -> Dict[str, float]:
        import pandas as pd
        from rca.common.io import resolve_date_dir

        date_dir = str((context or {}).get("date_dir", "") or "").strip()
        start_text = str((context or {}).get("query_start_local", "") or "").strip()
        end_text = str((context or {}).get("query_end_local", "") or "").strip()
        if not date_dir or not start_text or not end_text:
            return {}
        try:
            date_dir = resolve_date_dir(self.dataset, date_dir)
            start = pd.to_datetime(start_text)
            end = pd.to_datetime(end_text)
            if getattr(start, "tzinfo", None) is None:
                start = start.tz_localize("Asia/Shanghai")
            if getattr(end, "tzinfo", None) is None:
                end = end.tz_localize("Asia/Shanghai")
            start_s = int(start.tz_convert("UTC").timestamp())
            end_s = int(end.tz_convert("UTC").timestamp())
        except Exception:
            return {}
        log_path = os.path.join("dataset", self.dataset, "telemetry", date_dir, "log", "log_service.csv")
        if not os.path.exists(log_path):
            return {}

        reason_tokens = {
            "JVM Out of Memory (OOM) Heap": ["outofmemory", "oom", "java heap space", "gc overhead", "heap"],
            "high JVM CPU load": ["gc", "full gc", "jvm", "parnew", "user=", "real="],
            "high CPU usage": ["cpu", "load", "busy"],
            "high disk I/O read usage": ["slow", "query_time", "lock_time", "read", "select"],
            "high disk space usage": ["no space", "disk", "filesystem", "capacity"],
            "high memory usage": ["memory", "mem", "heap", "gc"],
            "network latency": ["timeout", "timed out", "latency", "delay", "connect"],
            "network packet loss": ["reset", "broken pipe", "connection refused", "connection reset", "packet", "loss", "timeout"],
        }
        raw_scores = {r: 0.0 for r in self.basil_reason_taxonomy()}
        try:
            chunks = pd.read_csv(log_path, usecols=lambda c: c in {"timestamp", "log_name", "value"}, chunksize=200_000, low_memory=False)
            for chunk in chunks:
                if "timestamp" not in chunk.columns:
                    continue
                ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
                chunk = chunk[(ts >= start_s) & (ts <= end_s)].copy()
                if chunk.empty:
                    continue
                text = (
                    chunk.get("log_name", "").astype(str).str.lower()
                    + " "
                    + chunk.get("value", "").astype(str).str.lower()
                )
                text = text.str.replace(r"[^a-z0-9=._/ -]+", " ", regex=True)
                joined = "\n".join(text.head(5000).tolist())
                for reason, tokens in reason_tokens.items():
                    raw_scores[reason] = raw_scores.get(reason, 0.0) + sum(1.0 for tok in tokens if tok in joined)
        except Exception:
            return {}
        mx = max([abs(v) for v in raw_scores.values()] + [0.0])
        if mx <= 1e-12:
            return {}
        return {k: max(0.0, float(v) / mx) for k, v in raw_scores.items()}

    # _component_prior: inherited neutral prior.  No query/record label priors.

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
