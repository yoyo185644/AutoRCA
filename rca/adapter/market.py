from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rca.common.text import norm_token, service_base
from rca.adapter.base import DatasetAdapter


_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def _parse_local_time_s(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(datetime.strptime(text, "%Y-%m-%d %H:%M:%S").timestamp())
    except Exception:
        return 0


def _market_component_level(component: str) -> str:
    c = str(component or "").strip()
    if re.fullmatch(r"node-\d+", c):
        return "node"
    if re.search(r"-\d+$", c):
        return "pod"
    if c:
        return "service"
    return ""


def _market_semantic_family(component: str) -> str:
    c = re.sub(r"^node-\d+\.", "", str(component or "").strip())
    c = re.sub(r"(\.ts|-grpc)$", "", c)
    if re.fullmatch(r"node-\d+", c):
        return c
    base = service_base(c)
    base = re.sub(r"([a-z]+service)\d+$", r"\1", base)
    base = re.sub(r"^frontend\d+$", "frontend", base)
    return base or c


def _is_trace_edge_component(value: str) -> bool:
    c = str(value or "").strip().lower()
    if not c:
        return False
    if re.search(r"\.(source|destination|unknown)(\.|$)", c):
        return True
    if c in {"source", "destination", "unknown"}:
        return True
    return False


class MarketAdapter(DatasetAdapter):
    """Market/cloudbed adapter.

    This adapter is intentionally isolated from Telecom. It normalizes Market
    component aliases, maps Market reason taxonomy, and lets BASIL/agent code
    rerank only inside Market candidate sets.
    """

    _REASON_ALIASES: Dict[str, str] = {
        "node cpu spike": "node CPU load",
        "node cpu usage": "node CPU load",
        "node memory load": "node memory consumption",
        "container packet loss": "container network packet retransmission",
        "container network packet loss": "container network packet retransmission",
        "container packet retransmission": "container network packet retransmission",
        "container packet corruption": "container network packet corruption",
        "network packet retransmission": "container network packet retransmission",
        "network packet corruption": "container network packet corruption",
    }

    def solve_reason(
        self,
        *,
        spec: Any,
        instruction: str,
        args: Any,
        need_reason: bool,
        **kwargs: Any,
    ) -> Optional[Tuple[str, str, bool, Any]]:
        return None

    def get_component_delta_s(
        self,
        *,
        spec: Any,
        need_comp: bool,
        need_time: bool,
        center_time_s: Optional[int],
        **kwargs: Any,
    ) -> Tuple[int, Optional[int]]:
        return 300, center_time_s

    def solve_component_joint_scoring(
        self,
        *,
        ev2: Any,
        ev3: Any,
        det_reason: str,
        det_reason_ready: bool,
        **kwargs: Any,
    ) -> Optional[Tuple[str, str]]:
        return None

    def _light_agent_task_ids(self) -> set:
        return set()  # Market does not use light-agent LLM for component selection

    def get_time_dt_override(
        self,
        *,
        need_time: bool,
        comp: str,
        ev3: Any,
        **kwargs: Any,
    ) -> Optional[str]:
        return None

    def parse_component_id(self, raw: str) -> str:
        c = str(raw or "").strip()
        c = re.sub(r"^node-\d+\.", "", c)
        c = re.sub(r"(\.ts|-grpc)$", "", c)
        c = re.sub(r"^frontend(\d+)-\d+$", r"frontend-\1", c)
        return c

    def component_semantic_type(self, component: str) -> str:
        level = _market_component_level(self.parse_component_id(str(component or "")))
        if level == "node":
            return "node"
        if level == "pod":
            return "container"
        if level == "service":
            return "service"
        return ""

    def reason_semantic_type(self, reason: str) -> str:
        r = norm_token(self.parse_root_cause_label(str(reason or "")))
        if r.startswith("node "):
            return "node"
        if r.startswith("container "):
            return "container"
        return ""

    def reason_metric_keywords(self, reason: str) -> List[str]:
        r = norm_token(self.parse_root_cause_label(str(reason or "")))
        if "cpu" in r:
            return ["cpu", "load", "util", "usage"]
        if "memory" in r:
            return ["mem", "memory", "rss", "cache"]
        if "read" in r:
            return ["read", "io", "i/o", "disk"]
        if "write" in r:
            return ["write", "io", "i/o", "disk"]
        if "network" in r or "latency" in r or "packet" in r:
            return ["network", "latency", "packet", "retrans", "corruption", "loss"]
        if "termination" in r or "process" in r:
            return ["restart", "exit", "kill", "process", "termination"]
        return []

    def reason_compatible_component_types(self, reason: str) -> List[str]:
        rtype = self.reason_semantic_type(str(reason or ""))
        return [rtype] if rtype else []

    def llm_reason_rules(self) -> List[str]:
        return [
            "Node reasons should align with node components and node-level metrics.",
            "Container reasons should align with pod/container components and container-level metrics.",
            "Network packet/latency reasons require network/packet/retransmission/latency evidence.",
            "Do not replace a level-compatible candidate with a level-incompatible candidate unless metric evidence is clearly stronger.",
        ]

    def canonicalize_component(self, *, component: str, component_level: str) -> str:
        raw = str(component or "").strip()
        if not raw:
            return ""
        if _is_trace_edge_component(raw):
            return ""
        node_match = re.match(r"^(node-\d+)(?:\.|$)", raw)
        parsed = self.parse_component_id(raw)
        level = str(component_level or "").strip().lower()
        if level == "node":
            return node_match.group(1) if node_match else (parsed if re.fullmatch(r"node-\d+", parsed) else parsed)
        if level == "service":
            if re.fullmatch(r"node-\d+", parsed):
                return parsed
            fam = _market_semantic_family(parsed)
            return fam or parsed
        if level == "mixed":
            if re.fullmatch(r"node-\d+", parsed):
                return parsed
            if _market_component_level(parsed) == "service":
                fam = _market_semantic_family(parsed)
                return fam or parsed
            return parsed
        # "pod" and "mixed" preserve pod-level ids while still stripping
        # node prefixes and mesh suffixes.
        if level in {"pod", "mixed", ""}:
            return parsed
        return parsed

    def parse_root_cause_label(self, raw_reason: str) -> str:
        reason = str(raw_reason or "").strip()
        if not reason:
            return ""
        key = re.sub(r"\s+", " ", reason.lower()).strip()
        if key in self._REASON_ALIASES:
            return self._REASON_ALIASES[key]
        taxonomy = self._reason_taxonomy()
        for label in taxonomy:
            if norm_token(label) == norm_token(reason):
                return str(label)
        return reason

    def basil_reason_taxonomy(self) -> List[str]:
        taxonomy = self._reason_taxonomy()
        return [str(x) for x in taxonomy.keys() if str(x).strip()]

    def extract_reason_hints(self) -> List[str]:
        return self.basil_reason_taxonomy()

    def coerce_closed_set_lists(self) -> bool:
        return True

    def supports_multi_root_joint_selection(self) -> bool:
        return True

    def select_multi_root_joint_tuples(
        self,
        *,
        items: Sequence[Dict[str, Any]],
        k: int,
        task_id: int,
        context: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        if int(task_id) != 6 or int(k) <= 1:
            return None
        pool = [dict(x) for x in list(items or []) if isinstance(x, Mapping)]
        if not pool:
            return None
        pool.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

        def is_node(comp: str) -> bool:
            return bool(re.fullmatch(r"node-\d+", str(comp or "").strip()))

        def uniq_by_component(seq: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for it in seq:
                c = str(it.get("component", "") or "").strip()
                if not c or c in seen:
                    continue
                out.append(it)
                seen.add(c)
            return out

        top = pool[: max(6, int(k) * 4)]
        nodes = uniq_by_component([x for x in top if is_node(str(x.get("component", "") or ""))])
        others = uniq_by_component([x for x in top if not is_node(str(x.get("component", "") or ""))])
        if len(nodes) >= int(k):
            best_other = float(others[0].get("score", 0.0) or 0.0) if others else -1e9
            node_scores = [float(x.get("score", 0.0) or 0.0) for x in nodes[: int(k)]]
            if node_scores and float(sum(node_scores) / max(1, len(node_scores))) >= best_other - 0.05:
                return nodes[: int(k)]
        if nodes and others and int(k) == 2:
            return [nodes[0], others[0]]
        picked = uniq_by_component(pool)[: int(k)]
        return picked or None

    def enable_structured_time_candidates(self) -> bool:
        return True

    def should_collect_component_evidence_for_time(
        self,
        *,
        task_id: int,
        need_time: bool,
        need_comp: bool,
        need_reason: bool,
    ) -> bool:
        return bool(need_time)

    def prefer_deterministic_time_only(self) -> bool:
        return True

    def get_time_candidate_top_k(self, *, task_id: int, default_top_k: int) -> int:
        return max(int(default_top_k), 12)

    def extract_component_level(self) -> str:
        return ""

    def basil_query_config_overrides(
        self,
        *,
        task_id: int,
        need_time: bool,
        need_comp: bool,
        need_reason: bool,
        component_level: str,
    ) -> Dict[str, Any]:
        idx = dict(self.config.get("index", {}) or {})
        query = dict(idx.get("query", {}) or {})
        out: Dict[str, Any] = {
            "enable_trace_segments": False,
            "metric_baseline_mode": "auto",
            "w_causal": 0.0,
            "similarity_topology_weight": 0.0,
            "similarity_pattern_weight": 0.25,
            "bucket_size_s": 60,
        }
        out["top_k"] = max(24, int(query.get("top_k") or 12)) if query.get("top_k") is not None else 24
        out["candidate_reason_top_r"] = 8
        out["rerank_top_n"] = 80
        out["include_market_service_metrics"] = True
        out["w_reason"] = 0.40 if bool(need_reason) else 0.25
        out["w_local"] = 1.10
        return out

    def filter_candidate_hypotheses(
        self,
        *,
        candidates: Sequence[Any],
        component_level: str,
        need_comp: bool,
        need_reason: bool,
        need_time: bool,
    ) -> List[Any]:
        out = []
        for cand in list(candidates or []):
            comp = self.canonicalize_component(
                component=str(getattr(cand, "component", "") or ""),
                component_level=str(component_level or ""),
            )
            reason = self.parse_root_cause_label(str(getattr(cand, "reason", "") or ""))
            if need_comp and not comp:
                continue
            if need_reason and not reason:
                continue
            try:
                cand.component = comp
                cand.reason = reason
            except Exception:
                pass
            out.append(cand)
        return out

    # rerank_candidate_hypotheses: uses base implementation.
    # Market-specific behaviour injected via hooks below.

    def _type_consistency(self, comp: str, reason: str, metrics: List[str]) -> float:
        """Market: level–reason alignment + metric keyword signal."""
        score = 0.0
        expected_level = self._expected_level_for_reason(reason)
        if expected_level:
            actual_level = _market_component_level(comp)
            if actual_level == expected_level:
                score += 0.25
            elif expected_level == "pod" and actual_level == "service":
                score -= 0.05
            else:
                score -= 0.35

        keywords = [k.lower() for k in self.reason_metric_keywords(reason)]
        if keywords:
            metric_blob = " ".join(m.lower() for m in metrics)
            hits = sum(1 for k in keywords if k in metric_blob)
            if hits > 0:
                score += min(0.25, 0.08 * hits)
            elif any(k in str(reason).lower() for k in ["cpu", "memory", "disk", "io", "network", "packet"]):
                score -= 0.10
        return score

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
        """Market applies hint bonuses only when both comp and reason hints are present."""
        if not (hint_components and hint_reasons and need_comp and need_reason):
            return 0.0
        hint_bases = {service_base(x) for x in hint_components if x}
        bonus = 0.0
        if comp in hint_components:
            bonus += 1.10
        elif service_base(comp) in hint_bases:
            bonus += 0.55
        if reason in hint_reasons:
            bonus += 0.95
        return bonus

    def _postprocess_reranked(self, candidates: List[Any]) -> List[Any]:
        """Market: diversity filter — one candidate per (semantic-family, reason) pair first."""
        out: List[Any] = []
        deferred: List[Any] = []
        used: set = set()
        for cand in list(candidates or []):
            fam = _market_semantic_family(str(getattr(cand, "component", "") or ""))
            key = (fam, str(getattr(cand, "reason", "") or ""))
            if key in used:
                deferred.append(cand)
                continue
            out.append(cand)
            used.add(key)
        out.extend(deferred)
        return out

    # select_time_candidate_override: uses base implementation.

    def _preprocess_time_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._filter_times_to_query_window(candidates)

    def _score_time_candidate(self, item: Dict[str, Any], comp: str, reason: str) -> float:
        score = float(item.get("score", 0.0) or 0.0)
        c0 = self.canonicalize_component(component=str(item.get("component", "") or ""), component_level="mixed")
        r0 = self.parse_root_cause_label(str(item.get("reason", "") or ""))
        fam = _market_semantic_family(comp)
        src = str(item.get("source", "") or "").lower()
        if comp and c0:
            if c0 == comp:
                score += 0.45
            elif _market_semantic_family(c0) == fam:
                score += 0.22
            else:
                score -= 0.12
        if reason and r0:
            score += 0.28 if r0 == reason else -0.08
        if any(tok in src for tok in ["log", "event"]):
            score += 0.12
        if any(tok in src for tok in ["first", "onset", "breach"]):
            score += 0.08
        if any(tok in src for tok in ["trace_peak", "peak"]):
            score -= 0.04
        return score

    def postprocess_prediction_json_multi(
        self,
        *,
        prediction_json: str,
        selected_groups: Sequence[Dict[str, Any]],
        required_root_count: int,
        component_level: str,
    ) -> str:
        return str(prediction_json or "")

    def enable_trace_evidence(self) -> bool:
        return False

    def fallback_component_candidates_from_basil(self, evidence: Any) -> List[str]:
        scores = dict(getattr(evidence, "combined_scores", {}) or {})
        return [self.parse_component_id(c) for c, _s in sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)[:12]]

    def _reason_taxonomy(self) -> Dict[str, Dict[str, Any]]:
        data = self.config.get("reason_taxonomy", {})
        return data if isinstance(data, dict) else {}

    def _expected_level_for_reason(self, reason: str) -> str:
        taxonomy = self._reason_taxonomy()
        item = taxonomy.get(str(reason or ""))
        if isinstance(item, Mapping):
            lv = str(item.get("level", "") or "").strip().lower()
            return "pod" if lv == "container" else lv
        r = str(reason or "").lower()
        if r.startswith("node "):
            return "node"
        if r.startswith("container "):
            return "pod"
        return ""

    def _component_prior(self, component: str) -> float:
        c = str(component or "").lower()
        if not c:
            return 0.0
        if _is_trace_edge_component(c) or c in {"node", "source", "destination", "unknown"}:
            return 0.0
        if any(tok in c for tok in ["jaeger", "collector", "loadgenerator"]):
            return 0.45
        return 1.0

    def _filter_times_to_query_window(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        starts: List[int] = []
        ends: List[int] = []
        for item in candidates:
            ev = item.get("evidence")
            if not isinstance(ev, Mapping):
                continue
            s = _parse_local_time_s(str(ev.get("query_start_local", "") or ""))
            e = _parse_local_time_s(str(ev.get("query_end_local", "") or ""))
            if s:
                starts.append(s)
            if e:
                ends.append(e)
        if not starts or not ends:
            return candidates
        lo = min(starts) - 90
        hi = max(ends) + 90
        kept = []
        for item in candidates:
            ts = _parse_local_time_s(str(item.get("datetime", "") or ""))
            if not ts or lo <= ts <= hi:
                kept.append(item)
        return kept or candidates
