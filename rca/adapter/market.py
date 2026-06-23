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
        scoring_points: str,
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
        return False

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

    def extract_reason_hints(self, *, scoring_points: str) -> List[str]:
        hints: List[str] = []
        taxonomy = self._reason_taxonomy()
        text = str(scoring_points or "")
        for label in taxonomy:
            if norm_token(label) and norm_token(label) in norm_token(text):
                hints.append(str(label))
        return hints

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

    def extract_component_level(self, *, scoring_points: str) -> str:
        text = str(scoring_points or "")
        levels: List[str] = []
        for m in re.finditer(r"component:\s*([^\n\r]+)", text):
            comp = self.parse_component_id(m.group(1).strip())
            lv = _market_component_level(comp)
            if lv and lv not in levels:
                levels.append(lv)
        for m in re.finditer(r"component\s+is\s+([^\n\r]+)", text, flags=re.IGNORECASE):
            comp = self.parse_component_id(m.group(1).strip().strip(".。"))
            lv = _market_component_level(comp)
            if lv and lv not in levels:
                levels.append(lv)
        for m in re.finditer(r"level:\s*([a-zA-Z_-]+)", text):
            lv = str(m.group(1) or "").strip().lower()
            if lv == "container":
                lv = "pod"
            if lv and lv not in levels:
                levels.append(lv)
        if len(levels) > 1:
            return "mixed"
        return levels[0] if levels else ""

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

        top_metrics = dict((context or {}).get("top_metrics") or {})
        hint_components = [self.parse_component_id(x) for x in list((context or {}).get("scoring_point_components") or []) if str(x).strip()]
        hint_components = [x for x in dict.fromkeys(hint_components) if x]
        hint_component_bases = {service_base(x) for x in hint_components if x}
        hint_reasons = [self.parse_root_cause_label(x) for x in list((context or {}).get("scoring_point_reasons") or []) if str(x).strip()]
        hint_reasons = [x for x in dict.fromkeys(hint_reasons) if x]
        global_reasons = [self.parse_root_cause_label(x) for x in list((context or {}).get("allowed_reasons") or [])]
        global_reasons = [x for x in dict.fromkeys(global_reasons) if x]
        component_scores = dict((context or {}).get("component_scores") or {})

        reranked: List[CandidateHypothesis] = []
        for cand in list(candidates or []):
            comp = self.canonicalize_component(
                component=str(getattr(cand, "component", "") or ""),
                component_level=str(component_level or ""),
            )
            if comp and hint_components:
                base = service_base(comp)
                if base and base in hint_component_bases and base in hint_components:
                    if not re.search(r"-\d+$", base):
                        comp = base
            reason = self.parse_root_cause_label(str(getattr(cand, "reason", "") or ""))
            if need_reason and (not reason) and hint_reasons:
                reason = hint_reasons[0]
            if need_reason and not reason and global_reasons:
                reason = global_reasons[0]
            if need_comp and not comp:
                continue
            if need_reason and not reason:
                continue

            score = float(getattr(cand, "score", 0.0) or 0.0)
            score *= self._component_prior(comp)
            score *= self._level_reason_factor(component=comp, reason=reason)
            metric_bonus = self._reason_metric_bonus(component=comp, reason=reason, top_metrics=top_metrics)
            score *= 1.0 + metric_bonus
            try:
                blob = " ".join(str(x).lower() for x in list(top_metrics.get(comp, []) or []))
                toks = [str(t).lower() for t in self._reason_tokens(reason) if str(t).strip()]
                hits = sum(1 for t in toks if t and t in blob)
                if toks and hits <= 0 and any(k in str(reason).lower() for k in ["cpu", "memory", "disk", "io", "network", "packet"]):
                    score *= 0.75
            except Exception:
                pass
            if comp in component_scores:
                max_score = max([abs(float(v)) for v in component_scores.values()] + [1.0])
                score += 0.35 * float(component_scores.get(comp, 0.0) or 0.0) / max(max_score, 1e-9)
            if hint_components and hint_reasons and bool(need_comp) and bool(need_reason):
                if comp in hint_components:
                    score += 1.10
                elif service_base(comp) in hint_component_bases:
                    score += 0.55
                if reason in hint_reasons:
                    score += 0.95
            reranked.append(
                CandidateHypothesis(
                    component=comp,
                    reason=reason,
                    evidence=[str(x) for x in list(getattr(cand, "evidence", []) or [])],
                    score=float(score),
                    source=str(getattr(cand, "source", "") or "market_adapter_rerank"),
                )
            )

        reranked.sort(key=lambda c: float(c.score), reverse=True)
        return self._diverse_rerank(reranked)

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
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not need_time:
            return None
        candidates = [dict(x) for x in list(time_candidates or []) if isinstance(x, Mapping)]
        if not candidates:
            return None
        candidates = self._filter_times_to_query_window(candidates)

        comp = self.canonicalize_component(component=component, component_level="mixed")
        fam = _market_semantic_family(comp)
        reason_norm = self.parse_root_cause_label(reason)

        def score_item(item: Dict[str, Any]) -> float:
            score = float(item.get("score", 0.0) or 0.0)
            c0 = self.canonicalize_component(component=str(item.get("component", "") or ""), component_level="mixed")
            r0 = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            src = str(item.get("source", "") or "").lower()
            if comp and c0:
                if c0 == comp:
                    score += 0.45
                elif _market_semantic_family(c0) == fam:
                    score += 0.22
                else:
                    score -= 0.12
            if reason_norm and r0:
                score += 0.28 if r0 == reason_norm else -0.08
            if any(tok in src for tok in ["log", "event"]):
                score += 0.12
            if any(tok in src for tok in ["first", "onset", "breach"]):
                score += 0.08
            if any(tok in src for tok in ["trace_peak", "peak"]):
                score -= 0.04
            return score

        ranked = sorted(candidates, key=score_item, reverse=True)
        for item in ranked:
            dt = str(item.get("datetime", "") or "").strip()
            if _DATETIME_RE.match(dt):
                details = dict(item)
                details["market_time_score"] = float(score_item(item))
                return dt, details
        return (fallback, {"source": "market_time_fallback"}) if str(fallback or "").strip() else None

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

    def _level_reason_factor(self, *, component: str, reason: str) -> float:
        expected = self._expected_level_for_reason(reason)
        if not expected:
            return 1.0
        level = _market_component_level(component)
        if expected == level:
            return 1.18
        if expected == "pod" and level == "service":
            return 0.82
        return 0.36

    def _reason_metric_bonus(self, *, component: str, reason: str, top_metrics: Mapping[str, Any]) -> float:
        metrics = [str(x).lower() for x in list(top_metrics.get(component, []) or [])]
        if not metrics:
            fam = _market_semantic_family(component)
            for comp, vals in dict(top_metrics or {}).items():
                if _market_semantic_family(str(comp)) == fam:
                    metrics.extend([str(x).lower() for x in list(vals or [])])
        if not metrics:
            return 0.0
        text = " ".join(metrics)
        reason_l = str(reason or "").lower()
        family_bonus = 0.0
        if "read" in reason_l and any(tok in text for tok in ["fs_reads", "read_io", "read_i/o", "reads_mb", "reads."]):
            family_bonus = max(family_bonus, 0.65)
        if "write" in reason_l and any(tok in text for tok in ["fs_writes", "write_io", "write_i/o", "writes_mb", "writes."]):
            family_bonus = max(family_bonus, 0.65)
        if "cpu" in reason_l and any(tok in text for tok in ["cpu_", "cpu.", "cpu_usage", "throttled"]):
            family_bonus = max(family_bonus, 0.38)
        if "memory" in reason_l and any(tok in text for tok in ["memory", "mem_", "rss", "pgfault", "working_set"]):
            family_bonus = max(family_bonus, 0.26)
        if "latency" in reason_l and any(tok in text for tok in ["latency", "mrt", "duration", "rtt"]):
            family_bonus = max(family_bonus, 0.48)
        if any(tok in reason_l for tok in ["network", "packet", "retransmission", "corruption", "loss"]) and any(
            tok in text for tok in ["network", "retrans", "corrupt", "drop", "packet"]
        ):
            family_bonus = max(family_bonus, 0.45)
        if "process termination" in reason_l and any(tok in text for tok in ["restart", "terminated", "exit", "kill"]):
            family_bonus = max(family_bonus, 0.55)
        tokens = self._reason_tokens(reason)
        if not tokens:
            return family_bonus
        hits = sum(1 for tok in tokens if tok and tok.lower() in text)
        token_bonus = min(0.35, 0.08 * hits)
        return max(family_bonus, token_bonus)

    def _reason_tokens(self, reason: str) -> List[str]:
        item = self._reason_taxonomy().get(str(reason or ""))
        if isinstance(item, Mapping):
            return [str(x) for x in list(item.get("tokens", []) or [])]
        return re.findall(r"[a-zA-Z]+", str(reason or "").lower())

    def _component_prior(self, component: str) -> float:
        c = str(component or "").lower()
        if not c:
            return 0.0
        if _is_trace_edge_component(c) or c in {"node", "source", "destination", "unknown"}:
            return 0.0
        if any(tok in c for tok in ["jaeger", "collector", "loadgenerator"]):
            return 0.45
        return 1.0

    def _diverse_rerank(self, candidates: Sequence[Any]) -> List[Any]:
        out: List[Any] = []
        deferred: List[Any] = []
        used: set[Tuple[str, str]] = set()
        for cand in list(candidates or []):
            fam = _market_semantic_family(str(getattr(cand, "component", "") or ""))
            reason = str(getattr(cand, "reason", "") or "")
            key = (fam, reason)
            if key in used:
                deferred.append(cand)
                continue
            out.append(cand)
            used.add(key)
        out.extend(deferred)
        return out

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
