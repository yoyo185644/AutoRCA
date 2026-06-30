from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from rca.adapter.base import DatasetAdapter
from rca.execution.fast_solver import infer_reason_with_evidence
from rca.adapter.utils.time_candidates import (
    estimate_occurrence_datetime_candidates,
    select_time_candidate,
)


_TELECOM_LIGHT_AGENT_TASKS: Set[int] = {1, 2, 3, 4, 5, 6, 7}
_TELECOM_REASON_TAXONOMY: Tuple[str, ...] = (
    "CPU fault",
    "db close",
    "db connection limit",
    "network delay",
    "network loss",
)


class TelecomAdapter(DatasetAdapter):
    """Telecom adapter — wrapper that faithfully reproduces existing Telecom logic.

    Every method here is a direct, unchanged copy of the corresponding
    `if dataset == "Telecom"` block from field_solvers.py.
    No rewrite, no refactor — only relocation.
    """

    def basil_reason_taxonomy(self) -> List[str]:
        """Closed Telecom reason labels used by BASIL candidate expansion."""

        return list(_TELECOM_REASON_TAXONOMY)

    def enable_structured_time_candidates(self) -> bool:
        return True

    def coerce_closed_set_lists(self) -> bool:
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
        if int(task_id) in {2, 3, 4, 5, 6, 7}:
            return max(int(default_pool_k), 20)
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

        diversity = dict(out.get("candidate_diversity") or {})
        time_candidates = [x for x in list((context or {}).get("time_candidates") or []) if isinstance(x, Mapping)]
        time_clusters: set[str] = set()
        time_scores: List[float] = []
        for item in time_candidates[:12]:
            dt = str(item.get("datetime", "") or "")
            if dt:
                # Use the same coarse cluster as the shared gate. The LLM is
                # useful only when there are genuinely different time anchors.
                parts = dt.split()
                hm = parts[-1] if parts else dt
                toks = hm.split(":")
                if len(toks) >= 2:
                    try:
                        time_clusters.add(f"{int(toks[0]):02d}:{(int(toks[1]) // 5) * 5:02d}")
                    except Exception:
                        time_clusters.add(dt)
                else:
                    time_clusters.add(dt)
            time_scores.append(self._safe_float(item.get("score", 0.0)))

        time_gap = 1.0
        time_top_score = max(time_scores or [0.0])
        if len(time_scores) >= 2:
            ranked_time_scores = sorted(time_scores, reverse=True)
            time_gap = max(0.0, ranked_time_scores[0] - ranked_time_scores[1])
        if need_time and len(time_clusters) >= 2 and (time_gap < 0.12 or time_top_score < 0.55):
            out["need_llm"] = True
            reasons.append("telecom_critical_time_ambiguity")

        if bool(out.get("need_llm")):
            if need_comp and int(diversity.get("component_family", 0) or 0) <= 1:
                out["need_llm"] = False
                reasons.append("telecom_skip_llm_low_component_diversity")
            if need_reason and int(diversity.get("reason", 0) or 0) <= 1:
                out["need_llm"] = False
                reasons.append("telecom_skip_llm_low_reason_diversity")
            if need_time and len(time_clusters) <= 1:
                out["need_llm"] = False
                reasons.append("telecom_skip_llm_low_time_diversity")

        top1_score = self._safe_float(out.get("top1_score", 0.0))
        gap = self._safe_float(out.get("score_gap", 0.0))
        if bool(out.get("need_llm")) and top1_score >= 0.55 and gap >= 0.12 and "telecom_critical_time_ambiguity" not in reasons:
            out["need_llm"] = False
            reasons.append("telecom_skip_llm_confident_deterministic")

        # Soft evidence sufficiency penalty: when the pool lacks metric support,
        # let the LLM run only under a stricter adoption policy rather than
        # hard-dropping all difficult cases.
        top_metrics = dict((context or {}).get("top_metrics", {}) or {})
        has_metric_support = False
        for item in list(candidates or [])[:8]:
            if not isinstance(item, Mapping):
                continue
            comp = str(item.get("component", "") or "")
            if comp and list(top_metrics.get(comp, []) or []):
                has_metric_support = True
                break
            if list(item.get("supporting_metrics", []) or []) or list(item.get("evidence", []) or []):
                has_metric_support = True
                break
        if bool(out.get("need_llm")) and not has_metric_support:
            context["llm_change_min_confidence"] = max(self._safe_float(context.get("llm_change_min_confidence", 0.85)), 0.92)
            context["llm_max_score_drop_vs_det_top1"] = min(self._safe_float(context.get("llm_max_score_drop_vs_det_top1", 0.05)), 0.02)
            reasons.append("telecom_llm_evidence_sufficiency_soft_penalty")

        out["telecom_time_cluster_count"] = int(len(time_clusters))
        out["telecom_time_score_gap"] = float(time_gap)
        out["telecom_time_top_score"] = float(time_top_score)
        out["reasons"] = list(dict.fromkeys(reasons))
        return out

    def parse_root_cause_label(self, raw_reason: str) -> str:
        text = str(raw_reason or "").strip()
        norm = text.lower().replace("_", " ").replace("-", " ")
        norm = " ".join(norm.split())
        aliases = {
            "cpu": "CPU fault",
            "cpu fault": "CPU fault",
            "cpu load": "CPU fault",
            "high cpu": "CPU fault",
            "db close": "db close",
            "database close": "db close",
            "db connection": "db connection limit",
            "db connection limit": "db connection limit",
            "database connection limit": "db connection limit",
            "network delay": "network delay",
            "network latency": "network delay",
            "latency": "network delay",
            "network loss": "network loss",
            "packet loss": "network loss",
            "loss": "network loss",
            "memory": "high memory usage",
            "high memory": "high memory usage",
            "high memory usage": "high memory usage",
        }
        return aliases.get(norm, text)

    @staticmethod
    def _component_type(component: str) -> str:
        c = str(component or "")
        if c.startswith("db_"):
            return "db"
        if c.startswith(("os_", "node-")):
            return "node"
        if c.startswith(("docker_", "container")):
            return "container"
        if "-" in c:
            return "service"
        return "other"

    @staticmethod
    def _reason_type(reason: str) -> str:
        r = str(reason or "")
        if r in ("db close", "db connection limit"):
            return "db"
        if r == "CPU fault":
            return "container"
        if r in ("network delay", "network loss"):
            return "node"
        if r == "high memory usage":
            return "container"
        return ""

    @classmethod
    def _type_consistency_impl(cls, component: str, reason: str, top_metrics: Sequence[str] = ()) -> float:
        """Component–reason type alignment score (classmethod for internal reuse)."""
        rtype = cls._reason_type(reason)
        if not rtype:
            return 0.0
        ctype = cls._component_type(component)
        metric_blob = " ".join(str(x).lower() for x in top_metrics or [])
        if reason == "CPU fault":
            has_cpu_signal = any(tok in metric_blob for tok in ("cpu", "proc", "load", "iowait", "util"))
            if ctype == "container":
                return 0.36 if has_cpu_signal else 0.24
            if has_cpu_signal:
                return 0.06
            return -0.20
        if reason == "db connection limit" and any(tok in metric_blob for tok in ("conn", "login", "session", "tps", "sql", "query", "exec")):
            return 0.30
        if reason == "db close" and any(tok in metric_blob for tok in ("close", "abort", "disconnect")):
            return 0.25
        if reason in ("network delay", "network loss") and any(tok in metric_blob for tok in ("network", "recv", "send", "packet", "latency", "delay", "loss", "drop", "retrans")):
            return 0.22
        if reason == "high memory usage":
            if any(tok in metric_blob for tok in ("mem", "memory", "rss", "heap")):
                return 0.24
            if ctype in {"container", "node"}:
                return 0.18
            return -0.10
        return 0.35 if ctype == rtype else -0.35

    def _type_consistency(self, comp: str, reason: str, metrics: List[str]) -> float:
        """Instance-method hook required by base.rerank_candidate_hypotheses."""
        score = self._type_consistency_impl(comp, reason, metrics) if reason else 0.0
        if reason and reason not in _TELECOM_REASON_TAXONOMY:
            score -= 0.25
        return score

    @classmethod
    def _reason_compatible_types(cls, reason: str) -> Set[str]:
        reason = str(reason or "")
        if reason in ("db close", "db connection limit"):
            return {"db"}
        if reason == "CPU fault":
            return {"container"}
        if reason in ("network delay", "network loss"):
            return {"node"}
        if reason == "high memory usage":
            return {"container", "node"}
        return set()

    def component_semantic_type(self, component: str) -> str:
        return self._component_type(str(component or ""))

    def reason_semantic_type(self, reason: str) -> str:
        return self._reason_type(self.parse_root_cause_label(str(reason or "")))

    def reason_metric_keywords(self, reason: str) -> List[str]:
        reason = self.parse_root_cause_label(str(reason or ""))
        return {
            "CPU fault": ["cpu", "proc", "load", "iowait", "util"],
            "db close": ["close", "closed", "abort", "disconnect"],
            "db connection limit": ["conn", "connection", "login", "session", "tps", "sql", "query", "lock"],
            "network delay": ["delay", "latency", "rtt", "ping", "tnsping"],
            "network loss": ["loss", "drop", "retrans", "timeout", "reset", "packet"],
        }.get(reason, [])

    def reason_compatible_component_types(self, reason: str) -> List[str]:
        return list(self._reason_compatible_types(self.parse_root_cause_label(str(reason or ""))))

    def llm_reason_rules(self) -> List[str]:
        return [
            "CPU fault: prefer docker/container components only when metrics contain cpu/proc/load/iowait/util.",
            "db close or db connection limit: prefer db components only when metrics contain conn/session/query/sql/close/disconnect.",
            "network delay or network loss: prefer os/node components only when metrics contain packet/loss/drop/retrans/latency/delay.",
            "Do not switch to a common reason unless component type and metric evidence both support it.",
        ]

    def _postprocess_reranked(self, candidates: List[Any]) -> List[Any]:
        ordered = list(candidates or [])
        if len(ordered) < 2:
            return ordered
        top = ordered[0]
        top_comp = str(getattr(top, "component", "") or "")
        top_reason = self.parse_root_cause_label(str(getattr(top, "reason", "") or ""))
        if self._component_type(top_comp) != "db" or top_reason != "db connection limit":
            return ordered
        top_metrics = [str(x).lower() for x in list(getattr(top, "evidence", []) or [])]
        metric_blob = " ".join(top_metrics)
        if not any(tok in metric_blob for tok in ("lfsync", "dbtime", "db_time", "redo")):
            return ordered
        if any(tok in metric_blob for tok in ("proc_user", "proc user", "proc_user_used")):
            return ordered
        top_score = self._safe_float(getattr(top, "score", 0.0))
        for idx, cand in enumerate(ordered[1:8], start=1):
            comp = str(getattr(cand, "component", "") or "")
            reason = self.parse_root_cause_label(str(getattr(cand, "reason", "") or ""))
            if comp != top_comp or reason != "db close":
                continue
            cand_score = self._safe_float(getattr(cand, "score", 0.0))
            if cand_score >= top_score - 0.35:
                ordered.pop(idx)
                ordered.insert(0, cand)
                break
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
        if len(ordered) < 2:
            return ordered
        top = ordered[0]
        top_comp = str(top.get("component", "") or "")
        top_reason = self.parse_root_cause_label(str(top.get("reason", "") or ""))
        if self._component_type(top_comp) != "db" or top_reason != "db connection limit":
            return ordered
        metrics = [str(x).lower() for x in list(top.get("supporting_metrics", []) or [])]
        metric_blob = " ".join(metrics)
        if not any(tok in metric_blob for tok in ("lfsync", "dbtime", "db_time", "redo")):
            return ordered
        if any(tok in metric_blob for tok in ("proc_user", "proc user", "proc_user_used")):
            return ordered
        top_score = self._safe_float(top.get("score", 0.0))
        for idx, item in enumerate(ordered[1:8], start=1):
            comp = str(item.get("component", "") or "")
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            if comp != top_comp or reason != "db close":
                continue
            if self._safe_float(item.get("score", 0.0)) >= top_score - 0.35:
                ordered.pop(idx)
                ordered.insert(0, item)
                break
        return ordered

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _metric_reason_scores(metrics: Sequence[str]) -> Dict[str, float]:
        blob = " ".join(str(x or "").lower() for x in metrics or [])

        def hits(tokens: Sequence[str]) -> int:
            return sum(1 for tok in tokens if tok in blob)

        return {
            "CPU fault": float(hits(["cpu", "load", "iowait", "util", "proc"])),
            "db close": float(hits(["close", "closed", "abort", "disconnect"])),
            "db connection limit": float(hits(["conn", "connection", "login", "session", "tps", "sql", "query", "lock"])),
            "network delay": float(hits(["delay", "latency", "rtt", "ping", "tnsping"])),
            "network loss": float(hits(["loss", "drop", "retrans", "timeout", "reset", "packet"])),
        }

    @staticmethod
    def _surface_metric_reason_scores(metrics: Sequence[str]) -> Dict[str, float]:
        """Reason evidence from metric names without assuming a root component.

        This is deliberately separate from component/reason compatibility.  It
        serves reason-only and reason+time tasks, where the answer surface has
        no component and using component-conditioned scores can overfit to the
        currently top-ranked component.
        """

        scores = {r: 0.0 for r in _TELECOM_REASON_TAXONOMY}
        seen = {str(x or "").lower() for x in metrics or [] if str(x or "").strip()}
        for metric in seen:
            if any(tok in metric for tok in ("cpu", "proc", "util")):
                scores["CPU fault"] += 1.0
            if any(tok in metric for tok in ("load", "iowait")):
                scores["CPU fault"] += 0.45

            if any(tok in metric for tok in ("close", "closed", "abort", "disconnect")):
                scores["db close"] += 1.0
            if any(tok in metric for tok in ("conn", "connection", "login", "session", "sess", "tps", "sql", "query", "exec", "lock")):
                scores["db connection limit"] += 1.0
            if "dbtime" in metric or "db_time" in metric:
                scores["db connection limit"] += 0.45

            if any(tok in metric for tok in ("delay", "latency", "rtt", "ping", "tnsping")):
                scores["network delay"] += 1.0
            if "queue" in metric:
                scores["network delay"] += 0.45
                scores["network loss"] += 0.20
            if any(tok in metric for tok in ("loss", "drop", "retrans", "timeout", "reset", "packet")):
                scores["network loss"] += 1.0
            if any(tok in metric for tok in ("network", "recv", "send", "incoming", "outgoing", "traffic")):
                scores["network loss"] += 0.35
                scores["network delay"] += 0.20
        return scores

    def _surface_reason_scores(
        self,
        *,
        top_metrics: Mapping[str, Sequence[str]],
        comp_scores: Mapping[str, float],
    ) -> Dict[str, float]:
        ranked_all = [
            (str(c), self._safe_float(score))
            for c, score in sorted(
                dict(comp_scores or {}).items(),
                key=lambda kv: self._safe_float(kv[1]),
                reverse=True,
            )
        ]
        by_type: Dict[str, List[Tuple[str, float]]] = {}
        for comp, score in ranked_all:
            by_type.setdefault(self._component_type(comp), []).append((comp, score))

        selected: Dict[str, Tuple[float, int]] = {}
        for rank, (comp, score) in enumerate(ranked_all[:8], start=1):
            selected[comp] = (score, rank)
        rank_offset = 9
        for comps in by_type.values():
            for local_rank, (comp, score) in enumerate(comps[:4], start=0):
                selected.setdefault(comp, (score, rank_offset + local_rank))
            rank_offset += 4

        max_comp = max([abs(self._safe_float(v)) for _c, v in ranked_all] + [1.0])
        accum: Dict[str, float] = {r: 0.0 for r in _TELECOM_REASON_TAXONOMY}
        support: Dict[str, float] = {r: 0.0 for r in _TELECOM_REASON_TAXONOMY}
        for comp, (score, rank) in selected.items():
            metrics = [str(x) for x in list((top_metrics or {}).get(comp, []) or [])]
            if not metrics:
                continue
            local = self._surface_metric_reason_scores(metrics)
            comp_weight = max(0.10, self._safe_float(score) / max(max_comp, 1e-9))
            rank_weight = 1.0 / (1.0 + 0.12 * float(rank - 1))
            for reason, value in local.items():
                if value <= 0:
                    continue
                norm_value = min(1.0, float(value) / 3.0)
                accum[reason] += norm_value * comp_weight * rank_weight
                support[reason] = max(support[reason], norm_value)

        raw_scores = {
            reason: 0.60 * accum.get(reason, 0.0) + 0.40 * support.get(reason, 0.0)
            for reason in _TELECOM_REASON_TAXONOMY
        }
        max_score = max([abs(v) for v in raw_scores.values()] + [0.0])
        if max_score <= 1e-12:
            return {r: 0.0 for r in _TELECOM_REASON_TAXONOMY}
        return {r: float(max(0.0, v / max_score)) for r, v in raw_scores.items()}

    def _component_reason_metric_score(
        self,
        *,
        component: str,
        reason: str,
        metrics: Sequence[str],
    ) -> float:
        """Reason evidence for one component, calibrated by component type.

        This prevents a single generic CPU/DB token from dominating all
        reasons.  Evidence is strongest when metric tokens and component type
        both agree with the reason taxonomy.
        """
        r = self.parse_root_cause_label(str(reason or ""))
        if r not in _TELECOM_REASON_TAXONOMY:
            return 0.0
        ctype = self._component_type(str(component or ""))
        expected = self._reason_compatible_types(r)
        raw = self._metric_reason_scores(metrics)
        max_hits = {
            "CPU fault": 5.0,
            "db close": 4.0,
            "db connection limit": 8.0,
            "network delay": 5.0,
            "network loss": 6.0,
        }.get(r, 4.0)
        score = self._safe_float(raw.get(r, 0.0)) / max(max_hits, 1.0)

        if expected:
            if ctype in expected:
                score *= 1.25
            else:
                # Keep cross-type evidence possible, but weak: e.g. OS CPU
                # signals are often symptoms for container CPU faults.
                score *= 0.35

        # Reason-level competition inside close pairs.
        if r == "CPU fault":
            competitor = max(
                self._safe_float(raw.get("network delay", 0.0)),
                self._safe_float(raw.get("network loss", 0.0)),
                self._safe_float(raw.get("db connection limit", 0.0)),
                self._safe_float(raw.get("db close", 0.0)),
            )
            if competitor >= self._safe_float(raw.get("CPU fault", 0.0)):
                score *= 0.65
        elif r == "db close":
            if self._safe_float(raw.get("db close", 0.0)) <= 0 and self._safe_float(raw.get("db connection limit", 0.0)) > 0:
                score *= 0.35
        elif r == "db connection limit":
            if self._safe_float(raw.get("db connection limit", 0.0)) <= 0 and self._safe_float(raw.get("db close", 0.0)) > 0:
                score *= 0.45
        elif r == "network delay":
            if self._safe_float(raw.get("network delay", 0.0)) <= 0 and self._safe_float(raw.get("network loss", 0.0)) > 0:
                score *= 0.45
        elif r == "network loss":
            if self._safe_float(raw.get("network loss", 0.0)) <= 0 and self._safe_float(raw.get("network delay", 0.0)) > 0:
                score *= 0.45

        return float(max(0.0, min(1.0, score)))

    def _component_reason_rank_factor(
        self,
        *,
        component: str,
        reason: str,
        metrics: Sequence[str],
    ) -> float:
        """Safe score multiplier for an already-built component/reason tuple."""

        r = self.parse_root_cause_label(str(reason or ""))
        if not r:
            return 1.0
        if r not in _TELECOM_REASON_TAXONOMY:
            return 0.50
        ctype = self._component_type(str(component or ""))
        compat = self._reason_compatible_types(r)
        support = self._component_reason_metric_score(component=component, reason=r, metrics=metrics)
        consistency = self._type_consistency_impl(str(component or ""), r, metrics)
        metric_blob = " ".join(str(x).lower() for x in metrics or [])

        if r == "db connection limit":
            has_login_signal = any(tok in metric_blob for tok in ("login", "conn", "connection"))
            has_db_sync_signal = any(tok in metric_blob for tok in ("lfsync", "dbtime", "db_time", "redo"))
            if has_db_sync_signal and not has_login_signal:
                return 0.72
        if r == "db close":
            has_login_signal = any(tok in metric_blob for tok in ("login", "conn", "connection"))
            has_db_sync_signal = any(tok in metric_blob for tok in ("lfsync", "dbtime", "db_time", "redo"))
            if has_db_sync_signal and not has_login_signal:
                return 1.18

        if compat and ctype in compat:
            if support > 0:
                return 1.0 + min(0.18, 0.12 * support)
            return 0.92
        if support >= 0.25 or consistency > 0:
            return 0.82
        return 0.55

    def _latent_reason_component_bonus(
        self,
        *,
        component: str,
        metrics: Sequence[str],
        reason_scores: Mapping[str, float],
    ) -> Tuple[float, str]:
        """Best hidden reason support for tasks that do not output reason."""

        best_reason = ""
        best = 0.0
        max_reason = max([abs(self._safe_float(v)) for v in dict(reason_scores or {}).values()] + [1.0])
        for reason in _TELECOM_REASON_TAXONOMY:
            support = self._component_reason_metric_score(component=component, reason=reason, metrics=metrics)
            if support <= 0:
                continue
            prior = self._safe_float((reason_scores or {}).get(reason, 0.0)) / max(max_reason, 1e-9)
            compat = 1.0 if self._component_type(component) in self._reason_compatible_types(reason) else 0.45
            value = (0.70 * support + 0.30 * prior) * compat
            if value > best:
                best = float(value)
                best_reason = reason
        return float(min(1.0, best)), best_reason

    def _calibrated_reason_scores(
        self,
        *,
        top_metrics: Mapping[str, Sequence[str]],
        comp_scores: Mapping[str, float],
        tuple_reason_scores: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, float]:
        max_comp = max([abs(self._safe_float(v)) for v in dict(comp_scores or {}).values()] + [1.0])
        accum: Dict[str, float] = {r: 0.0 for r in _TELECOM_REASON_TAXONOMY}
        support: Dict[str, float] = {r: 0.0 for r in _TELECOM_REASON_TAXONOMY}

        ranked_all = [
            (str(c), self._safe_float(score))
            for c, score in sorted(
                dict(comp_scores or {}).items(),
                key=lambda kv: self._safe_float(kv[1]),
                reverse=True,
            )
        ]
        by_type: Dict[str, List[Tuple[str, float]]] = {}
        for comp, score in ranked_all:
            by_type.setdefault(self._component_type(comp), []).append((comp, score))

        # Reason evidence should not be dominated by whichever component type
        # has the largest raw anomaly scale.  Build a type-balanced evidence
        # pool: global top components plus top representatives from every
        # semantic channel.  This is the same principle as typed-channel
        # candidate recall, applied to reason scoring.
        selected: Dict[str, Tuple[float, int]] = {}
        for rank, (comp, score) in enumerate(ranked_all[:8], start=1):
            selected[comp] = (score, rank)
        rank_offset = 9
        for comps in by_type.values():
            for local_rank, (comp, score) in enumerate(comps[:5], start=0):
                selected.setdefault(comp, (score, rank_offset + local_rank))
            rank_offset += 5

        for comp, (comp_score, rank) in selected.items():
            metrics = [str(x) for x in list((top_metrics or {}).get(comp, []) or [])]
            if not metrics:
                continue
            comp_weight = max(0.05, self._safe_float(comp_score) / max(max_comp, 1e-9))
            rank_weight = 1.0 / (1.0 + 0.15 * float(rank - 1))
            for reason in _TELECOM_REASON_TAXONOMY:
                s = self._component_reason_metric_score(component=comp, reason=reason, metrics=metrics)
                if s <= 0:
                    continue
                expected = self._reason_compatible_types(reason)
                if expected and self._component_type(comp) in expected:
                    # Strong local evidence in the expected semantic channel
                    # should survive even if that component is not globally top.
                    s = min(1.0, s * 1.15)
                    rank_weight = max(rank_weight, 0.55)
                value = s * comp_weight * rank_weight
                accum[reason] += value
                support[reason] = max(support[reason], s)

        tuple_scores = dict(tuple_reason_scores or {})
        if tuple_scores:
            max_tuple = max([abs(self._safe_float(v)) for v in tuple_scores.values()] + [1.0])
            for raw_reason, value in tuple_scores.items():
                reason = self.parse_root_cause_label(str(raw_reason or ""))
                if reason in accum:
                    accum[reason] += 0.25 * self._safe_float(value) / max(max_tuple, 1e-9)

        # Blend aggregate support and best local support. Aggregate captures
        # multi-source consistency; best support preserves sharp local evidence.
        raw_scores = {
            reason: 0.70 * accum.get(reason, 0.0) + 0.30 * support.get(reason, 0.0)
            for reason in _TELECOM_REASON_TAXONOMY
        }
        # Competitive calibration among broad reason families.  A family wins
        # only when its compatible channel has evidence, not merely because its
        # tokens appear in a high-scoring symptom component.
        family_best = {
            "container": raw_scores.get("CPU fault", 0.0),
            "db": max(raw_scores.get("db close", 0.0), raw_scores.get("db connection limit", 0.0)),
            "node": max(raw_scores.get("network delay", 0.0), raw_scores.get("network loss", 0.0)),
        }
        for reason in list(raw_scores.keys()):
            rtype = self._reason_type(reason)
            if not rtype:
                continue
            own = family_best.get(rtype, 0.0)
            competing = max([v for k, v in family_best.items() if k != rtype] + [0.0])
            if competing > 0 and own < 0.75 * competing:
                raw_scores[reason] *= 0.70
        max_score = max([abs(v) for v in raw_scores.values()] + [0.0])
        if max_score <= 1e-12:
            return {r: 0.0 for r in _TELECOM_REASON_TAXONOMY}
        return {r: float(max(0.0, v / max_score)) for r, v in raw_scores.items()}

    def _allow_light_agent(self, *, spec: Any, need_comp: bool) -> bool:
        task_id = int(getattr(spec, "task_id", 0) or 0)
        difficulty = str(getattr(spec, "difficulty", "") or "")
        return (task_id in _TELECOM_LIGHT_AGENT_TASKS) or (
            difficulty != "easy" and bool(need_comp)
        )

    def get_component_delta_s(
        self,
        *,
        spec: Any,
        need_comp: bool,
        need_time: bool,
        center_time_s: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[int, Optional[int]]:
        difficulty = str(getattr(spec, "difficulty", "") or "")
        if difficulty != "easy":
            return 1800, None
        if need_time:
            return 1800, center_time_s
        return 600, center_time_s

    def get_time_dt_override(
        self,
        *,
        need_time: bool,
        comp: str,
        ev3: Any,
        **kwargs: Any,
    ) -> Optional[str]:
        if not need_time:
            return None
        ft = {}
        pt = {}
        try:
            ft = getattr(ev3, "first_times", None) or {}
            pt = getattr(ev3, "peak_times", None) or {}
        except Exception:
            pass
        dt2 = ft.get(comp) or pt.get(comp) or str(getattr(ev3, "estimated_failure_time", "") or "")
        if str(dt2 or "").strip():
            return str(dt2)
        return None

    def solve_reason(
        self,
        *,
        spec: Any,
        instruction: str,
        args: Any,
        need_reason: bool,
        **kwargs: Any,
    ) -> Optional[Tuple[str, str, bool, Any]]:
        if not need_reason:
            return None
        det_reason, ev2 = infer_reason_with_evidence(instruction)
        # LLM call removed: field_solvers.solve_reason owns the unified LLM rerank step.
        return str(det_reason or ""), "", True, ev2

    def solve_component_joint_scoring(
        self,
        *,
        ev2: Any,
        ev3: Any,
        det_reason: str,
        det_reason_ready: bool,
        **kwargs: Any,
    ) -> Optional[Tuple[str, str]]:
        if not (str(self.dataset) == "Telecom" and det_reason_ready):
            return None
        rs = (ev2 or {}).get("scores", {}) if isinstance(ev2, dict) else {}
        cs = getattr(ev3, "combined_scores", None) or {}
        if not rs or not cs:
            return None

        def _ctype(c: str) -> str:
            c = str(c or "")
            if c.startswith("db_"):
                return "db"
            if c.startswith(("os_", "node-")):
                return "node"
            if c.startswith(("docker_", "container")):
                return "container"
            if "-" in c:
                return "service"
            return "other"

        def _rtype(r: str) -> str:
            r = str(r or "")
            if r in ("db close", "db connection limit"):
                return "db"
            if r == "CPU fault":
                return "container"
            if r in ("network delay", "network loss"):
                return "node"
            return ""

        max_r = max([float(v) for v in rs.values()] + [0.0])
        max_c = max([abs(float(v)) for v in cs.values()] + [0.0])
        best_pair = None
        best_joint = -1.0
        for r, rv in rs.items():
            rn = float(rv) / max(max_r, 1e-9)
            desired_t = _rtype(r)
            pool = (
                [c for c in cs.keys() if desired_t and _ctype(c) == desired_t]
                if desired_t
                else list(cs.keys())
            )
            if not pool:
                pool = list(cs.keys())
            c_best = max(pool, key=lambda c: float(cs.get(c, 0.0)))
            cn = float(cs.get(c_best, 0.0)) / max(max_c, 1e-9)
            joint = 0.55 * cn + 0.45 * rn
            if joint > best_joint:
                best_joint = joint
                best_pair = (str(r), str(c_best))
        if best_pair is not None:
            return best_pair
        return None

    def solve_component_desired_pool(
        self,
        *,
        comp: str,
        det_reason_out: str,
        ev3: Any,
        **kwargs: Any,
    ) -> str:
        def _ctype(c: str) -> str:
            c = str(c or "")
            if c.startswith("db_"):
                return "db"
            if c.startswith(("os_", "node-")):
                return "node"
            if c.startswith(("docker_", "container")):
                return "container"
            if "-" in c:
                return "service"
            return "other"

        top1_score = float((ev3.combined_scores or {}).get(comp, 0.0))
        desired = ""
        desired_ratio = 0.0
        if det_reason_out in ("db close", "db connection limit"):
            desired = "db"
            # After per-file normalisation in the extractor, db and node scores are
            # on the same scale.  Lower threshold so a db component at 30 % of the
            # top-1 node score still wins over the node.
            desired_ratio = 0.30
        elif det_reason_out == "CPU fault":
            desired = "container"
            desired_ratio = 0.30
        elif det_reason_out in ("network delay", "network loss"):
            desired = "node"
            desired_ratio = 0.30
        if desired and top1_score > 1e-12:
            pool = [c for c in (ev3.combined_scores or {}).keys() if _ctype(c) == desired]
            if pool:
                best_c = max(pool, key=lambda c: float((ev3.combined_scores or {}).get(c, 0.0)))
                best_s = float((ev3.combined_scores or {}).get(best_c, 0.0))
                if best_s >= desired_ratio * top1_score:
                    return best_c
        return comp

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
        if not need_reason:
            return list(reasons or []), {}, {}

        evidence_scores: Dict[str, float] = {}

        # Reuse evidence already produced by the BASIL index. Avoid re-reading
        # raw trace files here; this hook runs for every indexed query.
        tuples = [dict(x) for x in list(getattr(result, "tuples", []) or []) if isinstance(x, Mapping)]
        for item in tuples:
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            if reason in _TELECOM_REASON_TAXONOMY:
                evidence_scores[reason] = max(
                    evidence_scores.get(reason, 0.0),
                    self._safe_float(item.get("score")),
                )

        comp_ev = getattr(result, "component_evidence", None)
        top_metrics = dict(getattr(comp_ev, "top_metrics", {}) or {})
        comp_scores = dict(getattr(comp_ev, "combined_scores", {}) or {})
        if not top_metrics:
            top_metrics = dict(getattr(result, "top_metrics", {}) or {})
        if not comp_scores:
            comp_scores = dict(getattr(result, "combined_scores", {}) or {})
        metric_names: List[str] = []
        ranked_components = [
            str(c)
            for c, _score in sorted(
                comp_scores.items(),
                key=lambda kv: self._safe_float(kv[1]),
                reverse=True,
            )[:12]
        ]
        for comp in ranked_components:
            metric_names.extend([str(x) for x in list(top_metrics.get(comp, []) or [])])

        calibrated = self._calibrated_reason_scores(
            top_metrics=top_metrics,
            comp_scores=comp_scores,
            tuple_reason_scores=evidence_scores,
        )
        surface_scores = self._surface_reason_scores(
            top_metrics=top_metrics,
            comp_scores=comp_scores,
        )
        for reason, score in calibrated.items():
            evidence_scores[reason] = max(self._safe_float(evidence_scores.get(reason, 0.0)), self._safe_float(score))

        # Keep the closed taxonomy available, but let metric evidence decide order.
        for raw in list(reasons or []) + list(_TELECOM_REASON_TAXONOMY):
            reason = self.parse_root_cause_label(str(raw))
            if reason in _TELECOM_REASON_TAXONOMY:
                evidence_scores.setdefault(reason, 0.0)

        max_score = max([abs(v) for v in evidence_scores.values()] + [0.0])
        if max_score > 1e-12:
            norm_scores = {k: max(0.0, float(v) / max_score) for k, v in evidence_scores.items()}
        else:
            norm_scores = {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(_TELECOM_REASON_TAXONOMY)}

        ordered = [
            r
            for r, _ in sorted(
                norm_scores.items(),
                key=lambda kv: (float(kv[1]), -list(_TELECOM_REASON_TAXONOMY).index(kv[0]) if kv[0] in _TELECOM_REASON_TAXONOMY else -999),
                reverse=True,
            )
        ]
        return ordered, norm_scores, {
            "telecom_reason_scores": norm_scores,
            "telecom_reason_surface_scores": surface_scores,
            "telecom_reason_metrics": metric_names[:30],
        }

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
        if not need_time:
            return [dict(x) for x in list(time_candidates or []) if isinstance(x, Mapping)]

        reasons = list((context_overrides or {}).get("telecom_reason_scores", {}).keys())
        estimated = ""
        try:
            times = list(getattr(result, "times", []) or [])
            estimated = str(times[0]) if times else ""
        except Exception:
            estimated = ""

        merged: List[Dict[str, Any]] = [dict(x) for x in list(time_candidates or []) if isinstance(x, Mapping)]
        try:
            generated = estimate_occurrence_datetime_candidates(
                dataset=str(self.dataset),
                instruction=str(instruction or ""),
                component_first_times=dict(component_first_times or {}),
                component_peak_times=dict(component_peak_times or {}),
                estimated_failure_time=str(estimated or ""),
                component_scores=dict(component_scores or {}),
                reasons=[str(r) for r in reasons],
                top_k=max(12, int(getattr(args, "time_candidate_top_k", 8) or 8)),
            )
            merged.extend([cand.to_dict() for cand in generated])
        except Exception:
            pass

        best_by_key: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for item in merged:
            dt = str(item.get("datetime", "") or "").strip()
            if not dt:
                continue
            key = (
                dt,
                str(item.get("component", "") or ""),
                str(item.get("reason", "") or ""),
                str(item.get("source", "") or ""),
            )
            old = best_by_key.get(key)
            if old is None or self._safe_float(item.get("score")) > self._safe_float(old.get("score")):
                best_by_key[key] = dict(item)

        out = sorted(best_by_key.values(), key=lambda x: self._safe_float(x.get("score")), reverse=True)
        return out[: max(12, int(getattr(args, "time_candidate_top_k", 8) or 8))]

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
        try:
            dt, details = select_time_candidate(
                candidates=list(time_candidates or []),
                component=str(component or ""),
                reason=self.parse_root_cause_label(str(reason or "")),
                fallback=str(fallback or ""),
            )
            if str(dt or "").strip():
                return str(dt), dict(details or {})
        except Exception:
            return None
        return None

    def expand_index_candidate_tuples(
        self,
        *,
        selected: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        task_id: int,
        k: int,
    ) -> List[Dict[str, Any]]:
        need_time = bool((context or {}).get("need_time", False))
        need_comp = bool((context or {}).get("need_comp", False))
        need_reason = bool((context or {}).get("need_reason", False))
        if not need_comp:
            if need_reason:
                reason_scores = dict((context or {}).get("reason_scores", {}) or {})
                surface_scores = dict((context or {}).get("telecom_reason_surface_scores", {}) or {})
                default_dt = str((context or {}).get("time_choice", "") or "")
                if need_time and not default_dt:
                    for item in list((context or {}).get("time_candidates", []) or []):
                        if not isinstance(item, Mapping):
                            continue
                        dt0 = str(item.get("datetime", "") or "").strip()
                        if dt0:
                            default_dt = dt0
                            break
                latent_time_type = ""
                if need_time:
                    for item in list((context or {}).get("time_candidates", []) or []):
                        if not isinstance(item, Mapping):
                            continue
                        dt0 = str(item.get("datetime", "") or "").strip()
                        if default_dt and dt0 != default_dt:
                            continue
                        comp0 = str(item.get("component", "") or "").strip()
                        if not comp0:
                            continue
                        latent_time_type = self._component_type(comp0)
                        if latent_time_type and score < 0.90:
                            break
                hypothesis_scores: Dict[str, float] = {}
                for rank, item in enumerate([dict(x) for x in list(selected or []) if isinstance(x, Mapping)], start=1):
                    reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
                    if reason not in _TELECOM_REASON_TAXONOMY:
                        continue
                    hypothesis_scores[reason] = hypothesis_scores.get(reason, 0.0) + self._safe_float(item.get("score")) / float(rank)
                if reason_scores or hypothesis_scores:
                    max_reason = max([abs(self._safe_float(v)) for v in reason_scores.values()] + [1.0])
                    max_hyp = max([abs(self._safe_float(v)) for v in hypothesis_scores.values()] + [1.0])
                    max_surface = max([abs(self._safe_float(v)) for v in surface_scores.values()] + [1.0])
                    all_reasons = list(dict.fromkeys(list(hypothesis_scores.keys()) + list(reason_scores.keys()) + list(surface_scores.keys()) + list(_TELECOM_REASON_TAXONOMY)))
                    out: List[Dict[str, Any]] = []
                    for reason in all_reasons:
                        r = self.parse_root_cause_label(str(reason))
                        if r not in _TELECOM_REASON_TAXONOMY:
                            continue
                        hyp_norm = self._safe_float(hypothesis_scores.get(r, 0.0)) / max(max_hyp, 1e-9)
                        ev_norm = self._safe_float(reason_scores.get(r, 0.0)) / max(max_reason, 1e-9)
                        surface_norm = self._safe_float(surface_scores.get(r, 0.0)) / max(max_surface, 1e-9)
                        # For reason-only / reason-time tasks, component
                        # tuples are latent support, not the answer surface.
                        # Combine component-calibrated evidence with a
                        # component-free metric surface vote so the answer is
                        # not forced by the currently top-ranked component.
                        score = 0.10 * hyp_norm + 0.45 * ev_norm + 0.45 * surface_norm
                        if latent_time_type:
                            compat = self._reason_compatible_types(r)
                            if compat:
                                if latent_time_type in compat:
                                    score += 0.65
                                else:
                                    score *= 0.55
                        out.append(
                            {
                                "component": "",
                                "reason": r,
                                "datetime": default_dt if need_time else "",
                                "score": float(score),
                                "source": "telecom_reason_evidence",
                                "evidence": [
                                    f"reason_score={ev_norm:.3f}",
                                    f"surface_score={surface_norm:.3f}",
                                    f"hypothesis_score={hyp_norm:.3f}",
                                ],
                            }
                        )
                    if out:
                        return sorted(out, key=lambda x: self._safe_float(x.get("score")), reverse=True)
            return [dict(x) for x in list(selected or []) if isinstance(x, Mapping)]

        base_items = [dict(x) for x in list(selected or []) if isinstance(x, Mapping)]
        comp_scores = dict((context or {}).get("component_scores", {}) or {})
        reason_scores = dict((context or {}).get("reason_scores", {}) or {})
        top_metrics = dict((context or {}).get("top_metrics", {}) or {})
        first_times = dict((context or {}).get("component_first_times", {}) or {})

        if need_comp and not need_reason and not need_time:
            out: List[Dict[str, Any]] = []
            seen: Set[str] = set()
            for item in base_items:
                comp = str(item.get("component", "") or "").strip()
                if not comp or comp in seen:
                    continue
                seen.add(comp)
                candidate = dict(item)
                candidate["datetime"] = str(first_times.get(comp, "") or candidate.get("datetime", "") or "")
                candidate["first_anomaly_time"] = str(first_times.get(comp, "") or candidate.get("first_anomaly_time", "") or "")
                candidate["component_type"] = self._component_type(comp)
                candidate["symptomness"] = self._safe_float(candidate.get("symptomness", 0.0))
                if not candidate.get("supporting_metrics"):
                    candidate["supporting_metrics"] = [str(x) for x in list(top_metrics.get(comp, []) or [])]
                out.append(candidate)
            if out:
                return out

        ranked_components = [
            str(c)
            for c, _score in sorted(
                comp_scores.items(),
                key=lambda kv: self._safe_float(kv[1]),
                reverse=True,
            )
            if str(c or "").strip()
        ]
        for item in base_items:
            comp = str(item.get("component", "") or "").strip()
            if comp and comp not in ranked_components:
                ranked_components.append(comp)

        raw_reasons: List[str] = []
        if need_reason:
            for reason, _score in sorted(
                reason_scores.items(),
                key=lambda kv: self._safe_float(kv[1]),
                reverse=True,
            ):
                r = self.parse_root_cause_label(str(reason))
                if r in _TELECOM_REASON_TAXONOMY and r not in raw_reasons:
                    raw_reasons.append(r)
            for reason in list((context or {}).get("allowed_reasons", []) or []):
                r = self.parse_root_cause_label(str(reason))
                if r in _TELECOM_REASON_TAXONOMY and r not in raw_reasons:
                    raw_reasons.append(r)
            if not raw_reasons:
                raw_reasons = list(_TELECOM_REASON_TAXONOMY)
        else:
            raw_reasons = [""]

        max_comp = max([abs(self._safe_float(v)) for v in comp_scores.values()] + [1.0])
        max_reason = max([abs(self._safe_float(v)) for v in reason_scores.values()] + [1.0])
        max_existing = max([self._safe_float(item.get("score")) for item in base_items] + [1.0])
        component_budget = max(24, max(1, int(k)) * 12, len(base_items) * 6)
        reason_budget = max(3, min(len(raw_reasons), max(3, int(k) * 3)))

        merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        def add_item(item: Dict[str, Any]) -> None:
            comp = str(item.get("component", "") or "").strip()
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            dt = str(item.get("datetime", "") or "").strip()
            if need_comp and not comp:
                return
            if need_reason and not reason:
                return
            if reason and reason not in _TELECOM_REASON_TAXONOMY:
                return
            key = (comp, reason, dt)
            candidate = dict(item)
            candidate["reason"] = reason
            if comp and reason:
                metrics = [str(x) for x in list(candidate.get("supporting_metrics", []) or top_metrics.get(comp, []) or [])]
                candidate["score"] = self._safe_float(candidate.get("score", 0.0)) * self._component_reason_rank_factor(
                    component=comp,
                    reason=reason,
                    metrics=metrics,
                )
            old = merged.get(key)
            if old is None or self._safe_float(candidate.get("score")) > self._safe_float(old.get("score")):
                merged[key] = candidate

        for item in base_items:
            add_item(item)

        for rank, comp in enumerate(ranked_components[:component_budget], start=1):
            comp = str(comp or "").strip()
            if not comp:
                continue
            comp_norm = self._safe_float(comp_scores.get(comp, 0.0)) / max(max_comp, 1e-9)
            rank_prior = 1.0 / float(rank)
            metrics = [str(x) for x in list(top_metrics.get(comp, []) or [])]
            latent_bonus, latent_reason = self._latent_reason_component_bonus(
                component=comp,
                metrics=metrics,
                reason_scores=reason_scores,
            )

            for reason in raw_reasons[:reason_budget]:
                compat = self._reason_compatible_types(reason)
                ctype = self._component_type(comp)
                component_reason_support = self._component_reason_metric_score(
                    component=comp,
                    reason=reason,
                    metrics=metrics,
                )
                if compat and ctype not in compat:
                    # Keep strong metric-backed alternatives, but do not let
                    # generic symptom components flood the tuple pool.
                    if component_reason_support <= 0.05:
                        continue
                reason_norm = self._safe_float(reason_scores.get(reason, 0.0)) / max(max_reason, 1e-9)
                consistency = self._type_consistency_impl(comp, reason, metrics)
                score = (
                    0.45 * comp_norm
                    + (0.25 * reason_norm if need_reason else 0.0)
                    + 0.15 * rank_prior
                    + max(0.0, consistency)
                    + 0.25 * component_reason_support
                    + (0.22 * latent_bonus if need_comp and need_time and not need_reason else 0.0)
                )
                if compat and ctype in compat:
                    score += 0.12
                if score <= 0:
                    continue
                add_item(
                    {
                        "component": comp,
                        "reason": reason if (need_reason or reason) else "",
                        "datetime": str(first_times.get(comp, "") or "") if need_time else "",
                        "score": float(min(max_existing * 1.05, score)),
                        "source": "telecom_reason_conditioned_expand",
                        "supporting_metrics": metrics[:6],
                        "latent_reason": latent_reason,
                    }
                )

        return sorted(merged.values(), key=lambda x: self._safe_float(x.get("score")), reverse=True)

    # rerank_candidate_hypotheses: uses base implementation.
    # Telecom-specific behaviour is injected via _type_consistency() override above.
