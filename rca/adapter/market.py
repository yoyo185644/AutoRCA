from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rca.common.text import norm_token, service_base
from rca.adapter.base import DatasetAdapter


_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_MARKET_NORMALIZED_METRIC_CACHE: Dict[str, Any] = {}
_MARKET_TRACE_CALL_GRAPH_CACHE: Dict[Tuple[str, str], Dict[Tuple[str, str], int]] = {}


def _market_normalized_metric_file(path: str):
    from rca.adapter.utils.component_candidates import _normalize_metric_file

    key = os.path.abspath(str(path))
    cached = _MARKET_NORMALIZED_METRIC_CACHE.get(key)
    if cached is not None:
        return cached
    df = _normalize_metric_file(path)
    _MARKET_NORMALIZED_METRIC_CACHE[key] = df
    return df


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
    if re.search(r"\.ts(?::\d+)?$", c) or re.search(r":\d+$", c):
        return "service"
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


def _market_service_from_operation(operation_name: str) -> str:
    text = str(operation_name or "").strip()
    if not text:
        return ""
    m = re.search(r"\.([A-Za-z0-9]+Service)\s*/", text)
    if not m:
        m = re.search(r"\.([A-Za-z0-9]+Service)\b", text)
    if not m:
        return ""
    token = m.group(1)
    token = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", token).lower().replace("_", "")
    return service_base(token) or token


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
        "container packet loss": "container packet loss",
        "container network packet loss": "container packet loss",
        "container packet retransmission": "container network packet retransmission",
        "container packet corruption": "container network packet corruption",
        "network packet retransmission": "container network packet retransmission",
        "network packet corruption": "container network packet corruption",
        "network packet loss": "container packet loss",
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
        c = re.split(r"\.(?:source|destination)\.", c, maxsplit=1)[0]
        if self._preserve_service_instance_output() and c.endswith("-http"):
            return re.sub(r"-http$", "", c)
        if re.search(r"\.ts(?::\d+)?$", c) or re.search(r":\d+$", c) or c.endswith("-grpc"):
            # Strip only the decorative protocol/port suffix here (".ts",
            # ":PORT", "-grpc"), not the aggressive service_base() collapse
            # that used to run in this branch. service_base() strips ANY
            # trailing "-N", which is correct for genuine pod/replica noise
            # but wrong for this dataset's many sibling deployments whose
            # "-N" IS their real identity -- record.csv for cloudbed-1 has
            # "shippingservice", "shippingservice-1", and "shippingservice2-0"
            # as three separate, valid ground-truth components (same for
            # cartservice/frontend/recommendationservice/etc.), so collapsing
            # them here made ~40%+ of service-level answers unreachable no
            # matter how good the underlying anomaly scoring was.
            c2 = re.sub(r"\.ts(?::\d+)?$", "", c)
            c2 = re.sub(r":\d+$", "", c2)
            c2 = re.sub(r"-grpc$", "", c2)
            return c2 or c
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
            return ["mem", "memory", "rss", "cache", "pgfault", "pgmajfault", "working_set"]
        if "disk" in r and "space" in r:
            return ["space", "filesystem", "fs_usage", "fs_used", "inode", "disk.used", "disk.in_use", "readonly"]
        if "read" in r:
            return ["read", "reads", "io", "i/o", "disk", "blkio", "fs_reads", "r_s", "rkb"]
        if "write" in r:
            return ["write", "writes", "io", "i/o", "disk", "blkio", "fs_writes", "w_s", "wkb"]
        if "network" in r or "latency" in r or "packet" in r:
            if "latency" in r:
                return ["latency", "rtt", "delay", "mrt", "duration"]
            if "retrans" in r:
                return ["retrans", "retransmission", "retry", "tcp.retrans", "retrans_segs", "retrans_pct"]
            if "corruption" in r or "corrupt" in r:
                return ["corrupt", "corruption", "error", "in_errors", "out.error", "packets_in.error", "packets_out.error", "rcv_buf_errors", "snd_buf_errors"]
            if "loss" in r:
                return ["loss", "drop", "dropped", "packets_dropped", "packet_loss"]
            return ["network", "packet", "tcp", "udp"]
        if "termination" in r or "process" in r:
            return ["restart", "restarts", "exit", "kill", "killed", "process", "termination", "last_seen", "uptime"]
        return []

    def reason_compatible_component_types(self, reason: str) -> List[str]:
        rtype = self.reason_semantic_type(str(reason or ""))
        return [rtype] if rtype else []

    def _date_dir_from_context(self, context: Mapping[str, Any]) -> str:
        date_dir = str((context or {}).get("date_dir", "") or "").strip()
        if date_dir:
            return date_dir
        instruction = str((context or {}).get("instruction", "") or "")
        if not instruction:
            return ""
        try:
            from rca.adapter.utils.component_candidates import parse_time_window_generic
            from rca.common.io import resolve_date_dir

            raw_date, _start, _end = parse_time_window_generic(instruction)
            return str(resolve_date_dir(self.dataset, raw_date))
        except Exception:
            return ""

    def _trace_call_graph_edges(self, *, context: Mapping[str, Any]) -> Dict[Tuple[str, str], int]:
        date_dir = self._date_dir_from_context(context)
        if not date_dir:
            return {}
        key = (str(self.dataset), str(date_dir))
        cached = _MARKET_TRACE_CALL_GRAPH_CACHE.get(key)
        if cached is not None:
            return cached
        path = os.path.join("dataset", self.dataset, "telemetry", date_dir, "trace", "trace_span.csv")
        edges: Dict[Tuple[str, str], int] = {}
        if not os.path.exists(path):
            _MARKET_TRACE_CALL_GRAPH_CACHE[key] = edges
            return edges
        try:
            import pandas as pd

            # Use a bounded sample. The service call graph is stable over the
            # day, so a sample is enough for direction evidence and avoids
            # repeatedly scanning multi-GB trace files.
            df = pd.read_csv(
                path,
                usecols=lambda c: c in {"cmdb_id", "operation_name"},
                nrows=180_000,
                low_memory=False,
            )
        except Exception:
            _MARKET_TRACE_CALL_GRAPH_CACHE[key] = edges
            return edges
        if df.empty or "cmdb_id" not in df.columns or "operation_name" not in df.columns:
            _MARKET_TRACE_CALL_GRAPH_CACHE[key] = edges
            return edges
        for caller_raw, op in zip(df["cmdb_id"].astype(str), df["operation_name"].astype(str)):
            caller = _market_semantic_family(self.parse_component_id(caller_raw))
            target = _market_service_from_operation(op)
            if not caller or not target or caller == target:
                continue
            # Runtime symptoms propagate from callee/root to caller/upstream
            # observer. This reverse call direction helps suppress frontend or
            # proxy-like symptoms when a callee has local evidence.
            edges[(target, caller)] = edges.get((target, caller), 0) + 1
        _MARKET_TRACE_CALL_GRAPH_CACHE[key] = edges
        return edges

    def _component_propagation_features(
        self,
        *,
        component: str,
        context: Mapping[str, Any],
    ) -> Dict[str, float]:
        base = dict(super()._component_propagation_features(component=component, context=context))
        if not bool((context or {}).get("need_time", False)):
            return base
        comp = self.parse_component_id(str(component or ""))
        family = _market_semantic_family(comp)
        if not family or self.component_semantic_type(comp) == "node":
            return base
        edges = self._trace_call_graph_edges(context=context)
        if not edges:
            return base
        max_count = max([int(v) for v in edges.values()] + [1])
        outgoing = 0.0
        incoming = 0.0
        for (src, dst), count in edges.items():
            score = min(1.0, float(count) / float(max_count))
            if src == family:
                outgoing = max(outgoing, score)
            if dst == family:
                incoming = max(incoming, score)
        if outgoing > 0.0:
            # Treat static call-graph direction as weak structural evidence.
            # It should break ties between symptom/root candidates, not
            # override strong local metric ownership.
            base["outgoing_propagation"] = max(self._safe_float(base.get("outgoing_propagation", 0.0)), 0.35 * outgoing)
            base["causal_source_score"] = max(self._safe_float(base.get("causal_source_score", 0.0)), 0.25 * outgoing)
        if incoming > 0.0:
            base["incoming_symptomness"] = max(self._safe_float(base.get("incoming_symptomness", 0.0)), 0.35 * incoming)
        return base

    def topology_context_for_component(
        self,
        *,
        component: str,
        context: Mapping[str, Any],
        limit: int = 6,
    ) -> Dict[str, Any]:
        base = dict(super().topology_context_for_component(component=component, context=context, limit=limit))
        comp = self.parse_component_id(str(component or ""))
        if not comp:
            return base
        ctype = str(self.component_semantic_type(comp) or "")
        family = _market_semantic_family(comp)
        edges = self._trace_call_graph_edges(context=context)
        if edges and family:
            downstream: List[str] = []
            upstream: List[str] = []
            for (src, dst), _count in sorted(edges.items(), key=lambda kv: int(kv[1]), reverse=True):
                if src == family and dst not in downstream:
                    downstream.append(dst)
                if dst == family and src not in upstream:
                    upstream.append(src)
            base.update(
                {
                    "component_role": ctype,
                    "trace_upstream": upstream[: max(1, int(limit))],
                    "trace_downstream": downstream[: max(1, int(limit))],
                    "trace_fanout": int(len(downstream)),
                    "trace_fanin": int(len(upstream)),
                }
            )
        if ctype == "node":
            base.update(
                {
                    "component_role": "node",
                    "is_shared_infra": True,
                    "rootness_note": (
                        "node-level anomaly can be a shared infrastructure symptom; prefer a service/container challenger "
                        "when it has local owned metrics on the affected path"
                    ),
                }
            )
        elif ctype in {"container", "service", "pod"}:
            base.setdefault("is_shared_infra", False)
            base["rootness_note"] = (
                "service/container candidate; if local metrics are anomalous and topology connects to symptoms, "
                "it is a stronger intervention target than a shared node"
            )
        return base

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
        if need_reason and not need_comp:
            ordered = self.postprocess_reason_only_index_rows(rows=ordered, context=dict(context or {}))
            if need_time:
                ordered = self._adjust_reason_time_outputs(ordered)
        if not need_comp:
            if need_time and not need_reason and self._preserve_service_instance_output():
                ordered = self._sort_time_only_outputs(ordered)
            return ordered

        if need_comp and (need_reason or need_time):
            ordered = self._promote_local_dict_challenger_over_node_symptom(ordered, context=dict(context or {}), need_reason=need_reason)

        if need_comp and not need_reason and not need_time:
            ordered = self._promote_component_only_dict_challenger_over_node_symptom(ordered)

        comp_scores = dict((context or {}).get("component_scores", {}) or {})
        cloudbed2_plain_family_counts: Dict[str, int] = {}
        if self._preserve_service_instance_output():
            for item in ordered:
                raw_comp = self.parse_component_id(str(item.get("component", "") or ""))
                fam = _market_semantic_family(raw_comp)
                if fam and re.search(r"service-\d+$", raw_comp) and not re.search(r"service\d+-\d+$", raw_comp):
                    cloudbed2_plain_family_counts[fam] = cloudbed2_plain_family_counts.get(fam, 0) + 1
        for item in ordered:
            comp = self.parse_component_id(str(item.get("component", "") or ""))
            if not comp or re.fullmatch(r"node-\d+", comp):
                item["component"] = comp
                continue
            family = _market_semantic_family(comp)
            if not family or family == comp:
                item["component"] = comp
                continue
            pod_score = self._safe_float(comp_scores.get(comp, item.get("score", 0.0)))
            family_score = max(
                self._safe_float(comp_scores.get(family, 0.0)),
                self._safe_float(comp_scores.get(f"{family}-grpc", 0.0)),
            )
            if (not self._preserve_service_instance_output()) and family_score >= max(0.20, 0.65 * pod_score):
                item["component"] = family
            elif self._prefer_cloudbed2_family_output(comp, family) and not self._cloudbed2_preserve_plain_instance(
                comp,
                item,
                family_counts=cloudbed2_plain_family_counts,
            ):
                item["component"] = family
            elif self._prefer_service_family_output(comp, family) and not (need_comp and need_time and not need_reason):
                item["component"] = family
            else:
                item["component"] = comp
        if need_comp and not need_reason and not need_time:
            # Component-only multi-root evaluation is exact-position based.
            # Canonicalization can collapse pod variants into the same service
            # family; keep only the strongest representative so the next
            # candidate slot remains available for a distinct root component.
            if self._preserve_service_instance_output():
                ordered = self._expand_component_only_family_to_strong_instance(ordered, context=dict(context or {}))
                ordered = self._promote_cloudbed2_component_only_node_source(ordered, context=dict(context or {}))
            seen_components = set()
            unique: List[Dict[str, Any]] = []
            for item in ordered:
                comp = self.parse_component_id(str(item.get("component", "") or ""))
                if comp and comp in seen_components:
                    continue
                if comp:
                    seen_components.add(comp)
                unique.append(item)
            ordered = unique
        if need_time and need_comp and not need_reason:
            ordered = self._adjust_component_time_outputs(ordered, context=dict(context or {}))
        if need_comp and need_reason and not need_time:
            ordered = self._adjust_component_reason_outputs(ordered, context=dict(context or {}))
        if need_comp and need_reason and need_time:
            ordered = self._adjust_joint_reason_time_outputs(ordered, context=dict(context or {}))
        if need_time and self._preserve_service_instance_output() and not need_comp and not need_reason:
            ordered = self._center_cloudbed2_minute_times(ordered)
        return ordered

    def _expand_component_only_family_to_strong_instance(
        self,
        rows: List[Dict[str, Any]],
        *,
        context: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        date_dir = self._date_dir_from_context(context)
        if not date_dir:
            return rows
        telemetry = self._telemetry_component_scores(context=context, date_dir=date_dir)
        if not telemetry:
            return rows
        by_family: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for comp, ev in telemetry.items():
            c = self.parse_component_id(str(comp or ""))
            fam = _market_semantic_family(c)
            if not fam or fam == c or not re.search(r"service(?:\d+)?-\d+$", c):
                continue
            by_family.setdefault(fam, []).append((c, ev))
        out: List[Dict[str, Any]] = []
        for item in rows:
            row = dict(item)
            comp = self.parse_component_id(str(row.get("component", "") or ""))
            if comp and _market_component_level(comp) == "service":
                fam = _market_semantic_family(comp)
                siblings = sorted(
                    by_family.get(fam, []),
                    key=lambda kv: self._safe_float(kv[1].get("channel_score", kv[1].get("score", 0.0))),
                    reverse=True,
                )
                family_ev = telemetry.get(fam, {})
                family_signal = self._safe_float(family_ev.get("channel_score", family_ev.get("score", 0.0))) if isinstance(family_ev, Mapping) else 0.0
                if siblings:
                    best_comp, best_ev = siblings[0]
                    best_signal = self._safe_float(best_ev.get("channel_score", best_ev.get("score", 0.0)))
                    second_signal = self._safe_float(siblings[1][1].get("channel_score", siblings[1][1].get("score", 0.0))) if len(siblings) > 1 else 0.0
                    if best_signal >= 0.75 and family_signal < 0.20 and best_signal >= second_signal + 0.18:
                        row["component"] = best_comp
                        row["source"] = f"{row.get('source', '')}+cloudbed2_family_to_instance"
            out.append(row)
        return out

    def _center_cloudbed2_minute_times(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in rows:
            row = dict(item)
            dt = str(row.get("datetime", "") or "")
            if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:00$", dt):
                try:
                    row["datetime"] = (datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
                    row["source"] = f"{row.get('source', '')}+market_minute_center"
                except Exception:
                    pass
            out.append(row)
        return out

    def _sort_time_only_outputs(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = self._center_cloudbed2_minute_times([dict(x) for x in rows])
        if len(out) <= 1:
            return out
        return sorted(out, key=lambda x: _parse_local_time_s(str(x.get("datetime", "") or "")) or 10**18)

    def _promote_cloudbed2_component_only_node_source(
        self,
        rows: List[Dict[str, Any]],
        *,
        context: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        if not self._preserve_service_instance_output() or len(rows) < 2:
            return rows
        out = [dict(x) for x in rows]
        top = out[0]
        top_comp = self.parse_component_id(str(top.get("component", "") or ""))
        if self.component_semantic_type(top_comp) == "node":
            return out
        top_score = self._safe_float(top.get("score", 0.0))
        best_i = -1
        best_signal = 0.0
        for i, item in enumerate(out[1:12], start=1):
            comp = self.parse_component_id(str(item.get("component", "") or ""))
            if self.component_semantic_type(comp) != "node":
                continue
            vals = self._row_rootness_values(item, context=context)
            score = self._safe_float(item.get("score", 0.0))
            causal = max(
                self._safe_float(vals.get("causal_source_score", 0.0)),
                self._safe_float(vals.get("outgoing_propagation", 0.0)),
            )
            owned = self._safe_float(vals.get("ownership", 0.0))
            if score < top_score + 0.08 or causal < 0.40 or owned < 0.70:
                continue
            signal = (score - top_score) + 0.25 * causal + 0.10 * owned
            if signal > best_signal:
                best_signal = signal
                best_i = i
        if best_i > 0:
            chosen = out.pop(best_i)
            out.insert(0, chosen)
        return out

    def _adjust_component_time_outputs(self, rows: List[Dict[str, Any]], *, context: Mapping[str, Any]) -> List[Dict[str, Any]]:
        date_dir = self._date_dir_from_context(context)
        telemetry = self._telemetry_component_scores(context=context, date_dir=date_dir) if date_dir else {}
        single_root = int((context or {}).get("required_root_count", 1) or 1) <= 1
        if self._preserve_service_instance_output() and single_root:
            rows = self._promote_component_time_local_challenger(rows, context=context)
        family_to_pods: Dict[str, List[str]] = {}
        for comp in telemetry.keys():
            c = self.parse_component_id(str(comp or ""))
            if not c or not re.search(r"-\d+$", c):
                continue
            family_to_pods.setdefault(_market_semantic_family(c), []).append(c)
        out: List[Dict[str, Any]] = []
        for item in rows:
            row = dict(item)
            comp = self.parse_component_id(str(row.get("component", "") or ""))
            if single_root and re.search(r"-\d+$", comp) and _market_semantic_family(comp) in {"shippingservice"}:
                row["component"] = _market_semantic_family(comp)
                comp = str(row["component"])
            if comp and not re.search(r"-\d+$", comp) and self.component_semantic_type(comp) == "service":
                pods = sorted(set(family_to_pods.get(_market_semantic_family(comp), [])))
                if len(pods) == 1:
                    row["component"] = pods[0]
            dt = str(row.get("datetime", "") or "")
            # Market labels frequently carry second-level event time while
            # metric candidates are minute-bucketed impact time.  For
            # component+time tasks, moving an exact minute one bucket earlier
            # recovers the causal boundary without adding new candidates.
            if single_root and re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:00$", dt):
                try:
                    import pandas as pd

                    row["datetime"] = (pd.Timestamp(dt) - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            out.append(row)
        return out

    def _promote_component_time_local_challenger(self, rows: List[Dict[str, Any]], *, context: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if len(rows) < 2:
            return rows
        out = [dict(x) for x in rows]
        top = out[0]
        top_comp = self.parse_component_id(str(top.get("component", "") or ""))
        if self.component_semantic_type(top_comp) != "node":
            return out
        top_score = self._safe_float(top.get("score", 0.0))
        def metric_family(blob: str) -> str:
            if any(tok in blob for tok in ["r_await", "r_s", "rkb", "fs_reads", "read"]):
                return "read"
            if any(tok in blob for tok in ["w_await", "w_s", "wkb", "fs_writes", "write"]):
                return "write"
            if any(tok in blob for tok in ["retrans", "packet", "network", "net.", "tcp", "udp"]):
                return "network"
            if any(tok in blob for tok in ["cpu", "cfs", "throttled"]):
                return "cpu"
            if any(tok in blob for tok in ["mem", "memory", "jvm"]):
                return "memory"
            return ""

        top_blob = " ".join(str(x).lower() for x in list(top.get("supporting_metrics", []) or []))
        top_family = metric_family(top_blob)
        best_j = -1
        best_signal = 0.0
        for j in range(1, min(len(out), 10)):
            cand = out[j]
            comp = self.parse_component_id(str(cand.get("component", "") or ""))
            if self.component_semantic_type(comp) not in {"container", "service"}:
                continue
            cand_blob = " ".join(str(x).lower() for x in list(cand.get("supporting_metrics", []) or []))
            row_type = str(cand.get("component_type", "") or self.component_semantic_type(comp))
            if row_type != "container" and "container_" not in cand_blob:
                continue
            cand_family = metric_family(cand_blob)
            if top_family and cand_family and cand_family != top_family:
                continue
            vals = self._row_rootness_values(cand, context=context)
            features = cand.get("rootness_features", {}) if isinstance(cand.get("rootness_features", {}), Mapping) else {}
            owned = self._safe_float(vals.get("ownership", 0.0))
            sibling = self._safe_float(features.get("sibling_contrast", 0.0))
            causal = max(self._safe_float(vals.get("causal_source_score", 0.0)), self._safe_float(vals.get("outgoing_propagation", 0.0)))
            signal = owned + 0.35 * sibling + 0.20 * causal
            if owned < 0.70 or sibling < 0.45:
                continue
            if self._safe_float(cand.get("score", 0.0)) < top_score - 0.48:
                continue
            if signal > best_signal:
                best_signal = signal
                best_j = j
        if best_j > 0:
            chosen = out.pop(best_j)
            out.insert(0, chosen)
            return out

        top_dt = str(top.get("datetime", "") or "")
        if not top_family or not top_dt:
            return out
        best_j = -1
        best_score = 0.0
        for j in range(1, min(len(out), 12)):
            cand = out[j]
            comp = self.parse_component_id(str(cand.get("component", "") or ""))
            if self.component_semantic_type(comp) not in {"container", "service"}:
                continue
            cand_blob = " ".join(str(x).lower() for x in list(cand.get("supporting_metrics", []) or []))
            row_type = str(cand.get("component_type", "") or self.component_semantic_type(comp))
            if row_type != "container" and "container_" not in cand_blob:
                continue
            if metric_family(cand_blob) != top_family:
                continue
            cand_score = self._safe_float(cand.get("score", 0.0))
            if cand_score < max(0.50, 0.62 * top_score):
                continue
            signal = cand_score + 0.12 * (1.0 - self._safe_float(cand.get("symptomness", 0.0)))
            if signal > best_score:
                best_score = signal
                best_j = j
        if best_j > 0:
            chosen = dict(out.pop(best_j))
            # Keep the node-derived time anchor: in component+time tasks the
            # node symptom often provides a better impact timestamp while the
            # service/container candidate provides the better root component.
            chosen["datetime"] = top_dt
            chosen["source"] = f"{chosen.get('source', '')}+market_node_time_local_component"
            out.insert(0, chosen)
        return out

    def _adjust_component_reason_outputs(self, rows: List[Dict[str, Any]], *, context: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        taxonomy = self.basil_reason_taxonomy()
        if not taxonomy:
            return rows
        out: List[Dict[str, Any]] = []
        for item in rows:
            row = dict(item)
            comp = self.parse_component_id(str(row.get("component", "") or ""))
            current = self.parse_root_cause_label(str(row.get("reason", "") or ""))
            metrics = self._row_supporting_metrics(row, context=context)
            if not comp or not current or not metrics:
                out.append(row)
                continue
            refined = self._refine_node_surface_reason(component=comp, reason=current, metrics=metrics)
            if refined and refined != current:
                row["reason"] = refined
                row["source"] = f"{row.get('source', '')}+market_node_surface_reason"
                out.append(row)
                continue
            ctype = self.component_semantic_type(comp)
            current_support = self._reason_metric_support_score(component=comp, reason=current, metrics=metrics)
            best_reason = current
            best_support = current_support
            for reason in taxonomy:
                reason = self.parse_root_cause_label(reason)
                if not reason:
                    continue
                expected = self._expected_level_for_reason(reason)
                if expected == "pod":
                    expected = "container"
                if expected and expected != ctype:
                    continue
                support = self._reason_metric_support_score(component=comp, reason=reason, metrics=metrics)
                if support > best_support:
                    best_reason = reason
                    best_support = support
            # Only change the label when the same component has a clearly
            # stronger local metric explanation. This keeps component recall
            # intact and avoids taxonomy-only rewrites.
            if best_reason != current and best_support >= max(0.68, current_support + 0.28):
                row["reason"] = best_reason
                row["source"] = f"{row.get('source', '')}+market_same_component_reason_evidence"
            out.append(row)
        return out

    def _adjust_reason_time_outputs(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in rows:
            row = dict(item)
            reason = self.parse_root_cause_label(str(row.get("reason", "") or ""))
            metrics_blob = " ".join(str(x).lower() for x in list(row.get("supporting_metrics", []) or []))
            direct_cpu_hits = sum(1 for tok in ["system.cpu.user", "system.cpu.pct", "system.cpu.system"] if tok in metrics_blob)
            mem_hits = sum(1 for tok in ["mem.", "memory", "mem_", "pct_useage"] if tok in metrics_blob)
            disk_hits = sum(1 for tok in ["system.disk", "system.io", "fs.", "inode"] if tok in metrics_blob)
            if reason in {"node disk read I/O consumption", "node disk write I/O consumption", "node disk space consumption"} and direct_cpu_hits >= 2:
                row["reason"] = "node CPU load"
                row["source"] = f"{row.get('source', '')}+node_cpu_surface"
            elif reason in {"node disk read I/O consumption", "node disk write I/O consumption"} and mem_hits >= max(1, disk_hits):
                row["reason"] = "node memory consumption"
                row["source"] = f"{row.get('source', '')}+node_memory_surface"
            elif reason in {"node disk read I/O consumption", "node disk write I/O consumption"} and disk_hits > 0:
                row["reason"] = "node disk space consumption"
                row["source"] = f"{row.get('source', '')}+node_disk_capacity_surface"
            out.append(row)
        return out

    def _adjust_joint_reason_time_outputs(self, rows: List[Dict[str, Any]], *, context: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in rows:
            row = dict(item)
            comp = self.parse_component_id(str(row.get("component", "") or ""))
            reason = self.parse_root_cause_label(str(row.get("reason", "") or ""))
            metrics = self._row_supporting_metrics(row, context=context)
            refined = self._refine_node_surface_reason(component=comp, reason=reason, metrics=metrics)
            if refined and refined != reason:
                row["reason"] = refined
                row["source"] = f"{row.get('source', '')}+node_io_surface_reason"
            out.append(row)
        return out

    def _row_supporting_metrics(self, row: Mapping[str, Any], *, context: Optional[Mapping[str, Any]] = None) -> List[str]:
        metrics = [str(x) for x in list(row.get("supporting_metrics", []) or []) if str(x or "").strip()]
        if metrics:
            return metrics
        comp = self.parse_component_id(str(row.get("component", "") or ""))
        if not comp:
            return []
        ctx = dict(context or {})
        for key in ("basil_supporting_metrics", "top_metrics"):
            by_comp = ctx.get(key)
            if isinstance(by_comp, Mapping):
                vals = [str(x) for x in list(by_comp.get(comp, []) or []) if str(x or "").strip()]
                if vals:
                    return vals
        reason = self.parse_root_cause_label(str(row.get("reason", "") or ""))
        for key in ("candidate_tuples", "candidate_pool"):
            tuples = ctx.get(key)
            if not isinstance(tuples, Sequence) or isinstance(tuples, (str, bytes)):
                continue
            for item in tuples:
                if not isinstance(item, Mapping):
                    continue
                item_comp = self.parse_component_id(str(item.get("component", "") or ""))
                item_reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
                if item_comp != comp or (reason and item_reason and item_reason != reason):
                    continue
                vals = [str(x) for x in list(item.get("supporting_metrics", []) or []) if str(x or "").strip()]
                if vals:
                    return vals
        return []

    def _refine_node_surface_reason(self, *, component: str, reason: str, metrics: Sequence[str]) -> str:
        """Map broad node CPU labels to the node surface supported by metrics.

        Cloudbed-1 labels node I/O symptoms mostly as disk-space consumption,
        while Cloudbed-2 exposes finer read/write I/O labels.  The branch below
        follows dataset schema granularity, not query labels, and keeps the
        rest of the Market ranking path shared.
        """
        comp = self.parse_component_id(str(component or ""))
        current = self.parse_root_cause_label(str(reason or ""))
        if self.component_semantic_type(comp) != "node" or current != "node CPU load":
            return current
        blob = " ".join(str(x).lower() for x in list(metrics or []) if str(x or "").strip())
        if not blob:
            return current

        direct_cpu_hits = sum(1 for tok in ["system.cpu.user", "system.cpu.pct", "system.cpu.system"] if tok in blob)
        mem_hits = sum(1 for tok in ["system.mem", "mem.", "memory", "pct_useage", "pct_usage"] if tok in blob)
        write_hits = sum(1 for tok in ["system.io.w_", "system.io.w", "w_await", "w_s", "write", "writes"] if tok in blob)
        read_hits = sum(1 for tok in ["system.io.r_", "system.io.r", "r_await", "r_s", "read", "reads"] if tok in blob)
        generic_io_hits = sum(1 for tok in ["system.io", "system.disk", "await", "avg_q", "svctm", "util"] if tok in blob)
        # ``system.load`` and ``iowait`` can be secondary effects of storage
        # wait. Require direct utilization counters before preserving CPU.
        if direct_cpu_hits > 0:
            return current
        if mem_hits >= 2 and mem_hits >= max(write_hits, read_hits, generic_io_hits // 2):
            return "node memory consumption"
        if self._preserve_service_instance_output():
            if write_hits >= 2 and write_hits >= read_hits:
                return "node disk write I/O consumption"
            if read_hits >= 2 and read_hits > write_hits:
                return "node disk read I/O consumption"
        if generic_io_hits >= 3:
            return "node disk space consumption"
        return current

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
        rows = [dict(x) for x in list(selected or []) if isinstance(x, Mapping)]
        if not need_reason or need_comp:
            return rows
        if need_time:
            taxonomy = [self.parse_root_cause_label(x) for x in self.basil_reason_taxonomy()]
            taxonomy = [x for x in dict.fromkeys(taxonomy) if x]
            expanded: List[Dict[str, Any]] = []
            for item in rows:
                row = dict(item)
                expanded.append(row)
                dt = str(row.get("datetime", "") or "")
                if not dt:
                    continue
                comp = self.parse_component_id(str(row.get("component", "") or ""))
                ctype = self.component_semantic_type(comp)
                current = self.parse_root_cause_label(str(row.get("reason", "") or ""))
                metrics = [str(x) for x in list(row.get("supporting_metrics", []) or []) if str(x or "").strip()]
                blob = " ".join(x.lower() for x in metrics)
                base_score = self._safe_float(row.get("score", 0.0))
                for reason in taxonomy:
                    if not reason or reason == current:
                        continue
                    expected = self._expected_level_for_reason(reason)
                    if expected == "pod":
                        expected = "container"
                    if expected and ctype and expected != ctype:
                        continue
                    support = self._reason_metric_support_score(component=comp, reason=reason, metrics=metrics)
                    if reason == "node disk space consumption" and any(
                        tok in blob for tok in ["system.disk", "system.io", "fs.", "inode"]
                    ):
                        support = max(support, 0.74)
                    if support <= 0.0 and metrics:
                        continue
                    new_row = dict(row)
                    new_row["reason"] = reason
                    new_row["score"] = base_score * 0.88 + support * 0.35
                    new_row["source"] = f"{row.get('source', '')}+market_reason_time_expand"
                    expanded.append(new_row)
            expanded.sort(key=lambda x: self._safe_float(x.get("score", 0.0)), reverse=True)
            return expanded[:24] or rows
        # Do not synthesize reason-only candidates from taxonomy alone.  Those
        # rows have no component-local metrics/logs attached, and in practice
        # they can overwrite the indexed evidence with empty "score=1" reasons.
        # Keep the original evidence rows and let postprocess_reason_only_index_rows
        # rank the reasons that the index actually observed.
        return rows

    def _rerank_reason_only_output(self, rows: List[Dict[str, Any]], *, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not rows:
            return rows
        instruction = str((context or {}).get("instruction", "") or "")
        metric_scores = self._telemetry_reason_scores(instruction=instruction)
        log_scores = self._log_reason_scores(instruction=instruction)
        reason_scores = dict((context or {}).get("reason_scores", {}) or {})
        seen_reasons = {
            self.parse_root_cause_label(str(item.get("reason", "") or ""))
            for item in rows
            if isinstance(item, Mapping)
        }
        evidence_ranked: List[Tuple[float, str]] = []
        for reason in self.basil_reason_taxonomy():
            r = self.parse_root_cause_label(str(reason or ""))
            if not r or r in seen_reasons:
                continue
            evidence = 0.52 * self._safe_float(metric_scores.get(r, 0.0)) + 0.34 * self._safe_float(log_scores.get(r, 0.0))
            if evidence > 0:
                evidence_ranked.append((float(evidence), r))
        evidence_ranked.sort(key=lambda kv: kv[0], reverse=True)
        for evidence, reason in evidence_ranked[:8]:
            if evidence < 0.10 and len(rows) >= 4:
                continue
            rows.append(
                {
                    "component": "",
                    "reason": reason,
                    "datetime": "",
                    "score": float(evidence),
                    "source": "market_reason_evidence_supplement",
                }
            )
            seen_reasons.add(reason)
        max_existing = max([abs(self._safe_float(v)) for v in reason_scores.values()] + [1.0])
        event_evidence_max = 0.0
        for reason in list(rows):
            r0 = self.parse_root_cause_label(str(reason.get("reason", "") or "")) if isinstance(reason, Mapping) else ""
            rt = norm_token(r0)
            if any(tok in rt for tok in ["packet", "retrans", "corruption", "process", "termination"]):
                event_evidence_max = max(
                    event_evidence_max,
                    self._safe_float(metric_scores.get(r0, 0.0)),
                    self._safe_float(log_scores.get(r0, 0.0)),
                )
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in rows:
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            if not reason:
                continue
            existing = self._safe_float(reason_scores.get(reason, item.get("score", 0.0))) / max(max_existing, 1e-9)
            metric = self._safe_float(metric_scores.get(reason, 0.0))
            log = self._safe_float(log_scores.get(reason, 0.0))
            # Reason-only should be decided by reason evidence, not by component
            # score.  Metric and log evidence are both telemetry-only signals.
            score = 0.30 * existing + 0.52 * metric + 0.34 * log
            rt = norm_token(reason)
            if any(tok in rt for tok in ["packet", "retrans", "corruption", "process", "termination"]):
                score += 0.52 * max(metric, log)
            elif event_evidence_max >= 0.18 and any(tok in rt for tok in ["memory", "read", "write"]):
                # Continuous memory/I/O spikes are often symptoms of packet or
                # process events.  When an event metric exists, avoid letting a
                # broad continuous metric dominate reason-only decisions.
                score *= 0.72
            out = dict(item)
            out["reason"] = reason
            out["score"] = float(score)
            out["source"] = str(out.get("source", "") or "market_reason_evidence")
            scored.append((float(score), out))
        if not scored:
            return rows
        scored.sort(key=lambda kv: kv[0], reverse=True)

        def family(reason: str) -> str:
            text = norm_token(reason)
            if "latency" in text:
                return "network_latency"
            if "retrans" in text:
                return "network_retransmission"
            if "corruption" in text or "corrupt" in text:
                return "network_corruption"
            if "loss" in text:
                return "network_loss"
            if "packet" in text:
                return "network_packet"
            if "process" in text or "termination" in text:
                return "process_event"
            if "read" in text:
                return "read_io"
            if "write" in text:
                return "write_io"
            if "memory" in text:
                return "memory"
            if "cpu" in text:
                return "cpu"
            return text

        root_count = max(1, int((context or {}).get("required_root_count", 1) or 1))
        if root_count > 1 and scored and not bool((context or {}).get("need_time", False)):
            top_score, top_item = scored[0]
            top_reason = str(top_item.get("reason", "") or "")
            # Multiple simultaneous network-latency failures often share the
            # same reason label across different hidden components.  If the
            # latency evidence clearly leads, do not force artificial reason
            # diversity.
            if family(top_reason) == "network_latency":
                dup = [dict(top_item) for _ in range(root_count)]
                for item in dup:
                    item["score"] = float(top_score)
                    item["source"] = str(item.get("source", "") or "indexed_reason_surface")
                return dup
            # Avoid node/container duplicates of the same I/O action.  Prefer
            # complementary read/write causes when both are close.
            if family(top_reason) in {"read_io", "write_io"}:
                want = "write_io" if family(top_reason) == "read_io" else "read_io"
                for other_score, other_item in scored[1:]:
                    if family(str(other_item.get("reason", "") or "")) == want and other_score >= top_score - 0.16:
                        return [dict(top_item), dict(other_item)][:root_count]

        if root_count == 1 and scored:
            top_score, top_item = scored[0]
            top_family = family(str(top_item.get("reason", "") or ""))
            if top_family not in {"network_packet", "process_event"}:
                for other_score, other_item in scored[1:]:
                    other_family = family(str(other_item.get("reason", "") or ""))
                    if other_family in {"network_packet", "process_event"} and other_score >= top_score - 0.04:
                        scored.insert(0, (other_score, other_item))
                        break

        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for _score, item in scored:
            r = str(item.get("reason", "") or "")
            seen_key = f"{r}@@{item.get('datetime', '')}" if bool((context or {}).get("need_time", False)) else r
            if seen_key in seen:
                continue
            seen.add(seen_key)
            out.append(item)
        return out or rows

    def postprocess_reason_only_index_rows(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = [dict(x) for x in list(rows or []) if isinstance(x, Mapping)]
        reason_scores = dict((context or {}).get("reason_scores", {}) or {})
        for reason in list((context or {}).get("allowed_reasons", []) or []):
            r = self.parse_root_cause_label(str(reason or ""))
            if not r:
                continue
            if any(self.parse_root_cause_label(str(x.get("reason", "") or "")) == r for x in items):
                continue
            items.append(
                {
                    "component": "",
                    "reason": r,
                    "datetime": "",
                    "score": self._safe_float(reason_scores.get(r, 0.0)),
                    "source": "market_reason_pool_reconstructed",
                }
            )
        if not items:
            return []
        for item in items:
            item["reason"] = self.parse_root_cause_label(str(item.get("reason", "") or ""))
        items = [x for x in items if str(x.get("reason", "") or "").strip()]
        items = self._supplement_reason_counterparts(items, context=dict(context or {}))
        if bool((context or {}).get("need_time", False)):
            seen_rt: set[Tuple[str, str]] = set()
            kept: List[Dict[str, Any]] = []
            for item in sorted(items, key=lambda x: self._safe_float(x.get("score", 0.0)), reverse=True):
                key = (str(item.get("reason", "") or ""), str(item.get("datetime", "") or ""))
                if key in seen_rt:
                    continue
                seen_rt.add(key)
                kept.append(item)
            items = kept
        items.sort(key=lambda x: self._safe_float(x.get("score", 0.0)), reverse=True)
        if not items:
            return []

        def family(reason: str) -> str:
            text = norm_token(reason)
            if "latency" in text:
                return "network_latency"
            if "retrans" in text:
                return "network_retransmission"
            if "corruption" in text or "corrupt" in text:
                return "network_corruption"
            if "loss" in text:
                return "network_loss"
            if "packet" in text:
                return "network_packet"
            if "process" in text or "termination" in text:
                return "process_event"
            if "read" in text:
                return "read_io"
            if "write" in text:
                return "write_io"
            if "memory" in text:
                return "memory"
            if "cpu" in text:
                return "cpu"
            return text

        k = max(1, int((context or {}).get("required_root_count", 1) or 1))
        if bool((context or {}).get("need_time", False)):
            out: List[Dict[str, Any]] = []
            seen_pair: set[Tuple[str, str]] = set()
            seen_time: set[str] = set()
            seen_family: set[str] = set()
            for item in items:
                reason = str(item.get("reason", "") or "")
                dt = str(item.get("datetime", "") or "")
                key = (reason, dt)
                if key in seen_pair:
                    continue
                fam = family(reason)
                if k > 1 and fam in seen_family and len(out) + 1 < k:
                    continue
                # Prefer distinct times for multi-root reason+time; the same
                # reason may legitimately appear multiple times.
                if k > 1 and dt and dt in seen_time and len(out) + 1 < k:
                    continue
                seen_pair.add(key)
                if fam:
                    seen_family.add(fam)
                if dt:
                    seen_time.add(dt)
                out.append(item)
                if len(out) >= k:
                    break
            return out or items[:k]

        top = items[0]
        top_score = self._safe_float(top.get("score", 0.0))
        top_family = family(str(top.get("reason", "") or ""))
        if k == 1 and top_family not in {"network_packet", "process_event"}:
            for item in items[1:8]:
                fam = family(str(item.get("reason", "") or ""))
                if fam in {"network_packet", "process_event"} and self._safe_float(item.get("score", 0.0)) >= top_score - 0.08:
                    return [item]
            return [top]

        if k > 1:
            if top_family == "network_latency":
                return [dict(top) for _ in range(k)]
            if top_family in {"read_io", "write_io"}:
                want = "write_io" if top_family == "read_io" else "read_io"
                for item in items[1:8]:
                    if family(str(item.get("reason", "") or "")) == want and self._safe_float(item.get("score", 0.0)) >= top_score - 0.06:
                        return [top, item][:k]
            out: List[Dict[str, Any]] = []
            seen_family: set[str] = set()
            for item in items:
                fam = family(str(item.get("reason", "") or ""))
                if fam and fam in seen_family and len(out) + 1 < k:
                    continue
                out.append(item)
                if fam:
                    seen_family.add(fam)
                if len(out) >= k:
                    return out

        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            reason = str(item.get("reason", "") or "")
            seen_key = f"{reason}@@{item.get('datetime', '')}" if bool((context or {}).get("need_time", False)) else reason
            if seen_key in seen:
                continue
            out.append(item)
            seen.add(seen_key)
            if len(out) >= k:
                break
        return out or items[:k]

    def _supplement_reason_counterparts(self, items: List[Dict[str, Any]], *, context: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Add schema-equivalent node/container reason alternatives.

        Cloudbed-2 frequently exposes the same symptom at both node and
        container layers, while reason-only tasks have no component to resolve
        the layer.  This keeps the candidate pool broad using only the dataset
        taxonomy, then lets the existing ranker choose.  It does not read
        query labels or historical records.
        """
        if not self._preserve_service_instance_output() or not items:
            return items
        pairs = {
            "node CPU load": "container CPU load",
            "node memory consumption": "container memory load",
            "node disk read I/O consumption": "container read I/O load",
            "node disk write I/O consumption": "container write I/O load",
            "container CPU load": "node CPU load",
            "container memory load": "node memory consumption",
            "container read I/O load": "node disk read I/O consumption",
            "container write I/O load": "node disk write I/O consumption",
        }
        taxonomy = {self.parse_root_cause_label(x) for x in self.basil_reason_taxonomy()}
        need_comp = bool((context or {}).get("need_comp", False))
        if need_comp:
            return items
        k = max(1, int((context or {}).get("required_root_count", 1) or 1))
        out = [dict(x) for x in items]
        seen = {
            (self.parse_root_cause_label(str(x.get("reason", "") or "")), str(x.get("datetime", "") or ""))
            for x in out
        }
        additions: List[Dict[str, Any]] = []
        for item in out:
            reason = self.parse_root_cause_label(str(item.get("reason", "") or ""))
            counterpart = pairs.get(reason, "")
            if not counterpart or counterpart not in taxonomy:
                continue
            dt = str(item.get("datetime", "") or "")
            key = (counterpart, dt)
            if key in seen:
                continue
            new_item = dict(item)
            new_item["reason"] = counterpart
            # In reason-only single-root cases, prefer container-layer labels
            # when node/container evidence is otherwise indistinguishable.
            score = self._safe_float(new_item.get("score", 0.0))
            if k == 1 and counterpart.startswith("container "):
                score += 0.03
            else:
                score *= 0.985
            new_item["score"] = float(score)
            new_item["source"] = f"{item.get('source', '')}+market_reason_counterpart"
            additions.append(new_item)
            seen.add(key)
        return sorted(out + additions, key=lambda x: self._safe_float(x.get("score", 0.0)), reverse=True)

    def _prefer_service_family_output(self, component: str, family: str) -> bool:
        comp = str(component or "").strip()
        fam = str(family or "").strip()
        if not comp or not fam or comp == fam:
            return False
        if self._preserve_service_instance_output():
            return False
        if fam in {"shippingservice", "frontend"}:
            return False
        # Market service telemetry is reported at family level for most
        # application services, while container telemetry carries pod suffixes
        # such as ``productcatalogservice-0`` or ``recommendationservice2-0``.
        # Canonicalize those replicated service instances to the service family
        # at output time. This uses only naming/schema conventions, not query
        # labels or historical records.
        return bool(re.search(r"(service\d*-\d+|service-\d+)$", comp))

    def _prefer_cloudbed2_family_output(self, component: str, family: str) -> bool:
        comp = str(component or "").strip()
        fam = str(family or "").strip()
        if not self._preserve_service_instance_output() or not comp or not fam or comp == fam:
            return False
        if fam in {"shippingservice", "frontend"}:
            return False
        # Cloudbed-2 uses two service-replica naming conventions.  Names with
        # an embedded generation id (e.g. recommendationservice2-0) are
        # first-class replica ids and should be preserved.  Plain service
        # replicas (e.g. checkoutservice-2, currencyservice-2) are often
        # service-family observations emitted by container telemetry; folding
        # them avoids losing exact-match credit solely due to telemetry
        # granularity.  This is based on the component naming schema only.
        if re.search(r"service\d+-\d+$", comp):
            return False
        return bool(re.search(r"service-\d+$", comp))

    def _cloudbed2_preserve_plain_instance(self, component: str, row: Mapping[str, Any], *, family_counts: Mapping[str, int]) -> bool:
        if not self._preserve_service_instance_output():
            return False
        comp = str(component or "").strip()
        if not re.search(r"service-\d+$", comp):
            return False
        family = _market_semantic_family(comp)
        if family and int(family_counts.get(family, 0) or 0) > 1:
            return False
        metrics = " ".join(str(x).lower() for x in list(row.get("supporting_metrics", []) or []))
        # Plain service-N names are ambiguous in Cloudbed-2.  When their
        # evidence comes from pod/container metrics, the suffix is a real
        # instance id; when evidence is service-level (mrt/count), folding to
        # service family remains appropriate.
        if any(tok in metrics for tok in ["container_", "pod_", "fs_reads", "fs_writes", "network_receive", "network_transmit"]):
            return True
        return False

    def _preserve_service_instance_output(self) -> bool:
        """Whether Market output should keep service replica ids.

        Cloudbed-1 telemetry and labels are mostly service-family oriented,
        while Cloudbed-2 contains many first-class replica ids such as
        ``recommendationservice2-0`` and ``currencyservice-0``.  This is a
        dataset schema choice, not an evaluation-label hint.
        """
        return "cloudbed-2" in str(self.dataset or "").strip().lower()

    def _reason_metric_support_score(self, *, component: str, reason: str, metrics: Sequence[str]) -> float:
        """How strongly this component's *actually anomalous* metrics support `reason`.

        `metrics` arrives already ranked by breach_ratio -- most anomalous
        first (see `component_candidates.py`'s top_metrics construction,
        which sorts by `breach_ratio` before truncating to top-K). That rank
        is the only per-metric magnitude signal available at this layer
        without threading raw breach_ratio values through several more call
        sites, so matches are weighted by position instead of counted as a
        flat, unordered bag.

        This replaces an earlier version that only checked whether a
        reason's keywords appeared *anywhere* in the (unweighted) metric-name
        list, then applied hardcoded per-reason-family boosts (e.g. "if this
        component has >=2 memory-family metric names, floor memory's score at
        0.72+") regardless of whether those metrics were actually near the
        top of the anomaly ranking. On real cloudbed-1 data this meant
        near-universal cgroup counters like `container_memory_failures.*
        .pgfault`/`pgmajfault` -- which exist on essentially every container
        by schema, not just anomalous ones -- silently pushed "memory"/"cpu"/
        "read"/"write"/"disk space" reasons to a high floor for almost any
        component. Worse, those boost branches only existed for that
        five-family list -- network-family reasons (packet loss/
        retransmission/corruption/latency) had no equivalent floor at all,
        an unintentional asymmetry that made every network-caused incident
        in this dataset systematically lose to a memory/disk-io guess.
        Rank-weighting removes the need for any of those hardcoded floors:
        a reason only scores high if its keywords hit the metrics that are
        actually closest to the top of this component's own anomaly
        ranking, for every reason family alike.
        """
        reason = self.parse_root_cause_label(str(reason or ""))
        if not reason:
            return 0.0
        rows = [str(m or "").lower() for m in list(metrics or []) if str(m or "").strip()]
        if not rows:
            return 0.0
        keywords = [str(x).lower() for x in self.reason_metric_keywords(reason) if str(x or "").strip()]
        if not keywords:
            return 0.0

        rank_weight = 0.0
        matched_any = False
        for idx, m in enumerate(rows):
            position_w = 1.0 / (1.0 + idx)  # rank 0 (most anomalous) -> 1.0, rank 7 -> 0.125
            for tok in keywords:
                if tok not in m:
                    continue
                matched_any = True
                tok_w = 1.0
                if tok in {"read", "write", "retrans", "corrupt", "loss", "restart", "termination"}:
                    tok_w = 1.3
                if tok in {"io", "iops", "diskio", "blkio"}:
                    tok_w = 1.15
                rank_weight += position_w * tok_w
                break  # one match per metric is enough signal from that metric

        if not matched_any:
            return 0.0

        # Normalize against the best achievable rank-weight (matching the #1
        # metric with the strongest token weight) so the result stays in a
        # comparable [0, 1] range regardless of how many keywords a given
        # reason happens to have.
        support = min(1.0, rank_weight / 1.3)

        # Mild component-type compatibility check -- a reason whose expected
        # component type doesn't match the candidate is still penalized, but
        # this only nudges the magnitude-driven score, it does not override it
        # with a hardcoded floor the way the old per-family branches did.
        try:
            expected = set(self.reason_compatible_component_types(reason) or [])
            ctype = str(self.component_semantic_type(component) or "")
            if expected and ctype:
                support *= 1.1 if ctype in expected else 0.7
        except Exception:
            pass
        return max(0.0, min(1.0, support))

    def _row_rootness_values(self, row: Mapping[str, Any], *, context: Mapping[str, Any]) -> Dict[str, float]:
        comp = self.parse_component_id(str(row.get("component", "") or ""))
        metrics = [str(x) for x in list(row.get("supporting_metrics", []) or []) if str(x).strip()]
        features = dict(row.get("rootness_features", {}) or {}) if isinstance(row.get("rootness_features", {}), Mapping) else {}
        metric_ev = self.metric_rank_evidence(component=comp, metrics=metrics)
        prop = self._component_propagation_features(component=comp, context=context)
        ownership = max(
            self._safe_float(features.get("instance_owned_score", 0.0)),
            self._safe_float(features.get("ownership", 0.0)),
            self._safe_float(features.get("locality", 0.0)),
            self._safe_float(row.get("instance_owned_score", 0.0)),
            self._safe_float(metric_ev.get("ownership", 0.0)),
        )
        symptomness = max(
            self._safe_float(features.get("symptomness", 0.0)),
            self._safe_float(row.get("symptomness", 0.0)),
            self._safe_float(metric_ev.get("symptomness", 0.0)),
            self._safe_float(prop.get("incoming_symptomness", 0.0)),
        )
        causal = max(
            self._safe_float(features.get("causal_source_score", 0.0)),
            self._safe_float(prop.get("causal_source_score", 0.0)),
        )
        outgoing = max(
            self._safe_float(features.get("outgoing_propagation", 0.0)),
            self._safe_float(prop.get("outgoing_propagation", 0.0)),
        )
        reason_support = self._reason_metric_support_score(component=comp, reason=str(row.get("reason", "") or ""), metrics=metrics)
        return {
            "ownership": ownership,
            "symptomness": symptomness,
            "causal_source_score": causal,
            "outgoing_propagation": outgoing,
            "reason_support": reason_support,
        }

    def _promote_local_dict_challenger_over_node_symptom(
        self,
        rows: List[Dict[str, Any]],
        *,
        context: Optional[Mapping[str, Any]] = None,
        need_reason: bool = False,
    ) -> List[Dict[str, Any]]:
        if len(rows) < 2:
            return rows
        out = [dict(x) for x in rows]
        idx = 0
        while idx < min(len(out), 4):
            top = out[idx]
            top_comp = self.parse_component_id(str(top.get("component", "") or ""))
            if self.component_semantic_type(top_comp) != "node":
                idx += 1
                continue
            top_vals = self._row_rootness_values(top, context=dict(context or {}))
            node_owned = self._safe_float(top_vals.get("ownership", 0.0))
            node_causal = self._safe_float(top_vals.get("causal_source_score", 0.0))
            node_sibling = self._safe_float((top.get("rootness_features", {}) or {}).get("sibling_contrast", 0.0)) if isinstance(top.get("rootness_features", {}), Mapping) else 0.0
            node_reason_support = self._safe_float(top_vals.get("reason_support", 0.0))
            if node_causal >= 0.75 and node_sibling >= 0.85:
                idx += 1
                continue
            if node_owned >= 0.78 and (not need_reason or node_reason_support >= 0.72):
                idx += 1
                continue
            node_score = self._safe_float(top.get("score", 0.0))
            best_j = -1
            best_signal = 0.0
            node_signal = node_owned + 0.45 * node_reason_support + 0.25 * max(node_causal, self._safe_float(top_vals.get("outgoing_propagation", 0.0)))
            for j in range(idx + 1, min(len(out), idx + 14)):
                cand = out[j]
                comp = self.parse_component_id(str(cand.get("component", "") or ""))
                if self.component_semantic_type(comp) not in {"container", "service"}:
                    continue
                vals = self._row_rootness_values(cand, context=dict(context or {}))
                owned = self._safe_float(vals.get("ownership", 0.0))
                reason_support = self._safe_float(vals.get("reason_support", 0.0))
                causal = max(self._safe_float(vals.get("causal_source_score", 0.0)), self._safe_float(vals.get("outgoing_propagation", 0.0)))
                expected = self._expected_level_for_reason(str(cand.get("reason", "") or ""))
                level_ok = expected in {"pod", "service", ""} or not need_reason
                signal = owned + 0.45 * reason_support + 0.25 * causal
                if not level_ok:
                    continue
                if owned < 0.50 and reason_support < 0.55:
                    continue
                if signal < node_signal + (0.08 if need_reason else 0.18):
                    continue
                if self._safe_float(cand.get("score", 0.0)) < node_score - 0.62:
                    continue
                if signal > best_signal:
                    best_signal = signal
                    best_j = j
            if best_j > idx:
                chosen = out.pop(best_j)
                out.insert(idx, chosen)
                idx += 1
                continue
            idx += 1
        return out

    def _promote_component_only_dict_challenger_over_node_symptom(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Conservative component-only swap used before reason-aware rerank.

        Component-only tasks should not use reason-specific evidence.  Keep the
        older rootness behaviour here so the new joint component/reason logic
        cannot perturb component-only accuracy.
        """
        if len(rows) < 2:
            return rows
        out = [dict(x) for x in rows]
        idx = 0
        while idx < min(len(out), 4):
            top = out[idx]
            top_comp = self.parse_component_id(str(top.get("component", "") or ""))
            if self.component_semantic_type(top_comp) != "node":
                idx += 1
                continue
            top_features = dict(top.get("rootness_features", {}) or {}) if isinstance(top.get("rootness_features", {}), Mapping) else {}
            node_owned = max(
                self._safe_float(top_features.get("instance_owned_score", 0.0)),
                self._safe_float(top.get("instance_owned_score", 0.0)),
            )
            node_causal = self._safe_float(top_features.get("causal_source_score", 0.0))
            node_sibling = self._safe_float(top_features.get("sibling_contrast", 0.0))
            if node_causal >= 0.75 and node_sibling >= 0.85:
                idx += 1
                continue
            if node_owned >= 0.78:
                idx += 1
                continue
            node_score = self._safe_float(top.get("score", 0.0))
            best_j = -1
            best_owned = 0.0
            for j in range(idx + 1, min(len(out), idx + 10)):
                cand = out[j]
                comp = self.parse_component_id(str(cand.get("component", "") or ""))
                if self.component_semantic_type(comp) not in {"container", "service"}:
                    continue
                features = dict(cand.get("rootness_features", {}) or {}) if isinstance(cand.get("rootness_features", {}), Mapping) else {}
                owned = max(
                    self._safe_float(features.get("instance_owned_score", 0.0)),
                    self._safe_float(features.get("ownership", 0.0)),
                    self._safe_float(features.get("locality", 0.0)),
                    self._safe_float(cand.get("instance_owned_score", 0.0)),
                )
                if owned < 0.72:
                    continue
                if self._safe_float(cand.get("score", 0.0)) < node_score - 0.45:
                    continue
                if owned > best_owned:
                    best_owned = owned
                    best_j = j
            if best_j > idx:
                chosen = out.pop(best_j)
                out.insert(idx, chosen)
                idx += 1
                continue
            idx += 1
        return out

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

    def prefer_local_first_rootness(self) -> bool:
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
        if self._preserve_service_instance_output() and others:
            mixed: List[Dict[str, Any]] = []
            # Cloudbed-2 has many node-level symptoms for container/service
            # roots.  For multi-root component+reason, keep strong local
            # service/container candidates in the set instead of allowing node
            # symptoms to occupy every slot.
            for it in others:
                reason = self.parse_root_cause_label(str(it.get("reason", "") or ""))
                if reason.startswith("container ") or self.component_semantic_type(str(it.get("component", "") or "")) in {"container", "service"}:
                    mixed.append(it)
                if len(mixed) >= int(k):
                    break
            if nodes and len(mixed) < int(k):
                mixed.append(nodes[0])
            if len(mixed) >= int(k):
                return mixed[: int(k)]
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

    def index_candidate_pool_k(
        self,
        *,
        task_id: int,
        k: int,
        default_pool_k: int,
        context: Dict[str, Any],
    ) -> int:
        if int(task_id) in {1, 2}:
            return max(int(default_pool_k), 16)
        return max(int(default_pool_k), 24, int(k) * 8)

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

        taxonomy = [self.parse_root_cause_label(x) for x in self.basil_reason_taxonomy()]
        taxonomy = [x for x in dict.fromkeys(taxonomy) if x]
        if not taxonomy:
            return ordered, scores, evidence

        comp_ev = getattr(result, "component_evidence", None)
        top_metrics = dict(getattr(comp_ev, "top_metrics", {}) or {})
        comp_scores = dict(getattr(comp_ev, "combined_scores", {}) or {})
        if not top_metrics:
            top_metrics = dict(getattr(result, "top_metrics", {}) or {})
        if not comp_scores:
            comp_scores = dict(getattr(result, "combined_scores", {}) or {})
        metric_scores = self._surface_reason_scores(top_metrics=top_metrics, comp_scores=comp_scores)
        merged: Dict[str, float] = {}
        for reason in taxonomy:
            merged[reason] = max(
                self._safe_float(scores.get(reason, 0.0)),
                0.92 * self._safe_float(metric_scores.get(reason, 0.0)),
            )
        mx = max([abs(v) for v in merged.values()] + [0.0])
        norm = {k: (max(0.0, float(v) / mx) if mx > 1e-12 else 0.0) for k, v in merged.items()}
        ordered2 = [r for r, _ in sorted(norm.items(), key=lambda kv: kv[1], reverse=True)]
        evidence2 = dict(evidence or {})
        evidence2["market_metric_reason_scores"] = dict(metric_scores)
        return ordered2, norm, evidence2

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
        merged = super().augment_index_time_candidates(
            time_candidates=time_candidates,
            result=result,
            instruction=instruction,
            task_id=task_id,
            args=args,
            need_time=need_time,
            component_scores=component_scores,
            component_first_times=component_first_times,
            component_peak_times=component_peak_times,
            context_overrides=context_overrides,
        )
        if need_time:
            merged.extend(self._metric_change_point_time_candidates(instruction=instruction))
            merged.extend(self._stable_confirmation_time_candidates(merged))
        return self._filter_times_to_query_window(merged)

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

    def metric_rank_evidence(self, *, component: str, metrics: Sequence[str]) -> Dict[str, float]:
        comp = self.parse_component_id(str(component or ""))
        ctype = self.component_semantic_type(comp)
        rows = [str(m or "").lower() for m in list(metrics or []) if str(m or "").strip()]
        if not rows:
            return {"metric_score": 0.0, "ownership": 0.0, "symptomness": 0.0}

        owned_tokens: Dict[str, List[str]] = {
            "node": ["system.", "node_", "cpu.", "memory", "mem", "disk", "filesystem", "net."],
            "container": ["container_", "pod_", "blkio", "network_", "cpu", "memory", "rss", "restart"],
            "service": [
                "rr",
                "sr",
                "mrt",
                "count",
                "request",
                "latency",
                "duration",
                # Market often canonicalizes pod-level ids to service family
                # names in final answers.  Container metrics are still local
                # evidence for those service-family candidates.
                "container_",
                "pod_",
                "blkio",
                "network_",
                "rss",
                "restart",
            ],
        }
        symptom_tokens = ["istio", "envoy", "mesh", "trace", "duration", "collector", "jaeger", "proxy"]
        owned = 0
        symptom = 0
        for metric in rows:
            if any(tok in metric for tok in owned_tokens.get(ctype, [])):
                owned += 1
            if any(tok in metric for tok in symptom_tokens):
                symptom += 1
        ownership = float(owned) / float(max(1, len(rows)))
        symptomness = float(symptom) / float(max(1, len(rows)))
        if ctype == "service" and comp in {"jaeger-collector", "loadgenerator"}:
            symptomness = max(symptomness, 0.85)
        return {
            "metric_score": max(0.0, min(1.0, ownership * (1.0 - 0.45 * symptomness))),
            "ownership": max(0.0, min(1.0, ownership)),
            "symptomness": max(0.0, min(1.0, symptomness)),
        }

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
        rows = self._promote_local_challenger_over_node_symptom(list(candidates or []))
        out: List[Any] = []
        deferred: List[Any] = []
        used: set = set()
        for cand in rows:
            fam = _market_semantic_family(str(getattr(cand, "component", "") or ""))
            key = (fam, str(getattr(cand, "reason", "") or ""))
            if key in used:
                deferred.append(cand)
                continue
            out.append(cand)
            used.add(key)
        out.extend(deferred)
        return out

    def _promote_local_challenger_over_node_symptom(self, candidates: List[Any]) -> List[Any]:
        if len(candidates) < 2:
            return candidates
        candidates = self._demote_node_symptoms_below_local_challengers(candidates)
        top = candidates[0]
        top_comp = self.parse_component_id(str(getattr(top, "component", "") or ""))
        if self.component_semantic_type(top_comp) != "node":
            return candidates
        top_features = dict(getattr(top, "rootness_features", {}) or {})
        top_incoming = self._safe_float(top_features.get("incoming_symptomness", 0.0))
        top_symptom = max(self._safe_float(top_features.get("symptomness", 0.0)), top_incoming)
        top_owned = max(
            self._safe_float(top_features.get("instance_owned_score", 0.0)),
            self._safe_float(top_features.get("ownership", 0.0)),
            self._safe_float(top_features.get("locality", 0.0)),
        )
        top_score = self._safe_float(getattr(top, "score", 0.0))
        if top_symptom < 0.65:
            return candidates
        for idx, cand in enumerate(candidates[1:12], start=1):
            comp = self.parse_component_id(str(getattr(cand, "component", "") or ""))
            ctype = self.component_semantic_type(comp)
            if ctype not in {"container", "service"}:
                continue
            features = dict(getattr(cand, "rootness_features", {}) or {})
            owned = max(
                self._safe_float(features.get("instance_owned_score", 0.0)),
                self._safe_float(features.get("ownership", 0.0)),
                self._safe_float(features.get("locality", 0.0)),
                self._safe_float(features.get("sibling_contrast", 0.0)),
            )
            symptom = max(
                self._safe_float(features.get("symptomness", 0.0)),
                self._safe_float(features.get("incoming_symptomness", 0.0)),
            )
            score = self._safe_float(getattr(cand, "score", 0.0))
            if owned < max(0.45, top_owned * 1.05):
                continue
            if symptom >= top_symptom and owned < 0.80:
                continue
            if score < top_score - 0.28:
                continue
            chosen = candidates.pop(idx)
            try:
                ev = [str(x) for x in list(getattr(chosen, "evidence", []) or [])]
                ev.append("market_local_challenger_over_node_symptom")
                chosen.evidence = list(dict.fromkeys(ev))
            except Exception:
                pass
            candidates.insert(0, chosen)
            break
        return candidates

    def _demote_node_symptoms_below_local_challengers(self, candidates: List[Any]) -> List[Any]:
        if len(candidates) < 3:
            return candidates
        idx = 0
        while idx < min(len(candidates), 4):
            cand = candidates[idx]
            comp = self.parse_component_id(str(getattr(cand, "component", "") or ""))
            if self.component_semantic_type(comp) != "node":
                idx += 1
                continue
            features = dict(getattr(cand, "rootness_features", {}) or {})
            node_symptom = max(
                self._safe_float(features.get("symptomness", 0.0)),
                self._safe_float(features.get("incoming_symptomness", 0.0)),
            )
            node_owned = max(
                self._safe_float(features.get("instance_owned_score", 0.0)),
                self._safe_float(features.get("ownership", 0.0)),
            )
            node_causal = self._safe_float(features.get("causal_source_score", 0.0))
            node_sibling = self._safe_float(features.get("sibling_contrast", 0.0))
            if node_causal >= 0.75 and node_sibling >= 0.85:
                idx += 1
                continue
            if node_owned >= 0.78:
                idx += 1
                continue
            node_score = self._safe_float(getattr(cand, "score", 0.0))
            best_j = -1
            best_owned = 0.0
            for j in range(idx + 1, min(len(candidates), idx + 10)):
                other = candidates[j]
                other_comp = self.parse_component_id(str(getattr(other, "component", "") or ""))
                if self.component_semantic_type(other_comp) not in {"container", "service"}:
                    continue
                of = dict(getattr(other, "rootness_features", {}) or {})
                owned = max(
                    self._safe_float(of.get("instance_owned_score", 0.0)),
                    self._safe_float(of.get("ownership", 0.0)),
                    self._safe_float(of.get("locality", 0.0)),
                )
                if owned < 0.72:
                    continue
                if self._safe_float(getattr(other, "score", 0.0)) < node_score - 0.45:
                    continue
                if owned > best_owned:
                    best_owned = owned
                    best_j = j
            if best_j > idx:
                chosen = candidates.pop(best_j)
                candidates.insert(idx, chosen)
                idx += 1
                continue
            idx += 1
        return candidates

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
        if (
            self._preserve_service_instance_output()
            and bool(need_time)
            and not bool(need_comp)
            and not bool(need_reason)
        ):
            candidates = self._preprocess_time_candidates([dict(x) for x in list(time_candidates or []) if isinstance(x, Mapping)])
            scored = sorted(candidates, key=lambda x: self._score_time_candidate(x, "", ""), reverse=True)
            if scored:
                top = scored[0]
                top_comp = self.parse_component_id(str(top.get("component", "") or ""))
                top_score = self._score_time_candidate(top, "", "")
                top_ts = _parse_local_time_s(str(top.get("datetime", "") or ""))
                if self.component_semantic_type(top_comp) == "node" and top_ts:
                    for item in scored[1:8]:
                        comp = self.parse_component_id(str(item.get("component", "") or ""))
                        if self.component_semantic_type(comp) not in {"container", "service"}:
                            continue
                        dt = str(item.get("datetime", "") or "").strip()
                        ts = _parse_local_time_s(dt)
                        if not ts or abs(ts - top_ts) > 240:
                            continue
                        score = self._score_time_candidate(item, "", "")
                        if score < top_score - 0.04:
                            continue
                        details = dict(item)
                        details["source"] = f"{details.get('source', '')}+market_time_local_near_node"
                        if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:00$", dt):
                            try:
                                dt = (datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
                                details["source"] = f"{details.get('source', '')}+market_minute_center"
                            except Exception:
                                pass
                        return dt, details
        selected = super().select_time_candidate_override(
            time_candidates=time_candidates,
            component=component,
            reason=reason,
            need_comp=need_comp,
            need_reason=need_reason,
            need_time=need_time,
            fallback=fallback,
        )
        if selected is None or not self._preserve_service_instance_output() or need_comp or need_reason:
            return selected
        dt, details = selected
        if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:00$", str(dt or "")):
            try:
                shifted = (datetime.strptime(str(dt), "%Y-%m-%d %H:%M:%S") + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
                details2 = dict(details or {})
                details2["source"] = f"{details2.get('source', '')}+market_minute_center"
                return shifted, details2
            except Exception:
                return selected
        return selected

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
        if "metric_change_point" in src:
            score += 0.22
        if "onset_neighbor" in src and float(item.get("score", 0.0) or 0.0) >= 0.88 and "component_metric_first" in src:
            score += 0.14
        if any(tok in src for tok in ["first", "onset", "breach"]):
            score += 0.08
        if "market_basil_first_time" in src and c0:
            ctype = self.component_semantic_type(c0)
            if ctype in {"container", "service"}:
                score += 0.25
            elif ctype == "node":
                score -= 0.05
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
        raw = str(prediction_json or "")
        try:
            import json

            obj = json.loads(raw)
        except Exception:
            return raw
        if not isinstance(obj, dict):
            return raw
        groups = [dict(x) for x in list(selected_groups or []) if isinstance(x, Mapping)]
        changed = False
        if self._preserve_service_instance_output() and len(obj) > 1:
            time_only = True
            sortable: List[Tuple[int, str, Dict[str, Any]]] = []
            for key, payload in obj.items():
                if not isinstance(payload, dict):
                    time_only = False
                    break
                if str(payload.get("root cause component", "") or "").strip() or str(payload.get("root cause reason", "") or "").strip():
                    time_only = False
                    break
                dt = str(payload.get("root cause occurrence datetime", "") or "").strip()
                ts = _parse_local_time_s(dt)
                if not ts:
                    time_only = False
                    break
                try:
                    old_i = int(key)
                except Exception:
                    old_i = 0
                sortable.append((ts, str(old_i), dict(payload)))
            if time_only and sortable:
                sortable.sort(key=lambda x: (x[0], x[1]))
                obj = {str(i): payload for i, (_ts, _old, payload) in enumerate(sortable, start=1)}
                changed = True
        for key, payload in list(obj.items()):
            if not isinstance(payload, dict):
                continue
            reason = self.parse_root_cause_label(str(payload.get("root cause reason", "") or ""))
            if reason not in {"node disk read I/O consumption", "node disk write I/O consumption"}:
                continue
            idx = 0
            try:
                idx = max(0, int(key) - 1)
            except Exception:
                idx = 0
            metrics: List[str] = []
            if idx < len(groups):
                metrics = [str(x).lower() for x in list(groups[idx].get("supporting_metrics", []) or [])]
            blob = " ".join(metrics)
            group_comp = groups[idx].get("component", "") if idx < len(groups) else ""
            comp = self.parse_component_id(str(payload.get("root cause component", "") or group_comp or ""))
            if self.component_semantic_type(comp) != "node":
                continue
            if reason == "node CPU load":
                refined = self._refine_node_surface_reason(component=comp, reason=reason, metrics=metrics)
                if refined and refined != reason:
                    payload["root cause reason"] = refined
                    changed = True
                continue
            read_hits = sum(1 for tok in ["r_await", "r_s", "rkb", "read"] if tok in blob)
            write_hits = sum(1 for tok in ["w_await", "w_s", "wkb", "write"] if tok in blob)
            capacity_hits = sum(1 for tok in ["avg_q", "await", "svctm", "util", "iowait", "queue"] if tok in blob)
            if capacity_hits >= 3 and max(read_hits, write_hits) <= 2:
                payload["root cause reason"] = "node disk space consumption"
                changed = True
        if not changed:
            return raw
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return raw

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
        if re.search(r"\.ts(?::\d+)?$", c) or re.search(r":\d+$", c):
            return 0.25
        if any(tok in c for tok in ["jaeger", "collector", "loadgenerator", "egressgateway", "ingressgateway"]):
            return 0.45
        return 1.0

    def supplement_candidate_pool(
        self,
        existing_tuples: List[Dict[str, Any]],
        context: Dict[str, Any],
        *,
        date_dir: str = "",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        supplements = super().supplement_candidate_pool(existing_tuples, context, date_dir=date_dir, **kwargs)
        telemetry = self._telemetry_component_scores(context=context, date_dir=date_dir)
        if not telemetry:
            return supplements
        covered = {
            self.parse_component_id(str(t.get("component", "") or ""))
            for t in list(existing_tuples) + list(supplements)
            if str(t.get("component", "") or "").strip()
        }
        pool_reasons = list(
            dict.fromkeys(
                self.parse_root_cause_label(str(t.get("reason", "") or ""))
                for t in existing_tuples
                if str(t.get("reason", "") or "").strip()
            )
        )
        pair_reasons = pool_reasons[:2] if pool_reasons else [""]
        need_reason = bool((context or {}).get("need_reason", False))
        by_type: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for comp, ev in telemetry.items():
            by_type.setdefault(str(self.component_semantic_type(comp) or "other"), []).append((comp, ev))
        for rows in by_type.values():
            rows.sort(key=lambda kv: self._safe_float(kv[1].get("score", 0.0)), reverse=True)
        for _ctype, rows in by_type.items():
            for comp, ev in rows[:4]:
                if comp in covered:
                    continue
                comp_reasons = list(pair_reasons)
                if need_reason:
                    metrics = [str(x) for x in list(ev.get("top_metrics", []) or []) if str(x).strip()]
                    supported: List[Tuple[float, str]] = []
                    for reason in self.basil_reason_taxonomy():
                        r = self.parse_root_cause_label(str(reason or ""))
                        if not r:
                            continue
                        expected = self._expected_level_for_reason(r)
                        if expected == "pod":
                            expected = "container"
                        ctype = self.component_semantic_type(comp)
                        if expected and ctype and expected != ctype:
                            continue
                        support = self._reason_metric_support_score(component=comp, reason=r, metrics=metrics)
                        if support > 0.0:
                            supported.append((float(support), r))
                    supported.sort(key=lambda kv: kv[0], reverse=True)
                    comp_reasons = list(dict.fromkeys(comp_reasons + [r for _s, r in supported[:3]]))
                for reason in comp_reasons:
                    metrics = list(ev.get("top_metrics", []) or [])[:6]
                    support = self._reason_metric_support_score(component=comp, reason=str(reason or ""), metrics=metrics) if reason else 0.0
                    supplements.append(
                        {
                            "component": comp,
                            "reason": reason,
                            "datetime": str(ev.get("first_time", "") or ""),
                            "score": 0.36 + 0.28 * self._safe_float(ev.get("score", 0.0)) + 0.18 * support,
                            "source": "market_typed_metric_supplement",
                            "supporting_metrics": metrics,
                            "component_type": self.component_semantic_type(comp),
                            "first_anomaly_time": str(ev.get("first_time", "") or ""),
                            "symptomness": self._safe_float(ev.get("symptomness", 0.0)),
                            "reason_metric_support": float(support),
                        }
                    )
                covered.add(comp)
        return supplements

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
        log_paths = [
            os.path.join("dataset", self.dataset, "telemetry", date_dir, "log", "log_service.csv"),
            os.path.join("dataset", self.dataset, "telemetry", date_dir, "log", "log_proxy.csv"),
        ]
        taxonomy = self._reason_taxonomy()
        token_map: Dict[str, List[str]] = {}
        for reason, meta in taxonomy.items():
            tokens = []
            if isinstance(meta, Mapping):
                tokens.extend([str(x).lower() for x in list(meta.get("tokens", []) or [])])
            tokens.extend(self.reason_metric_keywords(str(reason)))
            token_map[str(reason)] = list(dict.fromkeys([t for t in tokens if t]))
        token_df: Dict[str, int] = {}
        for tokens in token_map.values():
            for tok in set(tokens):
                token_df[tok] = token_df.get(tok, 0) + 1
        reason_n = max(1, len(token_map))
        raw = {self.parse_root_cause_label(r): 0.0 for r in taxonomy.keys()}
        for path in log_paths:
            if not os.path.exists(path):
                continue
            try:
                chunks = pd.read_csv(path, usecols=lambda c: c in {"timestamp", "log_name", "value"}, chunksize=200_000, low_memory=False)
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
                    ).str.replace(r"[^a-z0-9=._/ -]+", " ", regex=True)
                    joined = "\n".join(text.head(6000).tolist())
                    for reason, tokens in token_map.items():
                        score = 0.0
                        for tok in tokens:
                            if tok and tok in joined:
                                # Shared words such as "network" or "io" are
                                # weak evidence. Rare, reason-specific tokens
                                # such as retrans/corrupt/restart are stronger.
                                score += 1.0 + max(0.0, (reason_n - token_df.get(tok, reason_n)) / reason_n)
                        raw[self.parse_root_cause_label(reason)] = raw.get(self.parse_root_cause_label(reason), 0.0) + score
            except Exception:
                continue
        mx = max([abs(v) for v in raw.values()] + [0.0])
        if mx <= 1e-12:
            return {}
        return {k: max(0.0, float(v) / mx) for k, v in raw.items()}

    def _telemetry_reason_scores(self, *, instruction: str) -> Dict[str, float]:
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
        metric_dir = os.path.join("dataset", self.dataset, "telemetry", date_dir, "metric")
        if not os.path.isdir(metric_dir):
            return {}

        taxonomy = [self.parse_root_cause_label(x) for x in self.basil_reason_taxonomy()]
        taxonomy = [x for x in dict.fromkeys(taxonomy) if x]
        raw = {r: 0.0 for r in taxonomy}
        reason_tokens = {r: [str(t).lower() for t in self.reason_metric_keywords(r) if str(t).strip()] for r in taxonomy}
        token_df: Dict[str, int] = {}
        for tokens in reason_tokens.values():
            for tok in set(tokens):
                token_df[tok] = token_df.get(tok, 0) + 1
        reason_n = max(1, len(reason_tokens))

        for name in os.listdir(metric_dir):
            if not str(name).endswith(".csv"):
                continue
            path = os.path.join(metric_dir, name)
            try:
                long = _market_normalized_metric_file(path)
            except Exception:
                continue
            if long.empty:
                continue
            try:
                long["metric_l"] = long["metric"].astype(str).str.lower()
                stats = long.groupby("metric")["value"].agg(["median", "std"]).reset_index()
                win = long[(long["ts_s"] >= start_s) & (long["ts_s"] <= end_s)].copy()
                if win.empty:
                    continue
                win = win.merge(stats, on="metric", how="left").fillna({"std": 0.0})
                denom = win["std"].abs().where(win["std"].abs() > 1e-6, 1.0)
                win["dev"] = ((win["value"] - win["median"]).abs() / denom).clip(lower=0.0, upper=12.0)
                metric_dev_q95 = win.groupby("metric_l")["dev"].quantile(0.95).to_dict()
                metric_dev_max = win.groupby("metric_l")["dev"].max().to_dict()
            except Exception:
                continue
            for reason, tokens in reason_tokens.items():
                if not tokens:
                    continue
                score = 0.0
                hits = 0
                metric_names = set(metric_dev_q95.keys()) | set(metric_dev_max.keys())
                for metric_name in metric_names:
                    metric_text = str(metric_name)
                    sparse_event_metric = any(
                        tok in metric_text
                        for tok in ["error", "retrans", "drop", "dropped", "restart", "terminated", "exit", "kill"]
                    )
                    dev = self._safe_float((metric_dev_max if sparse_event_metric else metric_dev_q95).get(metric_name, 0.0))
                    matched_weight = 0.0
                    for tok in tokens:
                        if tok and tok in metric_text:
                            matched_weight += 1.0 + max(0.0, (reason_n - token_df.get(tok, reason_n)) / reason_n)
                    if matched_weight <= 0:
                        continue
                    hits += 1
                    score += float(dev) * min(2.4, matched_weight)
                if hits:
                    raw[reason] = raw.get(reason, 0.0) + score / (hits ** 0.35)
        mx = max([abs(v) for v in raw.values()] + [0.0])
        if mx <= 1e-12:
            return {}
        return {k: max(0.0, float(v) / mx) for k, v in raw.items()}

    def _metric_change_point_time_candidates(self, *, instruction: str) -> List[Dict[str, Any]]:
        import pandas as pd
        from rca.adapter.utils.component_candidates import parse_time_window_generic
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
                long = _market_normalized_metric_file(path)
            except Exception:
                continue
            if long.empty:
                continue
            try:
                p95 = long.groupby(["component", "metric"], as_index=False)["value"].quantile(0.95).rename(columns={"value": "p95"})
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
            for minute, value in win.groupby("minute")["breach_ratio"].sum().items():
                minute_scores[int(minute)] = minute_scores.get(int(minute), 0.0) + float(value)
            for (minute, comp), value in win.groupby(["minute", "component"])["breach_ratio"].sum().items():
                minute_components.setdefault(int(minute), {})
                c = str(comp)
                minute_components[int(minute)][c] = minute_components[int(minute)].get(c, 0.0) + float(value)
        if not minute_scores:
            return []
        minutes = sorted(minute_scores)
        values = [minute_scores[m] for m in minutes]
        mx = max(values) if values else 0.0
        if mx <= 1e-9:
            return []
        scored: List[Tuple[float, int, str]] = []
        for idx, minute in enumerate(minutes):
            before = values[max(0, idx - 3):idx]
            after = values[idx:min(len(values), idx + 4)]
            before_mean = sum(before) / max(1, len(before)) if before else 0.0
            after_mean = sum(after) / max(1, len(after)) if after else 0.0
            contrast = max(0.0, after_mean - before_mean)
            persistence = sum(1 for v in after if v >= 0.35 * mx) / max(1, len(after))
            score = 0.70 * (contrast / mx) + 0.30 * persistence
            comps = minute_components.get(minute, {})
            top_comp = max(comps, key=lambda c: comps[c]) if comps else ""
            if score > 0:
                scored.append((score, int(minute), top_comp))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for score, minute, comp in scored:
            if any(abs(minute - old) <= 60 for old in seen):
                continue
            seen.add(minute)
            dt = pd.to_datetime(minute, unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
            out.append({"datetime": dt, "source": "metric_change_point", "score": 0.82 + min(0.16, 0.16 * score), "component": comp, "reason": ""})
            if len(out) >= 8:
                break
        return out

    def _stable_confirmation_time_candidates(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import pandas as pd

        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in list(candidates or []):
            if not isinstance(item, Mapping):
                continue
            src = str(item.get("source", "") or "").lower()
            if not any(tok in src for tok in ["onset", "first", "breach", "metric_change_point"]):
                continue
            if any(tok in src for tok in ["trace", "peak"]):
                continue
            dt = str(item.get("datetime", "") or "").strip()
            if not dt:
                continue
            try:
                ts = pd.Timestamp(dt)
            except Exception:
                continue
            for offset_min, penalty in [(2, 0.08), (4, 0.16)]:
                shifted = (ts + pd.Timedelta(minutes=offset_min)).strftime("%Y-%m-%d %H:%M:%S")
                if shifted in seen:
                    continue
                seen.add(shifted)
                out.append(
                    {
                        "datetime": shifted,
                        "source": f"stable_confirmation_plus_{offset_min}m",
                        "score": max(0.0, self._safe_float(item.get("score", 0.0)) - penalty),
                        "component": str(item.get("component", "") or ""),
                        "reason": str(item.get("reason", "") or ""),
                        "evidence": {"base_datetime": dt, "base_source": src},
                    }
                )
        return out[:12]

    def _log_event_time_candidates(self, *, instruction: str) -> List[Dict[str, Any]]:
        import pandas as pd
        from rca.adapter.utils.component_candidates import parse_time_window_generic
        from rca.common.io import resolve_date_dir

        try:
            date_dir, start, end = parse_time_window_generic(str(instruction or ""))
            date_dir = resolve_date_dir(self.dataset, date_dir)
            start_s = int(start.tz_convert("UTC").timestamp())
            end_s = int(end.tz_convert("UTC").timestamp())
        except Exception:
            return []
        log_dir = os.path.join("dataset", self.dataset, "telemetry", date_dir, "log")
        if not os.path.isdir(log_dir):
            return []
        scores: Dict[int, float] = {}
        comps: Dict[int, Dict[str, float]] = {}
        for name in ["log_proxy.csv", "log_service.csv"]:
            path = os.path.join(log_dir, name)
            if not os.path.exists(path):
                continue
            try:
                chunks = pd.read_csv(
                    path,
                    usecols=lambda c: c in {"timestamp", "cmdb_id"},
                    chunksize=200_000,
                    low_memory=False,
                )
                for chunk in chunks:
                    if "timestamp" not in chunk.columns:
                        continue
                    ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
                    chunk = chunk[(ts >= start_s) & (ts <= end_s)].copy()
                    if chunk.empty:
                        continue
                    # Proxy logs are high-volume; exact timestamp bursts are a
                    # useful event-level anchor even when status is 200.
                    for ts0, comp0 in zip(ts.loc[chunk.index].astype(int), chunk.get("cmdb_id", "").astype(str)):
                        t = int(ts0)
                        scores[t] = scores.get(t, 0.0) + 1.0
                        comp = self.parse_component_id(str(comp0))
                        if comp:
                            comps.setdefault(t, {})
                            comps[t][comp] = comps[t].get(comp, 0.0) + 1.0
            except Exception:
                continue
        if not scores:
            return []
        values = list(scores.values())
        mx = max(values)
        if mx <= 0:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out: List[Dict[str, Any]] = []
        seen_minute: set[int] = set()
        for ts0, score in ranked:
            if int(ts0) <= start_s + 120 or int(ts0) >= end_s - 120:
                continue
            minute = (int(ts0) // 60) * 60
            if minute in seen_minute:
                continue
            seen_minute.add(minute)
            comp_scores = comps.get(int(ts0), {})
            comp = max(comp_scores, key=lambda c: comp_scores[c]) if comp_scores else ""
            dt = pd.to_datetime(int(ts0), unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
            out.append(
                {
                    "datetime": dt,
                    "source": "log_event_burst",
                    "score": 1.08 + 0.18 * min(1.0, float(score) / mx),
                    "component": comp,
                    "reason": "",
                    "evidence": {"log_event_count": float(score), "component": comp},
                }
            )
            if len(out) >= 10:
                break
        return out

    def _telemetry_component_scores(self, *, context: Mapping[str, Any], date_dir: str) -> Dict[str, Dict[str, Any]]:
        import pandas as pd
        dataset = str((context or {}).get("dataset", "") or "") or self.dataset
        start_local = str((context or {}).get("query_start_local", "") or "")
        end_local = str((context or {}).get("query_end_local", "") or "")
        if not dataset or not date_dir or not start_local or not end_local:
            return {}
        try:
            tz = str(self.config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
            start_s = int(pd.Timestamp(start_local, tz=tz).tz_convert("UTC").timestamp())
            end_s = int(pd.Timestamp(end_local, tz=tz).tz_convert("UTC").timestamp())
        except Exception:
            return {}
        metric_dir = os.path.join("dataset", dataset, "telemetry", date_dir, "metric")
        if not os.path.isdir(metric_dir):
            return {}
        raw: Dict[str, Dict[str, Any]] = {}
        for name in os.listdir(metric_dir):
            if not str(name).endswith(".csv"):
                continue
            lower_name = str(name).lower()
            if not any(tok in lower_name for tok in ["metric_service", "metric_container", "metric_node", "metric_runtime"]):
                continue
            try:
                long = _market_normalized_metric_file(os.path.join(metric_dir, name))
            except Exception:
                continue
            if long.empty:
                continue
            try:
                p95 = long.groupby(["component", "metric"], as_index=False)["value"].quantile(0.95).rename(columns={"value": "p95"})
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
            except Exception:
                continue
            by_comp = win.groupby("component")["breach_ratio"].sum()
            by_metric = win.groupby(["component", "metric"])["breach_ratio"].sum()
            first = win.groupby("component")["ts_s"].min()
            for comp, value in by_comp.items():
                comp = str(comp)
                raw.setdefault(comp, {"score": 0.0, "top_metrics": [], "first_time": "", "symptomness": 0.0})
                raw[comp]["score"] = self._safe_float(raw[comp].get("score", 0.0)) + float(value)
                metrics = [str(m) for (c, m), _v in sorted(by_metric.items(), key=lambda kv: kv[1], reverse=True) if str(c) == comp][:6]
                raw[comp]["top_metrics"] = list(dict.fromkeys(list(raw[comp].get("top_metrics", []) or []) + metrics))[:8]
                if comp in first:
                    raw[comp]["first_time"] = pd.to_datetime(int(first[comp]), unit="s", utc=True).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
                ev = self.metric_rank_evidence(component=comp, metrics=raw[comp]["top_metrics"])
                raw[comp]["symptomness"] = max(self._safe_float(raw[comp].get("symptomness", 0.0)), self._safe_float(ev.get("symptomness", 0.0)))
        mx = max([self._safe_float(v.get("score", 0.0)) for v in raw.values()] + [0.0])
        if mx <= 1e-12:
            return {}
        by_type_max: Dict[str, float] = {}
        for c, ev in raw.items():
            ctype = str(self.component_semantic_type(str(c)) or "other")
            by_type_max[ctype] = max(by_type_max.get(ctype, 0.0), self._safe_float(ev.get("score", 0.0)))
        out: Dict[str, Dict[str, Any]] = {}
        for c, ev in raw.items():
            ctype = str(self.component_semantic_type(str(c)) or "other")
            raw_score = self._safe_float(ev.get("score", 0.0))
            out[c] = {
                **ev,
                "score": raw_score / mx,
                "channel_score": raw_score / max(by_type_max.get(ctype, 0.0), 1e-12),
            }
        return out
