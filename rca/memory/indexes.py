from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rca.memory.schemas import (
    AnomalySegment,
    BlockMaxBlock,
    CausalEdgeSketch,
    TraceTopologyEdge,
    component_type,
    lag_bin_name,
)


REASON_TOKENS: Dict[str, List[str]] = {
    "network loss": ["loss", "drop", "retrans", "timeout", "reset", "error"],
    "network delay": ["tnsping", "ping", "rtt", "latency", "delay"],
    "db connection limit": ["login", "session", "sess", "tps", "exec", "connection", "conn"],
    "db close": ["close", "disconnect", "logout", "on_off", "shutdown", "restart"],
    "CPU fault": ["cpu", "load", "iowait", "proc", "util"],
    "high memory usage": ["mem", "memory", "rss", "heap", "oom"],
    "container memory load": ["container", "mem", "memory", "rss", "heap", "oom"],
    "node memory consumption": ["node", "mem", "memory", "rss", "heap", "oom"],
    "node CPU load": ["node", "cpu", "load", "util"],
    "container CPU load": ["container", "cpu", "load", "util"],
}


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    mx = max([abs(float(v)) for v in scores.values()] + [0.0])
    if mx <= 1e-12:
        return {str(k): 0.0 for k in scores.keys()}
    return {str(k): max(0.0, float(v) / mx) for k, v in scores.items()}


def build_block_max(scores: Dict[str, float], block_size: int = 32) -> List[BlockMaxBlock]:
    items = sorted(((str(k), float(v)) for k, v in scores.items()), key=lambda kv: kv[1], reverse=True)
    size = max(1, int(block_size))
    blocks: List[BlockMaxBlock] = []
    for i in range(0, len(items), size):
        chunk = items[i : i + size]
        if not chunk:
            continue
        blocks.append(BlockMaxBlock(block_id=len(blocks), members=[k for k, _ in chunk], ub=max(v for _, v in chunk)))
    return blocks


class AnomalySegmentIndex:
    def __init__(self, segments: Sequence[AnomalySegment], bucket_size_s: int = 300, block_size: int = 32) -> None:
        self.segments = list(segments or [])
        self.bucket_size_s = max(1, int(bucket_size_s))
        self.block_size = max(1, int(block_size))
        self.by_component: Dict[str, List[AnomalySegment]] = defaultdict(list)
        self.by_bucket: Dict[int, List[AnomalySegment]] = defaultdict(list)
        for seg in self.segments:
            self.by_component[seg.component].append(seg)
            b0 = int(seg.start_s) // self.bucket_size_s
            b1 = int(seg.end_s) // self.bucket_size_s
            for b in range(b0, b1 + 1):
                self.by_bucket[b].append(seg)

    def components(self) -> List[str]:
        return sorted(self.by_component.keys())

    def overlapping_segments(self, start_s: int, end_s: int) -> List[AnomalySegment]:
        b0 = int(start_s) // self.bucket_size_s
        b1 = int(end_s) // self.bucket_size_s
        seen = set()
        out: List[AnomalySegment] = []
        for b in range(b0, b1 + 1):
            for seg in self.by_bucket.get(b, []):
                if seg.segment_id in seen or not seg.overlaps(start_s, end_s):
                    continue
                seen.add(seg.segment_id)
                out.append(seg)
        return out

    def local_scores(self, start_s: int, end_s: int) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        for seg in self.overlapping_segments(start_s, end_s):
            raw[seg.component] = max(raw.get(seg.component, 0.0), float(seg.severity or 0.0))
        return normalize_scores(raw)

    def metric_scores(self, start_s: int, end_s: int) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        for seg in self.overlapping_segments(start_s, end_s):
            if seg.signal_type == "metric":
                raw[seg.component] = max(raw.get(seg.component, 0.0), float(seg.severity or 0.0))
        return raw

    def trace_scores(self, start_s: int, end_s: int) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        for seg in self.overlapping_segments(start_s, end_s):
            if seg.signal_type == "trace":
                raw[seg.component] = max(raw.get(seg.component, 0.0), float(seg.severity or 0.0))
        return raw

    def reason_scores(self, reasons: Sequence[str], start_s: int, end_s: int) -> Dict[str, float]:
        reason_list = [str(r) for r in reasons or [] if str(r).strip()]
        if not reason_list:
            return {}
        raw: Dict[str, float] = {}
        for seg in self.overlapping_segments(start_s, end_s):
            metrics = " ".join([str(x).lower() for x in list(seg.top_metrics or []) + [seg.signal_name]])
            hints = set([str(x) for x in seg.reason_hints or []])
            best = 0.0
            for r in reason_list:
                score = 1.0 if r in hints else 0.0
                toks = REASON_TOKENS.get(str(r), [])
                if toks:
                    hit = sum(1 for t in toks if t and t.lower() in metrics)
                    score = max(score, min(1.0, hit / max(1.0, min(3.0, float(len(toks))))))
                if component_type(seg.component) in ("db", "service") and "db" in str(r).lower():
                    score = max(score, 0.25)
                if component_type(seg.component) in ("node", "container") and any(k in str(r).lower() for k in ["cpu", "memory", "network"]):
                    score = max(score, 0.20)
                best = max(best, score)
            if best > 0:
                raw[seg.component] = max(raw.get(seg.component, 0.0), best * max(0.25, min(1.0, float(seg.severity or 0.0))))
        return normalize_scores(raw)


class CausalPropagationIndex:
    def __init__(
        self,
        segments: Sequence[AnomalySegment],
        topology_edges: Sequence[TraceTopologyEdge],
        max_lag_s: int = 900,
        block_size: int = 32,
        support_tau: float = 3.0,
    ) -> None:
        self.segments = list(segments or [])
        self.topology_edges = list(topology_edges or [])
        self.max_lag_s = max(0, int(max_lag_s))
        self.block_size = max(1, int(block_size))
        self.support_tau = max(1e-9, float(support_tau))
        self.edge_sketches: List[CausalEdgeSketch] = self._build_edge_sketches()

    def _segment_norm(self) -> Dict[str, float]:
        raw = {seg.segment_id: float(seg.severity or 0.0) for seg in self.segments}
        return normalize_scores(raw)

    def _fallback_topology(self) -> List[TraceTopologyEdge]:
        by_comp: Dict[str, AnomalySegment] = {}
        for seg in sorted(self.segments, key=lambda s: float(s.severity or 0.0), reverse=True):
            if seg.component not in by_comp:
                by_comp[seg.component] = seg
            if len(by_comp) >= 80:
                break
        comps = list(by_comp.values())
        out: List[TraceTopologyEdge] = []
        for a in comps:
            for b in comps:
                if a.component == b.component:
                    continue
                lag = int(b.start_s) - int(a.start_s)
                if 0 <= lag <= self.max_lag_s:
                    out.append(TraceTopologyEdge(src=a.component, dst=b.component, edge_type="temporal_cooccurrence", support_count=1))
        return out

    def _build_edge_sketches(self) -> List[CausalEdgeSketch]:
        seg_by_comp: Dict[str, List[AnomalySegment]] = defaultdict(list)
        for seg in self.segments:
            seg_by_comp[seg.component].append(seg)
        norm = self._segment_norm()
        edges = self.topology_edges if self.topology_edges else self._fallback_topology()
        grouped: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
        support_bonus: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
        for edge in edges:
            src_segs = seg_by_comp.get(edge.src, [])
            dst_segs = seg_by_comp.get(edge.dst, [])
            if not src_segs or not dst_segs:
                continue
            for s in src_segs:
                for d in dst_segs:
                    lag = int(d.start_s) - int(s.start_s)
                    if lag < 0 or lag > self.max_lag_s:
                        continue
                    lb = lag_bin_name(lag)
                    key = (edge.src, edge.dst, edge.edge_type, lb)
                    raw = math.sqrt(max(0.0, norm.get(s.segment_id, 0.0)) * max(0.0, norm.get(d.segment_id, 0.0)))
                    grouped[key].append(float(raw))
                    support_bonus[key] += int(edge.support_count or 1)

        sketches: List[CausalEdgeSketch] = []
        for (src, dst, edge_type, lb), vals in grouped.items():
            n = max(1, len(vals))
            support = n + max(0, support_bonus.get((src, dst, edge_type, lb), 0) - 1)
            conf = 1.0 - math.exp(-float(support) / self.support_tau)
            transfer_max = max(vals) if vals else 0.0
            lift_max = min(1.0, transfer_max * (1.0 + math.log1p(float(support))))
            score = max(0.0, min(1.0, transfer_max * conf))
            sketches.append(
                CausalEdgeSketch(
                    src=src,
                    dst=dst,
                    edge_type=edge_type,
                    lag_bin=lb,
                    support_count=int(support),
                    transfer_max=float(transfer_max),
                    lift_max=float(lift_max),
                    score=float(score),
                )
            )
        sketches.sort(key=lambda e: e.score, reverse=True)
        return sketches

    def causal_scores(self, affected_components: Optional[Iterable[str]] = None) -> Dict[str, float]:
        affected = set([str(c) for c in affected_components or [] if str(c).strip()])
        raw: Dict[str, Dict[str, float]] = defaultdict(dict)
        for e in self.edge_sketches:
            if affected and e.dst not in affected:
                continue
            raw[e.src][e.dst] = max(raw[e.src].get(e.dst, 0.0), float(e.score or 0.0))
        summed = {src: sum(dst_scores.values()) for src, dst_scores in raw.items()}
        return normalize_scores(summed)

    def blocks(self, scores: Dict[str, float]) -> List[BlockMaxBlock]:
        return build_block_max(scores, block_size=self.block_size)
