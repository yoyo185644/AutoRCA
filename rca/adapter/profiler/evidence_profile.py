from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence


def _normalize(counter: Mapping[str, float]) -> Dict[str, float]:
    max_value = max([float(v) for v in counter.values()] + [0.0])
    if max_value <= 0:
        return {}
    return {str(k): float(v) / max_value for k, v in counter.items() if str(k).strip()}


@dataclass(frozen=True)
class DatasetEvidenceProfile:
    """Telemetry-only profile for generic evidence scoring."""

    component_prior: Dict[str, float] = field(default_factory=dict)
    reason_prior: Dict[str, float] = field(default_factory=dict)
    component_type_prior: Dict[str, float] = field(default_factory=dict)
    component_reason_prior: Dict[str, float] = field(default_factory=dict)
    type_reason_prior: Dict[str, float] = field(default_factory=dict)
    channel_prior: Dict[str, float] = field(default_factory=dict)
    metric_idf: Dict[str, float] = field(default_factory=dict)

    @property
    def profile_strength(self) -> float:
        # This profile is intentionally weak because it is unsupervised and
        # built without query labels or historical root-cause records.
        return 0.15 if self.metric_idf else 0.0

    def compatibility_score(self, *, component: str, component_type: str, reason: str) -> float:
        values = []
        comp = str(component or "")
        ctype = str(component_type or "")
        reason = str(reason or "")
        if comp:
            values.append(self.component_prior.get(comp, 0.0))
        if ctype:
            values.append(self.component_type_prior.get(ctype, 0.0))
            values.append(self.channel_prior.get(ctype, 0.0))
        if reason:
            values.append(self.reason_prior.get(reason, 0.0))
        if comp and reason:
            values.append(self.component_reason_prior.get(f"{comp}\t{reason}", 0.0))
        if ctype and reason:
            values.append(self.type_reason_prior.get(f"{ctype}\t{reason}", 0.0))
        values = [float(v) for v in values if float(v) > 0]
        if not values:
            return 0.0
        return self.profile_strength * float(sum(values) / len(values))

    def metric_specificity_score(self, metrics: Sequence[str]) -> float:
        values = [float(self.metric_idf.get(str(m), 0.0)) for m in list(metrics or []) if str(m).strip()]
        values = [v for v in values if v > 0]
        if not values:
            return 0.0
        return float(sum(values) / len(values))


def build_telemetry_evidence_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    component_type_lookup,
) -> DatasetEvidenceProfile:
    component = Counter()
    reason = Counter()
    component_type = Counter()
    component_reason = Counter()
    type_reason = Counter()

    for row in list(rows or []):
        comp = str(row.get("component", "") or "").strip()
        rsn = str(row.get("reason", "") or "").strip()
        ctype = str(component_type_lookup(comp) or "").strip()
        if comp:
            component[comp] += 1
        if rsn:
            reason[rsn] += 1
        if ctype:
            component_type[ctype] += 1
        if comp and rsn:
            component_reason[f"{comp}\t{rsn}"] += 1
        if ctype and rsn:
            type_reason[f"{ctype}\t{rsn}"] += 1

    return DatasetEvidenceProfile(
        component_prior=_normalize(component),
        reason_prior=_normalize(reason),
        component_type_prior=_normalize(component_type),
        component_reason_prior=_normalize(component_reason),
        type_reason_prior=_normalize(type_reason),
        channel_prior=_normalize(component_type),
    )


def metric_idf_from_top_metrics(top_metrics: Mapping[str, Sequence[str]]) -> Dict[str, float]:
    component_count = len([c for c in top_metrics.keys() if str(c).strip()])
    if component_count <= 0:
        return {}
    df = Counter()
    for metrics in top_metrics.values():
        seen = {str(m) for m in list(metrics or []) if str(m).strip()}
        for metric in seen:
            df[metric] += 1
    values: Dict[str, float] = {}
    denom = math.log1p(float(component_count))
    for metric, count in df.items():
        values[str(metric)] = math.log1p(float(component_count) / max(1.0, float(count))) / max(denom, 1e-9)
    return values
