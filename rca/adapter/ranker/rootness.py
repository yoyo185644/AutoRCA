from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, MutableMapping, Tuple


def _clip01(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass(frozen=True)
class RootnessFeatures:
    global_score: float = 0.0
    channel_score: float = 0.0
    channel_rank_score: float = 0.0
    local_change: float = 0.0
    ownership: float = 0.0
    locality: float = 0.0
    persistence: float = 0.0
    instance_owned_score: float = 0.0
    sibling_contrast: float = 0.0
    symptomness: float = 0.0
    causal_source_score: float = 0.0
    outgoing_propagation: float = 0.0
    incoming_symptomness: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RootnessFeatures":
        return cls(
            global_score=_clip01(values.get("global_score", 0.0)),
            channel_score=_clip01(values.get("channel_score", 0.0)),
            channel_rank_score=_clip01(values.get("channel_rank_score", 0.0)),
            local_change=_clip01(values.get("local_change", 0.0)),
            ownership=_clip01(values.get("ownership", 0.0)),
            locality=_clip01(values.get("locality", 0.0)),
            persistence=_clip01(values.get("persistence", 0.0)),
            instance_owned_score=_clip01(values.get("instance_owned_score", 0.0)),
            sibling_contrast=_clip01(values.get("sibling_contrast", 0.0)),
            symptomness=_clip01(values.get("symptomness", 0.0)),
            causal_source_score=_clip01(values.get("causal_source_score", 0.0)),
            outgoing_propagation=_clip01(values.get("outgoing_propagation", 0.0)),
            incoming_symptomness=_clip01(values.get("incoming_symptomness", 0.0)),
        )


class RootnessRanker:
    """Telemetry-only rootness scoring shared by dataset adapters.

    The ranker intentionally uses bounded, interpretable feature groups rather
    than dataset-specific constants.  Dataset adapters provide schema-level
    ownership and canonicalization; this class combines evidence uniformly.
    """

    def score(self, features: RootnessFeatures) -> float:
        f = features
        structural = (
            0.34 * f.global_score
            + 0.28 * f.channel_score
            + 0.08 * f.channel_rank_score
        )
        local = (
            0.14 * f.local_change
            + 0.08 * f.ownership
            + 0.05 * f.locality
            + 0.03 * f.persistence
            + 0.05 * f.instance_owned_score
            + 0.03 * f.sibling_contrast
        )
        propagation = (
            0.16 * f.causal_source_score
            + 0.12 * f.outgoing_propagation
        )
        score = structural + local + propagation
        local_support = max(f.ownership, f.locality, 0.5 * f.local_change)
        effective_incoming = f.incoming_symptomness * max(0.0, 1.0 - local_support)
        symptom = max(f.symptomness, effective_incoming)
        if symptom > 0.0:
            score *= max(0.45, 1.0 - 0.45 * symptom)
        return max(0.0, float(score))

    def pairwise_margin(self, left: RootnessFeatures, right: RootnessFeatures) -> float:
        """Return positive when ``left`` is more root-like than ``right``.

        This deterministic comparator focuses on causal/rootness properties
        rather than absolute anomaly magnitude: a candidate with more owned,
        local, persistent evidence and less shared-symptom evidence can beat a
        candidate with a slightly higher global anomaly score.
        """
        left_specific = 0.38 * left.ownership + 0.26 * left.locality + 0.18 * left.persistence + 0.18 * left.local_change
        right_specific = 0.38 * right.ownership + 0.26 * right.locality + 0.18 * right.persistence + 0.18 * right.local_change
        specific_margin = left_specific - right_specific

        symptom_margin = right.symptomness - left.symptomness
        structural_margin = (
            0.55 * (left.channel_score - right.channel_score)
            + 0.30 * (left.global_score - right.global_score)
            + 0.15 * (left.channel_rank_score - right.channel_rank_score)
        )
        return 0.50 * specific_margin + 0.30 * symptom_margin + 0.20 * structural_margin

    def rerank_pairwise(
        self,
        rows: Iterable[MutableMapping[str, Any]],
        *,
        feature_key: str = "rootness_features",
    ) -> List[MutableMapping[str, Any]]:
        items = [x for x in list(rows or []) if isinstance(x, MutableMapping)]
        if len(items) <= 1:
            return items

        features: List[RootnessFeatures] = [
            RootnessFeatures.from_mapping(dict(item.get(feature_key, {}) or {}))
            for item in items
        ]
        wins = [0.0 for _ in items]
        margins = [0.0 for _ in items]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                margin = self.pairwise_margin(features[i], features[j])
                margins[i] += margin
                margins[j] -= margin
                if margin > 0.02:
                    wins[i] += 1.0
                elif margin < -0.02:
                    wins[j] += 1.0
                else:
                    # Near ties keep the original score order softly.
                    si = _clip01(items[i].get("score", 0.0))
                    sj = _clip01(items[j].get("score", 0.0))
                    if si >= sj:
                        wins[i] += 0.5
                    else:
                        wins[j] += 0.5
        for item, win, margin in zip(items, wins, margins):
            item["pairwise_rootness_wins"] = float(win)
            item["pairwise_rootness_margin"] = float(margin)
        return sorted(
            items,
            key=lambda item: (
                -float(item.get("pairwise_rootness_wins", 0.0) or 0.0),
                -float(item.get("pairwise_rootness_margin", 0.0) or 0.0),
                -float(item.get("score", 0.0) or 0.0),
            ),
        )
