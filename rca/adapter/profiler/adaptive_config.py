from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from rca.adapter.profiler.schema import AdaptiveKnobs, DatasetProfile


@dataclass(frozen=True)
class AdaptiveBasilConfig:
    severity_floor: float
    top_k: int
    candidate_reason_top_r: int
    time_candidate_top_k: int
    max_lag_s: int
    bucket_size_s: int
    evidence_score_gap_threshold: float
    evidence_top1_score_floor: float
    gate_skip_gap_threshold: float
    historical_top_k: int
    history_weight: float
    profile_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_basil_mapping(self) -> Dict[str, Any]:
        """Return keys compatible with the current BASIL config naming."""

        return {
            "severity_floor": float(self.severity_floor),
            "top_k": int(self.top_k),
            "candidate_reason_top_r": int(self.candidate_reason_top_r),
            "time_candidate_top_k": int(self.time_candidate_top_k),
            "max_lag_s": int(self.max_lag_s),
            "bucket_size_s": int(self.bucket_size_s),
            "historical_top_k": int(self.historical_top_k),
            "history_weight": float(self.history_weight),
            "evidence_score_gap_threshold": float(self.evidence_score_gap_threshold),
            "evidence_top1_score_floor": float(self.evidence_top1_score_floor),
            "gate_skip_gap_threshold": float(self.gate_skip_gap_threshold),
        }


def build_adaptive_basil_config(
    profile: DatasetProfile,
    knobs: AdaptiveKnobs | Mapping[str, Any] | None = None,
) -> AdaptiveBasilConfig:
    """Calibrate BASIL query parameters from an unlabeled dataset profile."""

    if knobs is None:
        k = AdaptiveKnobs()
    elif isinstance(knobs, AdaptiveKnobs):
        k = knobs
    else:
        k = AdaptiveKnobs(**dict(knobs))

    confidence = _clip(float(k.confidence_level), 0.80, 0.995)
    budget_scale = _budget_scale(str(k.evidence_budget or "medium"))
    risk = str(k.llm_risk_tolerance or "conservative").lower()

    noise = _avg_metric(profile, "noise_mad_ratio")
    burst = _avg_metric(profile, "burstiness_p99_p50", default=1.0)
    sparsity = _avg_metric(profile, "sparsity")
    uncertainty = _clip(float(profile.budget.score_entropy_hint + noise + sparsity), 0.0, 2.5)

    top_k = int(round(profile.budget.estimated_top_k * budget_scale))
    top_k = int(max(5, min(64, top_k)))
    reason_top_r = int(max(2, min(10, round(profile.budget.estimated_reason_top_r * budget_scale))))
    time_top_k = int(max(4, min(20, round(profile.budget.estimated_time_top_k * budget_scale))))

    # Higher confidence and burstiness produce stricter segment admission.
    severity_floor = _clip((confidence - 0.80) / 0.195 + min(0.25, max(0.0, burst - 5.0) * 0.02), 0.0, 1.0)

    interval = float(profile.median_sampling_interval_s or 60.0)
    bucket_size_s = int(max(30, min(600, round(interval * 2))))
    max_lag_s = int(max(profile.typical_propagation_lag_s, bucket_size_s * 2))

    if risk == "aggressive":
        gap_threshold = 0.10 + 0.08 * uncertainty
        top1_floor = 0.20 + 0.10 * uncertainty
        skip_gap = 0.38
    elif risk == "balanced":
        gap_threshold = 0.13 + 0.08 * uncertainty
        top1_floor = 0.25 + 0.10 * uncertainty
        skip_gap = 0.32
    else:
        gap_threshold = 0.16 + 0.08 * uncertainty
        top1_floor = 0.30 + 0.10 * uncertainty
        skip_gap = 0.28

    history_weight = _clip(float(k.history_weight), 0.0, 1.0)
    historical_top_k = 0 if history_weight <= 1e-12 else max(1, min(20, int(round(4 + 8 * history_weight))))

    return AdaptiveBasilConfig(
        severity_floor=float(severity_floor),
        top_k=top_k,
        candidate_reason_top_r=reason_top_r,
        time_candidate_top_k=time_top_k,
        max_lag_s=max_lag_s,
        bucket_size_s=bucket_size_s,
        evidence_score_gap_threshold=float(_clip(gap_threshold, 0.05, 0.40)),
        evidence_top1_score_floor=float(_clip(top1_floor, 0.15, 0.65)),
        gate_skip_gap_threshold=float(_clip(skip_gap, 0.15, 0.60)),
        historical_top_k=historical_top_k,
        history_weight=history_weight,
        profile_summary={
            "component_count": profile.component_count,
            "metric_count": profile.metric_count,
            "day_count": profile.day_count,
            "median_sampling_interval_s": profile.median_sampling_interval_s,
            "noise_mad_ratio_avg": noise,
            "burstiness_p99_p50_avg": burst,
            "sparsity_avg": sparsity,
            "availability": profile.availability.__dict__,
        },
    )


def _avg_metric(profile: DatasetProfile, attr: str, default: float = 0.0) -> float:
    values = [float(getattr(p, attr)) for p in profile.metric_families.values() if getattr(p, attr, None) is not None]
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def _budget_scale(name: str) -> float:
    text = str(name or "medium").strip().lower()
    if text in {"small", "low", "fast"}:
        return 0.75
    if text in {"large", "high", "accurate"}:
        return 1.35
    return 1.0


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))
