from __future__ import annotations

import argparse
import json

from rca.adapter.profiler import AdaptiveKnobs, DatasetProfiler, build_adaptive_basil_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an unlabeled dataset profile for adaptive BASIL planning.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_files", type=int, default=80)
    parser.add_argument("--max_rows_per_file", type=int, default=200_000)
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument("--evidence_budget", default="medium", choices=["small", "medium", "large", "fast", "accurate"])
    parser.add_argument("--history_weight", type=float, default=0.0)
    parser.add_argument("--llm_risk_tolerance", default="conservative", choices=["conservative", "balanced", "aggressive"])
    args = parser.parse_args()

    profile = DatasetProfiler(
        args.dataset,
        max_files=int(args.max_files),
        max_rows_per_file=int(args.max_rows_per_file),
    ).build()
    config = build_adaptive_basil_config(
        profile,
        AdaptiveKnobs(
            confidence_level=float(args.confidence_level),
            evidence_budget=str(args.evidence_budget),
            history_weight=float(args.history_weight),
            llm_risk_tolerance=str(args.llm_risk_tolerance),
        ),
    )
    print(
        json.dumps(
            {
                "profile": profile.to_dict(),
                "adaptive_basil_config": config.to_dict(),
                "basil_mapping": config.to_basil_mapping(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
