from __future__ import annotations

import unittest
from datetime import datetime, timezone

from rca.memory.evidence import (
    BasilEvidence,
    TaskType,
    build_component_candidates,
    build_reason_candidates,
    build_time_candidates,
    get_task_type,
)


class BasilEvidenceBuilderTest(unittest.TestCase):
    def _mock_evidence(self) -> BasilEvidence:
        t0 = datetime(2020, 4, 11, 20, 40, tzinfo=timezone.utc)
        t1 = datetime(2020, 4, 11, 20, 43, tzinfo=timezone.utc)
        return BasilEvidence(
            incident_id="Telecom:2020-04-11:1",
            query_window=(t0, t1),
            task_type=TaskType.COMPONENT_REASON_TIME,
            component_scores={"docker_008": 0.9, "os_020": 0.7},
            time_candidates={"docker_008": [t0, t1], "os_020": [t1]},
            reason_scores={"docker_008": {"CPU fault": 0.8}, "os_020": {"network loss": 0.6}},
            supporting_metrics={"docker_008": ["cpu_util"], "os_020": ["net_loss"]},
            causal_scores={"docker_008": 0.4, "os_020": 0.2},
            low_confidence=True,
            confidence_reason="insufficient segments",
        )

    def test_task_type_mapping_is_complete(self) -> None:
        self.assertEqual(get_task_type(1), TaskType.TIME_ONLY)
        self.assertEqual(get_task_type(2), TaskType.REASON_ONLY)
        self.assertEqual(get_task_type(3), TaskType.COMPONENT_ONLY)
        self.assertEqual(get_task_type(4), TaskType.REASON_TIME)
        self.assertEqual(get_task_type(5), TaskType.COMPONENT_TIME)
        self.assertEqual(get_task_type(6), TaskType.COMPONENT_REASON)
        self.assertEqual(get_task_type(7), TaskType.COMPONENT_REASON_TIME)

    def test_component_builder_uses_scores_and_propagates_confidence(self) -> None:
        result = build_component_candidates(self._mock_evidence(), top_k=2)
        self.assertEqual(result.candidates[0], "docker_008")
        self.assertTrue(result.low_confidence)
        self.assertEqual(result.confidence_reason, "insufficient segments")

    def test_time_builder_returns_component_times(self) -> None:
        result = build_time_candidates(self._mock_evidence(), top_k=3)
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(result.candidates[0].component, "docker_008")
        self.assertIn("basil_", result.candidates[0].source)
        self.assertTrue(result.low_confidence)

    def test_reason_builder_prefers_basil_and_keeps_fallback_for_comparison(self) -> None:
        result = build_reason_candidates(self._mock_evidence(), fallback_candidates=["network loss", "db close"])
        self.assertEqual(result.candidates[0], "CPU fault")
        self.assertIn("db close", result.fallback_candidates)
        self.assertTrue(result.low_confidence)


if __name__ == "__main__":
    unittest.main()
