from __future__ import annotations

import re
import unittest

from tools.storm.benchgen.stream_eqa import balanced_natural_evidence_qa as v25


class BalancedNaturalEvidenceQATest(unittest.TestCase):
    def test_scene_mix_counts_are_exact(self):
        self.assertEqual(
            sum(v25._is_scene_anchor(i, v25.CURRENT_KNOWN_TOTAL, v25.CURRENT_SCENE_COUNT)
                for i in range(v25.CURRENT_KNOWN_TOTAL)),
            v25.CURRENT_SCENE_COUNT,
        )
        self.assertEqual(
            sum(v25._is_scene_anchor(i, v25.FACTUAL_KNOWN_TOTAL, v25.FACTUAL_SCENE_COUNT)
                for i in range(v25.FACTUAL_KNOWN_TOTAL)),
            v25.FACTUAL_SCENE_COUNT,
        )

    def test_event_question_banks_are_unique_and_time_free(self):
        values = []
        for ordinal in range(v25.CURRENT_KNOWN_TOTAL):
            values.append(v25.v24._combine(v25.CURRENT_EVENT_LEADS, v25.CURRENT_EVENT_TAILS, ordinal))
        self.assertEqual(len(values), len(set(values)))
        self.assertFalse(any(re.search(r"\d", value) for value in values))

        for leads, tails in (
            (v25.FACTUAL_DISAPPEAR_LEADS, v25.FACTUAL_DISAPPEAR_TAILS),
            (v25.FACTUAL_APPEAR_LEADS, v25.FACTUAL_APPEAR_TAILS),
        ):
            values = [v25.v24._combine(leads, tails, ordinal) for ordinal in range(v25.FACTUAL_KNOWN_TOTAL)]
            self.assertEqual(len(values), len(set(values)))
            self.assertFalse(any(re.search(r"\d", value) for value in values))


if __name__ == "__main__":
    unittest.main()
