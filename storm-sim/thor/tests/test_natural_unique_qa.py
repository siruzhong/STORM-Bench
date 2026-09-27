from __future__ import annotations

import re
import unittest

from tools.storm.benchgen.stream_eqa import natural_unique_qa as v24e1


class NaturalUniqueQATest(unittest.TestCase):
    def test_variant_banks_cover_every_known_question_without_duplicates(self):
        specifications = (
            (v24e1.CURRENT_LEADS, v24e1.CURRENT_TAILS, 53),
            (v24e1.FACTUAL_LEADS, v24e1.FACTUAL_TAILS, 52),
            (v24e1.HISTORY_LEADS, v24e1.HISTORY_TAILS, 39),
            (v24e1.STATE_LEADS, v24e1.STATE_DISAPPEAR_TAILS, 48),
            (v24e1.STATE_LEADS, v24e1.STATE_APPEAR_TAILS, 48),
            (v24e1.TRACK_LEADS, v24e1.TRACK_TAILS, 26),
            (v24e1.TEMPORAL_LEADS, v24e1.TEMPORAL_TAILS, 27),
        )
        for leads, tails, count in specifications:
            with self.subTest(count=count, first=leads[0]):
                values = [v24e1._combine(leads, tails, ordinal) for ordinal in range(count)]
                self.assertEqual(len(values), len(set(values)))
                self.assertFalse(any(re.search(r"\d", value) for value in values))

    def test_scene_questions_are_time_free_and_natural(self):
        current = {
            "question_type": "current_state",
            "question": "At 60.0s, what type of room is the agent currently observing?",
            "diagnostics": {"epistemic_status": "known"},
        }
        factual = {
            "question_type": "factual_retrieval",
            "question": "Which type of room is the setting through 60.0s?",
            "diagnostics": {"epistemic_status": "known"},
        }
        for ordinal in range(52):
            self.assertNotRegex(v24e1._naturalize(current, ordinal), r"\d")
            self.assertNotRegex(v24e1._naturalize(factual, ordinal), r"\d")

    def test_uncertain_identity_question_removes_both_timestamps(self):
        question = {
            "question_type": "object_tracking",
            "question": "Is the bread visible at 35.0s the same physical instance seen before 5.0s?",
            "diagnostics": {"epistemic_status": "uncertain"},
            "_distractors": [
                "It is the same physical instance",
                "It is a different physical instance",
                "The earlier instance was destroyed and replaced",
            ],
        }
        natural = v24e1._naturalize(question, 0)
        self.assertNotRegex(natural, r"\d")
        self.assertIn("bread", natural)
        self.assertIn("same physical instance", natural)


if __name__ == "__main__":
    unittest.main()
