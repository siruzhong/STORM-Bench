from __future__ import annotations

import unittest

from tools.storm.benchgen.stream_eqa import dual_protocol_qa as v23e1


def _event(index: int, target: str, event_type: str, start: float, end: float):
    return {
        "start_sec": start,
        "end_sec": end,
        "event_type": event_type,
        "subject": target,
        "_target_id": f"{target}|instance",
        "_index": index,
        "_surface": "countertop",
    }


def _events():
    return [
        _event(0, "Apple", "disappearance", 2.0, 5.0),
        _event(1, "Bowl", "disappearance", 8.0, 11.0),
        _event(2, "Bread", "disappearance", 14.0, 17.0),
        _event(3, "Cup", "disappearance", 20.0, 23.0),
        _event(4, "Apple", "appearance", 32.0, 35.0),
        _event(5, "Bowl", "appearance", 37.0, 40.0),
        _event(6, "Bread", "appearance", 43.0, 46.0),
        _event(7, "Cup", "appearance", 49.0, 52.0),
    ]


class DualProtocolQATest(unittest.TestCase):
    def setUp(self):
        v23e1._configure_base()

    def _assert_final_return_question(self, question, phrase: str):
        options = [question["_correct"], *question["_distractors"]]
        self.assertEqual(set(options), {"apple", "bowl", "bread", "cup"})
        self.assertEqual(question["_correct"], "cup")
        self.assertEqual(question["query_time"], 60.0)
        self.assertIn(phrase, question["question"].lower())
        self.assertTrue(all(start < end <= 60.0 for start, end in question["evidence_spans"]))

    def test_current_uses_final_observed_state(self):
        question = v23e1._build_current_known(
            "episode", "FloorPlan1", _events(), 0, 60, v23e1.base.REFUSAL_OPTIONS[0],
        )
        self._assert_final_return_question(question, "final observed state")

    def test_factual_uses_final_absent_to_visible_event(self):
        question = v23e1._build_factual_known(
            "episode", "FloorPlan1", _events(), 0, 60, v23e1.base.REFUSAL_OPTIONS[0],
        )
        self._assert_final_return_question(question, "final change")

    def test_history_uses_complete_visible_history(self):
        question = v23e1._build_history_known(
            "episode", "FloorPlan1", _events(), 0, 60, v23e1.base.REFUSAL_OPTIONS[0],
        )
        self._assert_final_return_question(question, "complete visible history")

    def test_tracking_uses_last_completed_track(self):
        question = v23e1._build_object_known(
            "episode", "FloorPlan1", _events(), 0, 60, v23e1.base.REFUSAL_OPTIONS[0],
        )
        self._assert_final_return_question(question, "track last")

    def test_state_and_temporal_builders_are_retained(self):
        self.assertIs(v23e1.base._build_state_known, v23e1.exp2.exp1._build_state_known)
        self.assertIs(v23e1.base._build_temporal_known, v23e1.exp2._build_temporal_known)


if __name__ == "__main__":
    unittest.main()
