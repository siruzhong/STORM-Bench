from __future__ import annotations

import unittest

from tools.storm.benchgen.stream_eqa import visual_contrast_qa_exp2 as v22e2


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


class VisualContrastQAExp2Test(unittest.TestCase):
    def setUp(self):
        v22e2._configure_base()

    def _assert_object_options(self, question):
        options = [question["_correct"], *question["_distractors"]]
        self.assertEqual(set(options), {"apple", "bowl", "bread", "cup"})
        self.assertTrue(all(start < end <= question["query_time"] for start, end in question["evidence_spans"]))

    def test_current_state_is_short_visible_return(self):
        question = v22e2._build_current_known(
            "episode", "FloorPlan1", _events(), 0, 60, v22e2.base.REFUSAL_OPTIONS[0],
        )
        self._assert_object_options(question)
        self.assertIn("newly visible", question["question"])
        self.assertLessEqual(question["evidence_spans"][0][1] - question["evidence_spans"][0][0], 3.0)

    def test_history_aggregates_exactly_two_local_returns(self):
        question = v22e2._build_history_known(
            "episode", "FloorPlan1", _events(), 0, 60, v22e2.base.REFUSAL_OPTIONS[0],
        )
        self._assert_object_options(question)
        self.assertIn("which one returns last", question["question"].lower())
        self.assertEqual(question["question_subtype"], v22e2.SUBTYPES["history_aggregation"])

    def test_temporal_asks_for_second_local_return(self):
        question = v22e2._build_temporal_known(
            "episode", "FloorPlan1", _events(), 0, 60, v22e2.base.REFUSAL_OPTIONS[0],
        )
        self._assert_object_options(question)
        self.assertIn("second to change", question["question"])
        self.assertEqual(question["question_subtype"], v22e2.SUBTYPES["temporal_reasoning"])

    def test_exp1_state_change_builder_is_retained(self):
        self.assertIs(v22e2.base._build_state_known, v22e2.exp1._build_state_known)


if __name__ == "__main__":
    unittest.main()
