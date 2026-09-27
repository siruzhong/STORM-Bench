from __future__ import annotations

import unittest

from tools.storm.benchgen.stream_eqa import scene_grounded_qa as v23e2


def _event():
    return {
        "start_sec": 2.0,
        "end_sec": 5.0,
        "event_type": "disappearance",
        "subject": "Apple",
        "_target_id": "Apple|instance",
        "_index": 0,
        "_surface": "countertop",
    }


class SceneGroundedQATest(unittest.TestCase):
    def setUp(self):
        v23e2._configure_base()

    def _assert_scene_question(self, question, correct: str, question_type: str):
        options = [question["_correct"], *question["_distractors"]]
        self.assertEqual(set(options), set(v23e2.ROOM_OPTIONS))
        self.assertEqual(question["_correct"], correct)
        self.assertEqual(question["question_type"], question_type)
        self.assertEqual(question["query_time"], 60.0)
        self.assertEqual(question["evidence_spans"], [[0.0, 60.0]])

    def test_current_maps_kitchen_scene(self):
        question = v23e2._build_current_known(
            "episode", "FloorPlan12", [_event()], 0, 60, v23e2.base.REFUSAL_OPTIONS[0],
        )
        self._assert_scene_question(question, "kitchen", "current_state")

    def test_factual_maps_all_other_room_ranges(self):
        for scene, expected in (
            ("FloorPlan221", "living room"),
            ("FloorPlan308", "bedroom"),
            ("FloorPlan404", "bathroom"),
        ):
            with self.subTest(scene=scene):
                question = v23e2._build_factual_known(
                    "episode", scene, [_event()], 0, 60, v23e2.base.REFUSAL_OPTIONS[0],
                )
                self._assert_scene_question(question, expected, "factual_retrieval")

    def test_other_known_builders_are_retained_from_v22_exp2(self):
        self.assertIs(v23e2.base._build_history_known, v23e2.exp2._build_history_known)
        self.assertIs(v23e2.base._build_object_known, v23e2.exp2.exp1.base._build_object_known)
        self.assertIs(v23e2.base._build_state_known, v23e2.exp2.exp1._build_state_known)
        self.assertIs(v23e2.base._build_temporal_known, v23e2.exp2._build_temporal_known)


if __name__ == "__main__":
    unittest.main()
