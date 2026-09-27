from __future__ import annotations

import unittest

from tools.storm.benchgen.stream_eqa import balanced_natural_evidence_qa_exp2 as exp2


class BalancedNaturalEvidenceQAExp2Test(unittest.TestCase):
    def test_rebalanced_scene_mix_counts(self):
        self.assertEqual(
            sum(exp2.exp1._is_scene_anchor(i, exp2.exp1.CURRENT_KNOWN_TOTAL, exp2.CURRENT_SCENE_COUNT)
                for i in range(exp2.exp1.CURRENT_KNOWN_TOTAL)),
            exp2.CURRENT_SCENE_COUNT,
        )
        self.assertEqual(
            sum(exp2.exp1._is_scene_anchor(i, exp2.exp1.FACTUAL_KNOWN_TOTAL, exp2.FACTUAL_SCENE_COUNT)
                for i in range(exp2.exp1.FACTUAL_KNOWN_TOTAL)),
            exp2.FACTUAL_SCENE_COUNT,
        )


if __name__ == "__main__":
    unittest.main()
