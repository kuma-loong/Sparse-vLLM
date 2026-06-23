import random
import unittest

import numpy as np

from sparsevllm.analysis.select_omnikv_full_layers import (
    CalibrationPoint,
    add_topk_to_pair_scores,
    compute_segment_scores,
    sample_decode_points,
    select_full_layers_dp,
)


class OmniKVFullLayerSelectorTest(unittest.TestCase):
    def test_add_topk_to_pair_scores_counts_forward_layer_intersections(self):
        pair_scores = np.zeros((4, 4), dtype=np.int64)
        add_topk_to_pair_scores(
            pair_scores,
            [
                [1, 2, 3],
                [2, 3, 4],
                [5, 6, 7],
                [1, 3, 7],
            ],
        )

        self.assertEqual(pair_scores[0, 1], 2)
        self.assertEqual(pair_scores[0, 2], 0)
        self.assertEqual(pair_scores[0, 3], 2)
        self.assertEqual(pair_scores[1, 3], 1)
        self.assertEqual(pair_scores[2, 3], 1)
        self.assertEqual(pair_scores[3, 0], 0)

    def test_dp_selects_best_policy_and_counts_final_segment(self):
        pair_scores = np.zeros((5, 5), dtype=np.int64)
        pair_scores[0, 1] = 1
        pair_scores[0, 2] = 1
        pair_scores[0, 3] = 1
        pair_scores[0, 4] = 1
        pair_scores[2, 3] = 10
        pair_scores[2, 4] = 10

        segment_scores = compute_segment_scores(pair_scores)
        selected, score = select_full_layers_dp(segment_scores, 2)

        self.assertEqual(selected, [0, 2])
        self.assertEqual(score, 21)

    def test_dp_tie_breaks_to_earlier_layers(self):
        segment_scores = np.zeros((5, 6), dtype=np.int64)
        selected, score = select_full_layers_dp(segment_scores, 3)

        self.assertEqual(score, 0)
        self.assertEqual(selected, [0, 1, 2])

    def test_sample_decode_points_adds_answer_boundary_after_random_points(self):
        points = sample_decode_points(
            sample_idx=3,
            prompt_token_ids=list(range(100)),
            answer_query_token_id=999,
            random_points_per_sample=4,
            rng=random.Random(7),
            num_sink_tokens=0,
            num_recent_tokens=32,
            min_prefix_tokens=1,
        )

        self.assertEqual(len(points), 5)
        self.assertIsInstance(points[-1], CalibrationPoint)
        self.assertEqual(points[-1].kind, "answer_boundary")
        self.assertEqual(points[-1].prefix_len, 100)
        self.assertEqual(points[-1].query_token_id, 999)
        self.assertTrue(all(point.prefix_len >= 32 for point in points if point.kind == "random"))
        self.assertEqual([point.prefix_len for point in points], sorted(point.prefix_len for point in points))


if __name__ == "__main__":
    unittest.main()
