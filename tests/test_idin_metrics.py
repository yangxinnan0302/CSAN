import unittest

import numpy as np

from analyze_idin_stages import (
    compute_ground_truth_metrics,
    compute_label_free_metrics,
    js_divergence,
    normalized_entropy,
)


class IDINMetricTests(unittest.TestCase):
    def test_js_identity_and_symmetry(self):
        left = np.asarray([[0.8, 0.2], [0.1, 0.9]])
        right = np.asarray([[0.3, 0.7], [0.6, 0.4]])
        np.testing.assert_allclose(js_divergence(left, left), 0.0, atol=1e-10)
        np.testing.assert_allclose(js_divergence(left, right),
                                   js_divergence(right, left), atol=1e-10)

    def test_entropy_bounds(self):
        peaked = np.asarray([[1.0, 0.0, 0.0]])
        uniform = np.asarray([[1.0 / 3.0] * 3])
        self.assertAlmostEqual(float(normalized_entropy(peaked)[0]), 0.0, places=7)
        self.assertAlmostEqual(float(normalized_entropy(uniform)[0]), 1.0, places=7)

    def test_label_free_stage_metrics(self):
        attention = np.asarray([
            [[0.8, 0.2], [0.4, 0.6]],
            [[0.7, 0.3], [0.9, 0.1]],
        ])
        context = np.asarray([
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ])
        rows = compute_label_free_metrics(attention, context, [0, 1], top_k=1)
        self.assertEqual(rows[0]["stage"], 0)
        self.assertAlmostEqual(rows[1]["top1_switch_rate"], 0.5)
        self.assertAlmostEqual(rows[1]["topk_overlap"], 0.5)

    def test_correction_and_regression(self):
        attention = np.asarray([
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.1, 0.9], [0.8, 0.2]],
        ])
        labels = {"0": [1], "1": [1]}
        rows, summary = compute_ground_truth_metrics(attention, labels)
        self.assertAlmostEqual(rows[0]["alignment_accuracy"], 0.5)
        self.assertAlmostEqual(rows[1]["alignment_accuracy"], 0.5)
        self.assertAlmostEqual(summary["early_error_correction_rate"], 1.0)
        self.assertAlmostEqual(summary["regression_rate"], 1.0)
        self.assertEqual(summary["stage_correct_counts"], [1, 1])
        self.assertEqual(summary["stage_label_counts"], [2, 2])


if __name__ == "__main__":
    unittest.main()
