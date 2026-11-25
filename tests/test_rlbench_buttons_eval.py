import unittest

from rlbench_buttons.eval_buttons import score_presses, summarize


class ButtonMetricsTest(unittest.TestCase):
    def test_score_awards_points_for_each_correct_position(self):
        order = (0, 1, 2)
        self.assertEqual(score_presses(order, []), 0)
        self.assertEqual(score_presses(order, [(4, 0)]), 30)
        self.assertEqual(score_presses(order, [(4, 1), (9, 1), (12, 2)]), 60)
        self.assertEqual(score_presses(order, [(4, 0), (9, 1), (12, 2)]), 100)

    def test_simultaneous_presses_do_not_establish_an_order(self):
        self.assertEqual(score_presses((0, 1, 2), [(4, 0), (4, 1), (12, 2)]), 30)

    def test_score_and_strict_success_have_separate_denominators(self):
        summary = summarize([
            {"success": True, "score": 100},
            {"success": False, "score": 100},
            {"success": False, "score": 30},
            {"success": False, "score": 0},
        ])
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["success_rate_pct"], 25)
        self.assertEqual(summary["score_total"], 230)
        self.assertEqual(summary["mean_score"], 57.5)


if __name__ == "__main__":
    unittest.main()
