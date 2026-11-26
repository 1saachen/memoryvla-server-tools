from pathlib import Path
import tempfile
import unittest

from rlbench_buttons.eval_buttons import expert_prefix_steps, rank_layouts_near_start, score_presses, summarize
from rlbench_buttons.instructions import paper_instruction


class ButtonMetricsTest(unittest.TestCase):
    def test_near_start_ranks_by_all_buttons_and_uses_horizontal_distance(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            layouts = ([0.11, 0.31, 0.01], [0.13, 0.10, 0.15])
            for index, offsets in enumerate(layouts):
                x = np.asarray(offsets) + 0.4
                layout = np.column_stack((x, [0.2] * 3, [0.75] * 3))
                np.savez(root / f"order2_episode{index:05}.npz", layout=layout,
                         pose=[[0.4, 0.2, 1.5, 0, 0, 0, 1]])
            ranked = rank_layouts_near_start(root, 2)
        self.assertEqual([path.stem for path, _, _ in ranked],
                         ["order2_episode00001", "order2_episode00000"])
        self.assertAlmostEqual(ranked[0][1], 150.0)
        self.assertAlmostEqual(ranked[0][2], 100.0)

    def test_handoff_leaves_five_actions_including_the_trigger_action(self):
        ticks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.assertEqual(expert_prefix_steps(ticks, 15), 3)
        self.assertEqual(expert_prefix_steps(ticks, 16), 3)

    def test_handoff_requires_a_trigger_within_the_recorded_actions(self):
        ticks = [0, 2, 4, 6, 8, 10, 12]
        for tick in (0, 8, 13):
            with self.subTest(tick=tick), self.assertRaises(ValueError):
                expert_prefix_steps(ticks, tick)

    def test_paper_instruction_preserves_target_order(self):
        self.assertEqual(
            paper_instruction((1, 0, 2)),
            "Push the green, blue, and pink buttons in sequence",
        )

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
