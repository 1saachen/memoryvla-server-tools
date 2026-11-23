import json
import unittest

from sgt_metrics import summarize_pairs, summarize_results


def episode(seed, **overrides):
    return {
        "seed": seed,
        "success": False,
        "first_touch_correct": None,
        "first_touch_step": None,
        "first_touched_cups": [],
        "first_reach_correct": None,
        "first_reach_step": None,
        "memory_clear_step": None,
        **overrides,
    }


class SgtMetricsTests(unittest.TestCase):
    def test_all_episodes_count_toward_each_rate(self):
        results = [
            episode(
                7, success=True, first_touch_correct=True, first_touch_step=10,
                first_touched_cups=[0], first_reach_correct=True, first_reach_step=9,
                memory_clear_step=8,
            ),
            episode(8, first_touch_step=9, first_touched_cups=[0, 1]),
            episode(9, success=True),
            episode(10),
        ]
        self.assertEqual(summarize_results(results), {
            "num_eval": 4,
            "successes": 2,
            "success_rate": 0.5,
            "first_touch_successes": 1,
            "first_touch_success_rate": 0.25,
            "first_reach_successes": 1,
            "first_reach_success_rate": 0.25,
            "no_first_touch": 2,
            "ambiguous_first_touch": 1,
            "no_first_reach": 3,
            "memory_clears": 1,
        })

    def test_paired_counts_distinguish_outcomes_and_check_pre_clear_events(self):
        keep = [
            episode(
                7, success=True, first_touch_correct=True, first_touch_step=8,
                first_reach_correct=True, first_reach_step=7, pre_clear_trace_sha256="same",
            ),
            episode(
                8, success=True, first_touch_correct=True, first_touch_step=10,
                first_reach_correct=True, first_reach_step=9, pre_clear_trace_sha256="keep",
            ),
            episode(
                9, success=True, first_touch_correct=False, first_touch_step=6,
                first_reach_correct=True, first_reach_step=6,
            ),
            episode(10),
            episode(11, first_touch_step=4, first_reach_step=4),
        ]
        clear = [
            episode(
                7, success=True, first_touch_correct=True, first_touch_step=8,
                first_reach_correct=True, first_reach_step=7, memory_clear_step=8,
                pre_clear_trace_sha256="same",
            ),
            episode(
                8, first_touch_correct=False, first_touch_step=10,
                first_reach_correct=True, first_reach_step=9, memory_clear_step=8,
                pre_clear_trace_sha256="clear",
            ),
            episode(9, memory_clear_step=8),
            episode(
                10, success=True, first_touch_correct=True, first_touch_step=10,
                first_reach_correct=True, first_reach_step=9, memory_clear_step=8,
            ),
            episode(11),
        ]
        summary = summarize_pairs(keep, clear)
        self.assertEqual(summary["num_pairs"], 5)
        self.assertEqual(summary["keep_memory_clears"], 0)
        self.assertEqual(summary["clear_memory_clears"], 4)
        self.assertEqual(summary["official"], {
            "both_correct": 1, "keep_only": 2, "clear_only": 1, "neither": 1,
            "clear_minus_keep_percentage_points": -20.0,
        })
        self.assertEqual(summary["first_touch"], {
            "both_correct": 1, "keep_only": 1, "clear_only": 1, "neither": 2,
            "clear_minus_keep_percentage_points": 0.0,
        })
        self.assertEqual(summary["first_reach"], {
            "both_correct": 2, "keep_only": 1, "clear_only": 1, "neither": 1,
            "clear_minus_keep_percentage_points": 0.0,
        })
        self.assertEqual(summary["pre_clear_trace"], {"exact_match": 1, "differing": 1, "missing": 3})
        self.assertEqual(summary["first_touch_at_or_before_clear"], {"keep": 2, "clear": 1})
        self.assertEqual(summary["first_reach_at_or_before_clear"], {"keep": 2, "clear": 1})
        self.assertEqual(json.loads(json.dumps(summary, allow_nan=False)), summary)

    def test_paired_seed_validation_rejects_out_of_order_missing_or_duplicate_seeds(self):
        original = [episode(7), episode(8)]
        with self.assertRaisesRegex(ValueError, "same seeds in the same order"):
            summarize_pairs(original, list(reversed(original)))
        with self.assertRaisesRegex(ValueError, "same seeds in the same order"):
            summarize_pairs(original, original[:1])
        duplicates = [episode(7), episode(7)]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            summarize_pairs(duplicates, duplicates)

    def test_empty_results_have_no_rate_or_paired_comparison(self):
        summary = summarize_results([])
        self.assertEqual(summary["num_eval"], 0)
        for field in ("success_rate", "first_touch_success_rate", "first_reach_success_rate"):
            self.assertIsNone(summary[field])
        with self.assertRaisesRegex(ValueError, "At least one paired seed"):
            summarize_pairs([], [])


if __name__ == "__main__":
    unittest.main()
