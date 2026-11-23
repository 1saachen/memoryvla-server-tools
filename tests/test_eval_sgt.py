import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

try:
    import numpy as np

    import eval_sgt
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "PIL", "requests"}:
        raise
    raise unittest.SkipTest(f"SGT evaluator dependencies are missing: {error.name}") from error


class EvalSgtTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((1, 2, 2, 3), dtype=np.uint8)

    def observation(self, step):
        return {"sensor_data": {"base_camera": {"rgb": self.rgb + step}}}

    def run_rollout(self, mode, end_step):
        env = Mock()
        env.reset.return_value = (self.observation(0), {})
        env.step.side_effect = [
            (self.observation(step), 0, step == end_step, False, {"success": step == end_step})
            for step in range(1, end_step + 1)
        ]
        tracker = Mock()
        tracker.track_contacts.side_effect = lambda step: nullcontext()
        tracker.as_result.return_value = {"first_touch_correct": False}
        policy = Mock(return_value=np.zeros((eval_sgt.CHUNK_SIZE, 7), dtype=np.float32))
        with patch.object(eval_sgt, "CupTracker", return_value=tracker):
            result = eval_sgt.rollout(env, policy, 7, None, mode)
        env.reset.assert_called_once_with(seed=7)
        return result, policy, tracker

    def test_clear_only_at_first_post_occlusion_request(self):
        keep, keep_policy, _ = self.run_rollout("keep", 14)
        clear, clear_policy, tracker = self.run_rollout("clear", 14)
        self.assertEqual(keep["pre_clear_trace_sha256"], clear["pre_clear_trace_sha256"])
        self.assertEqual(clear["memory_clear_step"], 8)
        self.assertIsNone(keep["memory_clear_step"])
        self.assertEqual([call.kwargs["clear_memory"] for call in clear_policy.call_args_list],
                         [False, False, True, False])
        self.assertFalse(any(call.kwargs["clear_memory"] for call in keep_policy.call_args_list))
        self.assertEqual([call.kwargs["episode_seed"] for call in clear_policy.call_args_list],
                         [7, None, None, None])
        self.assertEqual([call.args[0] for call in tracker.observe.call_args_list], list(range(5, 15)))
        self.assertTrue(clear["success"])
        self.assertFalse(clear["first_touch_correct"])

    def test_rollout_can_finish_before_eligible_clear(self):
        result, policy, _ = self.run_rollout("clear", 7)
        self.assertIsNone(result["memory_clear_step"])
        self.assertFalse(result["first_post_cue_request_reached"])
        self.assertFalse(any(call.kwargs["clear_memory"] for call in policy.call_args_list))

    def test_requires_server_acknowledgements(self):
        session = Mock()
        payload = {"response": ";".join(["0 0 0 0 0 0 0"] * eval_sgt.CHUNK_SIZE)}
        session.post.return_value.json.return_value = payload
        with self.assertRaisesRegex(RuntimeError, "confirm the episode seed"):
            eval_sgt.request_actions(session, "sgt", self.rgb[0], True, episode_seed=7)
        with self.assertRaisesRegex(RuntimeError, "confirm the memory clear"):
            eval_sgt.request_actions(session, "sgt", self.rgb[0], False, clear_memory=True)
        payload["memory_cleared"] = True
        actions = eval_sgt.request_actions(session, "sgt", self.rgb[0], False, clear_memory=True)
        self.assertEqual(actions.shape, (eval_sgt.CHUNK_SIZE, 7))
        self.assertEqual(session.post.call_args.kwargs["data"]["clear_memory"], "True")
