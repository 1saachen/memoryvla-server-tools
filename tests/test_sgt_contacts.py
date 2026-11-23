import unittest
from types import SimpleNamespace

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    raise unittest.SkipTest("SGT contact tests require NumPy") from error

from sgt_contacts import CupTracker


def body_at(x):
    return SimpleNamespace(
        _bodies=[SimpleNamespace(entity=object())],
        pose=SimpleNamespace(p=np.array([[x, 0, 0]], dtype=np.float32)),
    )


def contact_between(first, second, separation=0.0, impulse=(0, 0, 0)):
    return SimpleNamespace(
        bodies=[first._bodies[0], second._bodies[0]],
        points=[SimpleNamespace(separation=separation, impulse=impulse)],
    )


class FakeShellGame:
    num_envs = 1
    gpu_sim_enabled = False
    TIME_OFFSET = 5
    GOAL_THRESH = 0.08

    def __init__(self):
        self.unwrapped = self
        self.cup_with_ball_number = np.array([2])
        self.mug_left = body_at(-0.2)
        self.mug_center = body_at(0.0)
        self.mug_right = body_at(0.2)
        self.finger = body_at(1)
        self.agent = SimpleNamespace(
            tcp=body_at(1), robot=SimpleNamespace(get_links=lambda: [self.finger])
        )
        self.contacts = []
        self.scene = SimpleNamespace(get_contacts=lambda: self.contacts)
        self.substep_calls = 0

    def _after_simulation_step(self):
        self.substep_calls += 1


class CupTrackerTests(unittest.TestCase):
    def test_transient_first_contact_survives_later_substeps_and_target_reach(self):
        env = FakeShellGame()
        tracker = CupTracker(env)
        original_hook = env._after_simulation_step

        with tracker.track_contacts(6):
            env.contacts = [contact_between(env.finger, env.mug_center)]
            env._after_simulation_step()
            env.contacts = [contact_between(env.mug_right, env.finger)]
            env._after_simulation_step()
        env.agent.tcp.pose.p = env.mug_right.pose.p
        tracker.observe(6)

        result = tracker.as_result()
        self.assertEqual(env._after_simulation_step, original_hook)
        self.assertEqual(env.substep_calls, 2)
        self.assertEqual(result["first_touched_cup"], "center")
        self.assertEqual(result["first_touch_step"], 6)
        self.assertEqual(result["first_touch_substep"], 1)
        self.assertFalse(result["first_touch_correct"])
        self.assertEqual(result["first_reached_cup"], "right")
        self.assertEqual(result["first_reach_step"], 6)
        self.assertTrue(result["first_reach_correct"])

    def test_ignores_cue_phase_and_noninteracting_pairs(self):
        env = FakeShellGame()
        tracker = CupTracker(env)
        env.contacts = [contact_between(env.finger, env.mug_right)]
        with tracker.track_contacts(4):
            env._after_simulation_step()
        tracker.observe(4)
        env.contacts = [
            contact_between(env.finger, env.mug_right, separation=0.005),
            contact_between(env.mug_center, env.mug_right),
        ]
        tracker.observe(5)

        result = tracker.as_result()
        self.assertIsNone(result["first_touch_step"])
        self.assertFalse(result["first_touch_correct"])
        self.assertIsNone(result["first_reach_step"])
        self.assertFalse(result["first_reach_correct"])

    def test_same_substep_contacts_remain_ambiguous(self):
        env = FakeShellGame()
        tracker = CupTracker(env)
        env.contacts = [
            contact_between(env.mug_right, env.finger),
            contact_between(env.finger, env.mug_left),
            contact_between(env.finger, env.mug_left),
        ]
        with tracker.track_contacts(6):
            env._after_simulation_step()
        tracker.observe(6)

        result = tracker.as_result()
        self.assertEqual(result["first_touched_cups"], ["left", "right"])
        self.assertIsNone(result["first_touched_cup"])
        self.assertFalse(result["first_touch_correct"])


if __name__ == "__main__":
    unittest.main()
