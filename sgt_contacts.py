"""Measure robot–cup contact and TCP proximity in legacy ShellGameTouch-v0."""

from contextlib import contextmanager

import numpy as np


CUP_NAMES = ("left", "center", "right")


class CupTracker:
    """Track the first cup contact and the first cup reached in one CPU rollout."""

    def __init__(self, env):
        self.env = env.unwrapped
        if self.env.num_envs != 1 or self.env.gpu_sim_enabled:
            raise ValueError("Cup tracking needs one CPU physics environment")

        self.target_cup = CUP_NAMES[self.env.cup_with_ball_number.item()]
        self.mugs = {name: getattr(self.env, f"mug_{name}") for name in CUP_NAMES}
        # ManiSkill 3.0.0b15 uses these same wrapped-actor entities for CPU contacts.
        self.cup_by_entity = {
            mug._bodies[0].entity: name for name, mug in self.mugs.items()
        }
        self.robot_entities = {
            link._bodies[0].entity for link in self.env.agent.robot.get_links()
        }
        self.first_touched_cups = []
        self.first_touch_step = None
        self.first_touch_substep = None
        self.first_reached_cups = []
        self.first_reach_step = None
        self.sampled_step = None

    @contextmanager
    def track_contacts(self, step):
        """Sample after each PhysX substep while an environment step is running."""
        if step < self.env.TIME_OFFSET or self.first_touch_step is not None:
            yield
            return

        previous_hook = self.env._after_simulation_step
        substep = 0

        def sample_contacts():
            nonlocal substep
            previous_hook()
            substep += 1
            if self.first_touch_step is None:
                self._observe_contacts(step, substep)

        self.env._after_simulation_step = sample_contacts
        try:
            yield
        finally:
            self.env._after_simulation_step = previous_hook

    def _observe_contacts(self, step, substep=None):
        self.sampled_step = step
        touched = set()
        for contact in self.env.scene.get_contacts():
            first, second = (body.entity for body in contact.bodies)
            if first in self.robot_entities:
                cup = self.cup_by_entity.get(second)
            elif second in self.robot_entities:
                cup = self.cup_by_entity.get(first)
            else:
                continue

            # PhysX also reports nearby pairs before they physically interact.
            if cup is not None and any(
                point.separation <= 0 or np.any(point.impulse)
                for point in contact.points
            ):
                touched.add(cup)

        if touched:
            self.first_touched_cups = [name for name in CUP_NAMES if name in touched]
            self.first_touch_step = step
            self.first_touch_substep = substep

    def observe(self, step):
        if step < self.env.TIME_OFFSET:
            return
        if self.first_touch_step is None and self.sampled_step != step:
            self._observe_contacts(step)
        if self.first_reach_step is not None:
            return

        tcp_position = np.asarray(self.env.agent.tcp.pose.p)
        reached = [
            name
            for name, mug in self.mugs.items()
            if np.linalg.norm(np.asarray(mug.pose.p) - tcp_position)
            <= self.env.GOAL_THRESH
        ]
        if reached:
            self.first_reached_cups = reached
            self.first_reach_step = step

    def as_result(self):
        first_touch = (
            self.first_touched_cups[0] if len(self.first_touched_cups) == 1 else None
        )
        first_reach = (
            self.first_reached_cups[0] if len(self.first_reached_cups) == 1 else None
        )
        return {
            "target_cup": self.target_cup,
            "first_touched_cup": first_touch,
            "first_touched_cups": self.first_touched_cups,
            "first_touch_step": self.first_touch_step,
            "first_touch_substep": self.first_touch_substep,
            "first_touch_correct": first_touch == self.target_cup,
            "first_reached_cup": first_reach,
            "first_reached_cups": self.first_reached_cups,
            "first_reach_step": self.first_reach_step,
            "first_reach_correct": first_reach == self.target_cup,
        }
