"""Collect strict three-button demonstrations with the pinned RLBench checkout."""

import argparse
import itertools
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError, TaskEnvironmentError
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.push_buttons import PushButtons as StockPushButtons
from scipy.spatial.transform import Rotation


COLORS = ("blue", "green", "pink")
PALETTE = ((0.0, 0.25, 0.9), (0.1, 0.65, 0.12), (0.95, 0.35, 0.65))
ORDERS = tuple(itertools.permutations(range(3)))
PRESS = 0.003  # meters, same as pinned upstream PushButtons
RELEASE = 0.001
HOME_XYZ_TOL = 0.012
HOME_ROT_TOL = 0.08  # radians


class PushButtons(StockPushButtons):
    """Keep the class name so RLBench loads the stock push_buttons.ttm."""

    def init_task(self):
        super().init_task()
        self.rest = np.array([joint.get_joint_position() for joint in self.target_joints])
        self.home = np.asarray(self.robot.arm.get_tip().get_pose())
        waypoint_index = 0
        while Object.exists(f"waypoint{waypoint_index}"):
            waypoint_index += 1
        self.home_point = Dummy.create()
        self.home_point.set_name(f"waypoint{waypoint_index}")
        self.home_point.set_parent(self.get_base())
        self.home_point.set_pose(self.home.tolist())
        self.register_waypoint_ability_start(waypoint_index, self._place_home)

    def init_episode(self, index):
        self.order = ORDERS[index]
        self.buttons_to_push = len(self.order)
        self.buttons_pushed = 0
        self.events = []
        self.down = np.zeros(3, dtype=bool)
        self.tick = 0
        self.home_ticks = {}
        self._at_home = False
        boundary = SpawnBoundary([self.boundaries])
        for button, top, wrap, rgb in zip(
            self.target_buttons, self.target_topPlates, self.target_wraps, PALETTE
        ):
            button.set_color(rgb)
            top.set_color(rgb)
            wrap.set_color(rgb)
            boundary.sample(button, min_distance=0.1)
        a, b, c = (COLORS[i] for i in self.order)
        return [f"Press the {a} button, then the {b} button, then the {c} button."]

    def variation_count(self):
        return len(ORDERS)

    def validate(self):
        self.home_point.set_pose(self.home.tolist())
        super().validate()

    def _move_above_next_target(self, waypoint):
        xyz = np.array(self.target_buttons[self.order[self.buttons_pushed]].get_position())
        xyz[2] += 0.083  # stock approach offset
        obj = waypoint.get_waypoint_object()
        obj.set_position(xyz.tolist())
        obj.set_orientation([np.pi, 0.0, np.pi])

    def _place_home(self, waypoint):
        waypoint.get_waypoint_object().set_pose(self.home.tolist())

    def displacement(self):
        return np.abs(
            np.array([joint.get_joint_position() for joint in self.target_joints]) - self.rest
        )

    def step(self):
        self.tick += 1
        displacement = self.displacement()
        rising = np.flatnonzero(~self.down & (displacement >= PRESS))
        self.events.extend((self.tick, int(button)) for button in rising)
        self.down = (self.down & (displacement > RELEASE)) | (displacement >= PRESS)
        pose = np.asarray(self.robot.arm.get_tip().get_pose())
        translation_ok = np.linalg.norm(pose[:3] - self.home[:3]) <= HOME_XYZ_TOL
        rotation_ok = abs(float(pose[3:] @ self.home[3:])) >= np.cos(HOME_ROT_TOL / 2)
        self._at_home = bool(translation_ok and rotation_ok and np.all(displacement <= RELEASE))
        if self._at_home:
            self.home_ticks.setdefault(len(self.events), self.tick)

    def success(self):
        pressed = tuple(button for _, button in self.events)
        simultaneous = any(a[0] == b[0] for a, b in zip(self.events, self.events[1:]))
        if simultaneous or pressed != self.order[:len(pressed)]:
            return False, True
        complete = pressed == self.order and self._at_home
        return complete, complete


def make_environment(image_size):
    config = ObservationConfig()
    config.set_all(False)
    config.front_camera.rgb = True
    config.front_camera.image_size = (image_size, image_size)
    config.gripper_pose = True
    config.gripper_open = True
    config.joint_positions = True
    return Environment(
        MoveArmThenGripper(EndEffectorPoseViaIK(absolute_mode=False), Discrete()),
        obs_config=config,
        headless=True,
    )


def delta_to(current, target, gripper_open):
    delta = Rotation.from_quat(target[3:]) * Rotation.from_quat(current[3:]).inv()
    return np.r_[target[:3] - current[:3], delta.as_euler("xyz"), gripper_open]


def apply_delta(env_task, action):
    quaternion = Rotation.from_euler("xyz", action[3:6]).as_quat()
    return env_task.step(np.r_[action[:3], quaternion, action[6]])


def button_positions(task):
    return np.array([button.get_position() for button in task.target_buttons])


def collect_episode(env_task, variation, stride):
    env_task.set_variation(variation)
    task = env_task._task  # pinned RLBench does not expose the task object's event state
    expert = env_task.get_demos(1, live_demos=True, max_attempts=2)[0]
    layout = button_positions(task)
    descriptions, obs = env_task.reset_to_demo(expert)
    if not np.allclose(button_positions(task), layout, atol=1e-4):
        raise RuntimeError("Expert replay did not restore the original button layout.")

    # Re-track the expert's measured poses through the same EEF controller used
    # to execute our labels. Each label is the command sent after its RGB frame.
    images, poses, closed, actions, ticks, joints = [], [], [], [], [], []

    def record_observation(observation):
        images.append(np.asarray(observation.front_rgb, dtype=np.uint8))
        poses.append(np.asarray(observation.gripper_pose, dtype=np.float32))
        closed.append([1.0 - float(observation.gripper_open)])
        ticks.append(task.tick)
        joints.append(np.asarray(observation.joint_positions, dtype=np.float32))

    record_observation(obs)
    if len(expert) < 2:
        raise RuntimeError("RLBench returned an empty expert trajectory.")
    reference_indices = list(range(1, len(expert), stride))
    if reference_indices[-1] != len(expert) - 1:
        reference_indices.append(len(expert) - 1)
    for i in reference_indices:
        target = expert[i]
        action = delta_to(obs.gripper_pose, target.gripper_pose, float(target.gripper_open))
        obs, reward, done = apply_delta(env_task, action)
        actions.append(action)
        record_observation(obs)
        if done:
            break
    else:
        reward = float(task.success()[0])

    if not reward or any(stage not in task.home_ticks for stage in (1, 2, 3)):
        raise RuntimeError(f"Invalid demonstration: presses={task.events}, home={task.home_ticks}")
    if len(actions) < 16:
        raise RuntimeError("Episode is too short for the proposed 16-frame training groups.")

    home_indices = np.searchsorted(ticks, [task.home_ticks[i] for i in (1, 2, 3)])
    return dict(
        image=np.stack(images),
        pose=np.stack(poses),
        gripper_closed=np.asarray(closed, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        arm_joints=np.stack(joints),
        instruction=np.asarray(descriptions[0]),
        order=np.asarray(task.order),
        press_events=np.asarray(task.events),
        home_indices=np.r_[0, home_indices],
        sim_ticks=np.asarray(ticks),
        layout=layout,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--per-order", type=int, required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-check-frames", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.per_order, args.stride, args.attempts) < 1 or args.seed < 0:
        parser.error("Counts must be positive and the seed must be nonnegative.")
    if any(i not in range(6) for i in args.orders):
        parser.error("Order indices must be in 0..5.")

    output = args.output / args.split
    output.mkdir(parents=True, exist_ok=True)
    settings = (args.seed, args.stride, args.image_size)
    domain = ("train", "val", "test").index(args.split)
    env = make_environment(args.image_size)
    env.launch()
    try:
        env_task = env.get_task(PushButtons)
        for order in args.orders:
            for episode in range(args.per_order):
                name = f"order{order}_episode{episode:05}"
                path = output / f"{name}.npz"
                if path.exists():
                    if not args.resume:
                        raise FileExistsError(f"{path} exists. Use --resume or a fresh directory.")
                    with np.load(path, allow_pickle=False) as saved:
                        if tuple(saved["collection_settings"]) != settings:
                            raise ValueError(f"{path}: collection settings differ from this run.")
                    print(f"Already collected: {path}", flush=True)
                    continue
                for attempt in range(args.attempts):
                    seed = np.random.SeedSequence(
                        [args.seed, domain, order, episode, attempt]
                    ).generate_state(1)[0]
                    np.random.seed(seed)
                    try:
                        record = collect_episode(env_task, order, args.stride)
                        break
                    except (RuntimeError, InvalidActionError, TaskEnvironmentError) as err:
                        print(f"{name} attempt {attempt + 1}: {err}", flush=True)
                else:
                    raise RuntimeError(f"{name} failed after {args.attempts} attempts.")
                record["seed"] = np.asarray(seed)
                record["collection_settings"] = np.asarray(settings)
                with NamedTemporaryFile(dir=output, delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    with tmp_path.open("wb") as stream:
                        np.savez_compressed(stream, **record)
                    os.link(tmp_path, path)
                finally:
                    tmp_path.unlink()
                if args.save_check_frames:
                    frames = record["image"][record["home_indices"]]
                    Image.fromarray(np.concatenate(frames, axis=1)).save(output / f"{name}_home.png")
                    print("home arm-joint max error (rad):",
                          np.max(np.abs(record["arm_joints"][record["home_indices"]]
                                        - record["arm_joints"][0]), axis=1), flush=True)
                print(f"{path}: {len(record['action'])} actions, presses {record['press_events'].tolist()}",
                      flush=True)
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
