"""Evaluate a MemoryVLA policy on the saved RLBench button test layouts."""

import argparse
from bisect import bisect_left
import contextlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


UNNORM_KEY = "custom_finetuning"
COLORS = ("blue", "green", "pink")
ORDERS = tuple(itertools.permutations(range(3)))
CHUNK_STEPS = 4
MAX_CHUNK_STEPS = 16
MAX_ACTIONS = 800
HANDOFF_LEAD = 5
HANDOFF_MAX_ACTIONS = 20
POLICY_SEED = 7
CFG_SCALE = 1.5
DDIM_STEPS = 10


def score_presses(order, events):
    """Apply the paper's positional points to recorded simulator presses."""
    tick_counts = {}
    for tick, _ in events:
        tick_counts[tick] = tick_counts.get(tick, 0) + 1
    correct = sum(actual == target and tick_counts[tick] == 1
                  for (tick, actual), target in zip(events, order))
    return 30 * correct + (10 if correct == 3 else 0)


def summarize(rows):
    total = len(rows)
    successes = sum(row["success"] for row in rows)
    score_total = sum(row["score"] for row in rows)
    return {
        "successes": successes,
        "episodes": total,
        "success_rate_pct": 100 * successes / total,
        "score_total": score_total,
        "mean_score": score_total / total,
    }


def policy_worker(checkpoint, chunk_steps, mem_length):
    # Keep the JSON pipe separate from logs from Python and native libraries.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as responses:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with contextlib.redirect_stdout(sys.stderr):
            import base64
            import io
            import numpy as np
            from PIL import Image
            import torch
            from vla import load_vla

            config = json.loads((checkpoint.parents[1] / "config.json").read_text())
            keys = (
                "action_dim", "action_model_type", "future_action_window_size",
                "use_ema", "dataloader_type", "group_size", "per_token_size",
                "mem_length", "retrieval_layers", "use_timestep_pe", "fusion_type",
                "consolidate_type", "update_fused",
            )
            model_config = {key: config[key] for key in keys if key in config}
            if mem_length is not None:
                model_config["mem_length"] = mem_length
            model = load_vla(
                str(checkpoint), hf_token=None, load_for_training=False,
                **model_config,
            ).to(device="cuda", dtype=torch.float32).eval()
            model.get_action_stats(UNNORM_KEY)
            print(json.dumps({"ready": True, "mem_length": model.mem_length}), file=responses)
            for line in sys.stdin:
                request = json.loads(line)
                first = request["first_frame"]
                if first:
                    torch.manual_seed(request["policy_seed"])
                with Image.open(io.BytesIO(base64.b64decode(request["image"]))) as image:
                    actions, _ = model.predict_action(
                        image=image.convert("RGB"), instruction=request["instruction"],
                        unnorm_key=UNNORM_KEY, cfg_scale=CFG_SCALE, use_ddim=True,
                        num_ddim_steps=DDIM_STEPS,
                        episode_first_frame="True" if first else "False",
                    )
                if (actions.ndim != 2 or actions.shape[1] != 7
                        or len(actions) < chunk_steps or not np.isfinite(actions).all()):
                    raise ValueError(f"Expected a finite action chunk with shape (N, 7), got {actions.shape}")
                print(json.dumps(actions[:chunk_steps].tolist()), file=responses)


def read_policy(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("MemoryVLA stopped responding. See the evaluation log for the traceback.")
    return json.loads(line)


def expert_prefix_steps(ticks, first_press_tick):
    # Observation i precedes action i, and the press is seen by observation i + 1.
    press_frame = bisect_left(ticks, first_press_tick)
    if not HANDOFF_LEAD <= press_frame < len(ticks):
        raise ValueError(f"Cannot start {HANDOFF_LEAD} actions before press tick {first_press_tick}")
    return press_frame - HANDOFF_LEAD


def rank_layouts_near_start(test_dir, order):
    import numpy as np

    ranked = []
    first_button = ORDERS[order][0]
    pattern = f"order{order}_episode" + "[0-9]" * 5 + ".npz"
    for path in sorted(test_dir.glob(pattern)):
        with np.load(path, allow_pickle=False) as demo:
            distances_mm = 1000 * np.linalg.norm(demo["layout"][:, :2] - demo["pose"][0, :2], axis=1)
        if not np.isfinite(distances_mm).all():
            raise ValueError(f"{path}: non-finite button distances")
        ranked.append((path, float(distances_mm.max()), float(distances_mm[first_button])))
    return sorted(ranked, key=lambda entry: entry[1])


def request_actions(policy, obs, instruction, first_frame, policy_seed):
    import base64
    import io
    import numpy as np
    from PIL import Image

    with io.BytesIO() as image:
        rgb = np.asarray(obs.front_rgb, dtype=np.uint8)
        Image.fromarray(rgb).save(image, format="PNG", compress_level=1)
        request = {
            "image": base64.b64encode(image.getvalue()).decode("ascii"),
            "instruction": instruction, "first_frame": first_frame, "policy_seed": policy_seed,
        }
    policy.stdin.write(json.dumps(request) + "\n")
    return read_policy(policy)


def evaluate(args):
    import numpy as np
    from rlbench.backend.exceptions import InvalidActionError
    from collect import PushButtons as CollectedPushButtons
    from collect import apply_delta, button_positions, delta_to, make_environment
    from demo_reset import reset_demo
    from instructions import paper_instruction

    class PushButtons(CollectedPushButtons):
        # Keep the class name for RLBench's asset lookup. During the policy
        # rollout, allow all three scored positions even after an early error.
        scoring_rollout = False
        peak_displacement = None

        def step(self):
            super().step()
            if self.peak_displacement is not None:
                np.maximum(self.peak_displacement, self.displacement(), out=self.peak_displacement)

        def success(self):
            success, done = super().success()
            if self.scoring_rollout and len(self.events) < 3:
                return False, False
            return success, done

    test_dir = args.data / "test"
    selected_orders = range(len(ORDERS)) if args.order is None else (args.order,)
    near_layouts = {}
    cases = []
    for order in selected_orders:
        if args.near_start:
            ranked = rank_layouts_near_start(test_dir, order)
            baseline = next((entry for entry in ranked if entry[0].stem.endswith("episode00000")), None)
            if len(ranked) < args.per_order or baseline is None:
                raise FileNotFoundError(f"Order {order} needs episode 00000 and at least {args.per_order} layouts")
            for idx, (path, radius, first_distance) in enumerate(ranked[:args.per_order]):
                near_layouts[path] = {
                    "start_radius_mm": round(radius, 1),
                    "first_button_start_xy_mm": round(first_distance, 1),
                    "episode0_start_radius_mm": round(baseline[1], 1),
                    "episode0_first_button_start_xy_mm": round(baseline[2], 1),
                }
                print(f"Selected {path.stem}: {json.dumps(near_layouts[path])}", flush=True)
                # Match the policy seed of the corresponding unsorted rollout.
                cases.append((order, idx, path))
        else:
            cases.extend((order, idx, test_dir / f"order{order}_episode{idx:05}.npz")
                         for idx in range(args.per_order))
    for _, _, path in cases:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    args.output.mkdir(parents=True, exist_ok=True)
    policy_env = {key: value for key, value in os.environ.items()
                  if key != "LD_PRELOAD" and not key.startswith("VGL_")}
    policy_env["CUDA_VISIBLE_DEVICES"] = args.model_gpu
    policy_env["TOKENIZERS_PARALLELISM"] = "false"
    policy_env["PYTHONPATH"] = str(args.model_repo) + (
        os.pathsep + policy_env["PYTHONPATH"] if policy_env.get("PYTHONPATH") else ""
    )
    model_command = [str(args.model_python), "-u", str(Path(__file__).resolve()),
                     "--policy-worker", str(args.checkpoint), str(args.chunk_steps),
                     str(args.mem_length or 0)]
    rows = []
    max_actions = HANDOFF_MAX_ACTIONS if args.first_press_handoff else MAX_ACTIONS
    close_gripper = args.close_gripper or args.handoff_close_gripper
    gripper_control = "closed" if close_gripper else "policy"
    with contextlib.ExitStack() as files:
        episodes = files.enter_context((args.output / "episodes.jsonl").open("x", encoding="utf-8"))
        trace = (files.enter_context((args.output / "trajectory.jsonl").open("x", encoding="utf-8"))
                 if args.trace else None)
        with subprocess.Popen(
            model_command, cwd=args.model_repo, env=policy_env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
        ) as policy:
            try:
                model_info = read_policy(policy)
                if model_info.get("ready") is not True:
                    raise RuntimeError("Unexpected MemoryVLA startup response")
                mem_length = model_info["mem_length"]
                if args.mem_length is not None and mem_length != args.mem_length:
                    raise RuntimeError(f"Expected memory capacity {args.mem_length}, got {mem_length}")
                print(f"Memory capacity: {mem_length}; actions per inference: {args.chunk_steps}", flush=True)
                env = make_environment(224)
                env.launch()
                try:
                    env_task = env.get_task(PushButtons)
                    task = env_task._task
                    for order, idx, path in cases:
                        expected_order = ORDERS[order]
                        prefix_steps = 0
                        with np.load(path, allow_pickle=False) as demo_file:
                            seed = int(demo_file["seed"].item())
                            layout = demo_file["layout"]
                            saved_order = tuple(int(button) for button in demo_file["order"])
                            saved_instruction = str(demo_file["instruction"].item())
                            if args.first_press_handoff:
                                prefix_steps = expert_prefix_steps(
                                    demo_file["sim_ticks"], int(demo_file["press_events"][0, 0]),
                                )
                                prefix_poses = demo_file["pose"][:prefix_steps + 1]
                                prefix_grippers = demo_file["action"][:prefix_steps, 6]
                        if saved_order != expected_order:
                            raise RuntimeError(
                                f"{path.name}: saved order {saved_order} does not match {expected_order}"
                            )
                        print(f"Starting {path.stem}, layout seed {seed}", flush=True)
                        # Reproduce the collector's reset, including any planner retry.
                        np.random.seed(seed)
                        env_task.set_variation(order)
                        task.scoring_rollout = False
                        task.peak_displacement = None
                        demo = env_task.get_demos(1, live_demos=True, max_attempts=2)[0]
                        descriptions, obs = reset_demo(env_task, demo, order, expected_order)
                        del demo
                        if not np.allclose(button_positions(task), layout, atol=1e-4, rtol=0):
                            raise RuntimeError(f"{path.name}: saved test layout could not be restored")
                        if descriptions[0] != saved_instruction:
                            raise RuntimeError(f"{path.name}: saved instruction did not match after reset")
                        print(f"Verified {path.stem}: {descriptions[0]}", flush=True)
                        instruction = (paper_instruction(expected_order)
                                       if args.instruction_style == "paper" else saved_instruction)
                        print(f"Policy instruction {path.stem}: {instruction}", flush=True)
                        reward, done, steps = 0, False, 0
                        error = None
                        policy_seed = POLICY_SEED + order * 20 + idx
                        if args.first_press_handoff:
                            print(f"Replaying {prefix_steps} expert actions with policy observations", flush=True)
                            for index, (target, gripper) in enumerate(zip(prefix_poses[1:], prefix_grippers)):
                                request_actions(policy, obs, instruction, index == 0, policy_seed)
                                expert_action = delta_to(obs.gripper_pose, target, gripper)
                                obs, _, expert_done = apply_delta(env_task, expert_action)
                                if task.events:
                                    raise RuntimeError(f"{path.name}: expert pressed before handoff at action {index + 1}")
                                if expert_done:
                                    raise RuntimeError(f"{path.name}: expert rollout ended before handoff at action {index + 1}")
                            if task.events:
                                raise RuntimeError(f"{path.name}: button already pressed at handoff")
                            task.peak_displacement = task.displacement()
                            handoff_pose_error_mm = float(1000 * np.linalg.norm(
                                obs.gripper_pose[:3] - prefix_poses[-1, :3],
                            ))
                            print(f"Model handoff after {prefix_steps} expert actions", flush=True)
                        task.scoring_rollout = True
                        while steps < max_actions and not done:
                            actions = request_actions(policy, obs, instruction,
                                                      prefix_steps == 0 and steps == 0, policy_seed)
                            for action in actions[:min(args.chunk_steps, max_actions - steps)]:
                                steps += 1
                                executed_action = action[:6] + [0.0] if close_gripper else action
                                if trace is not None:
                                    before = np.asarray(obs.gripper_pose).copy()
                                    joints = np.asarray(obs.joint_positions).copy()
                                    target_position = button_positions(task)[expected_order[0]]
                                    tick_before = task.tick
                                try:
                                    obs, reward, done = apply_delta(env_task, executed_action)
                                except InvalidActionError as exc:
                                    error = str(exc)
                                    done = True
                                if args.first_press_handoff and task.events:
                                    done = True
                                if trace is not None:
                                    print(json.dumps({
                                        "episode_id": path.stem, "step": steps,
                                        "sim_tick_before": tick_before,
                                        "eef_pose_before": before.tolist(),
                                        "joint_positions_before": joints.tolist(),
                                        "predicted_action": action,
                                        "executed_gripper": executed_action[6],
                                        "commanded_xyz": (before[:3] + action[:3]).tolist(),
                                        "eef_pose_after": (list(map(float, obs.gripper_pose))
                                                           if error is None else None),
                                        "first_button_xyz": target_position.tolist(),
                                        "first_button_xy_distance_mm": float(
                                            1000 * np.linalg.norm(before[:2] - target_position[:2])
                                        ),
                                        "button_displacement_mm": (1000 * task.displacement()).tolist(),
                                        "error": error,
                                    }), file=trace, flush=True)
                                if done:
                                    break
                        row = {
                            "episode_id": path.stem, "order_id": order,
                            "target": [COLORS[i] for i in expected_order],
                            "instruction": instruction,
                            "gripper_control": gripper_control,
                            "press_events": [[t, COLORS[i]] for t, i in task.events],
                            "layout_seed": seed, "policy_seed": policy_seed,
                            "steps": steps, "error": error,
                        }
                        if args.first_press_handoff:
                            first_button = expected_order[0]
                            row.update(
                                expert_steps=prefix_steps,
                                handoff_gripper=gripper_control,
                                handoff_pose_error_mm=handoff_pose_error_mm,
                                first_button_correct=score_presses((first_button,), task.events) == 30,
                                first_button_peak_mm=float(1000 * task.peak_displacement[first_button]),
                                end_reason=(
                                    "invalid_action" if error is not None else "button_press" if task.events
                                    else "action_limit"
                                ),
                            )
                        else:
                            row.update(
                                success=bool(reward),
                                score=score_presses(expected_order, task.events),
                                end_reason=(
                                    "success" if reward else "invalid_action" if error is not None
                                    else "wrong_or_simultaneous_press" if done else "action_limit"
                                ),
                            )
                        if args.near_start:
                            row.update(near_layouts[path])
                            row["first_button_correct"] = score_presses(expected_order[:1], task.events) == 30
                        rows.append(row)
                        line = json.dumps(row)
                        print(line, file=episodes, flush=True)
                        print(line, flush=True)
                finally:
                    env.shutdown()
            finally:
                policy.stdin.close()
                if policy.poll() is None:
                    policy.terminate()
    summary = {
        "checkpoint": str(args.checkpoint), "split": "test",
        "unnorm_key": UNNORM_KEY, "chunk_steps": args.chunk_steps,
        "mem_length": mem_length, "instruction_style": args.instruction_style,
        "gripper_control": gripper_control,
        "max_actions": max_actions, "policy_seed_base": POLICY_SEED,
        "cfg_scale": CFG_SCALE, "ddim_steps": DDIM_STEPS, "precision": "fp32",
        "press_detection": "RLBench button joint displacement >= 0.003 m",
    }
    if args.first_press_handoff:
        summary.update(
            mode="first_press_handoff",
            handoff_gripper=gripper_control,
            handoff_actions_before_press=HANDOFF_LEAD,
            first_button_correct=sum(row["first_button_correct"] for row in rows),
            episodes=len(rows),
        )
    else:
        summary.update(summarize(rows))
        summary["scoring"] = "30 per position matched by a distinct-time press, plus 10 for all three"
        summary["per_order"] = {str(order): summarize([r for r in rows if r["order_id"] == order])
                                for order in selected_orders}
    if args.near_start:
        summary.update(layout_selection="near_start",
                       first_button_correct=sum(row["first_button_correct"] for row in rows))
    summary_file = args.output / "summary.json"
    temporary = summary_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_file)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--policy-worker":
        policy_worker(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) or None)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="NPZ root containing test/")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--model-gpu", required=True, help="CUDA device UUID")
    parser.add_argument("--per-order", type=int, choices=range(1, 21), default=20, metavar="1..20")
    parser.add_argument("--order", type=int, choices=range(len(ORDERS)), help="Evaluate one button order")
    parser.add_argument("--near-start", action="store_true",
                        help="Select test layouts by the farthest button's XY distance from the start")
    parser.add_argument("--trace", action="store_true", help="Record per-action poses and model commands")
    parser.add_argument("--close-gripper", action="store_true",
                        help="Command a closed gripper whenever the policy controls the arm")
    parser.add_argument("--first-press-handoff", action="store_true",
                        help="Replay the expert up to five actions before the first press, then test the policy")
    parser.add_argument("--handoff-close-gripper", action="store_true",
                        help="Command a closed gripper during policy control in the handoff test")
    parser.add_argument("--chunk-steps", type=int, default=CHUNK_STEPS, metavar="1..16",
                        help="Actions executed before the next policy observation")
    parser.add_argument("--mem-length", type=int, help="Override checkpoint memory capacity for inference")
    parser.add_argument("--instruction-style", choices=("saved", "paper"), default="saved")
    args = parser.parse_args()
    if not 1 <= args.chunk_steps <= MAX_CHUNK_STEPS:
        parser.error(f"--chunk-steps must be between 1 and {MAX_CHUNK_STEPS}")
    if args.first_press_handoff and args.chunk_steps != 1:
        parser.error("--first-press-handoff requires --chunk-steps 1")
    if args.handoff_close_gripper and not args.first_press_handoff:
        parser.error("--handoff-close-gripper requires --first-press-handoff")
    if args.mem_length is not None and args.mem_length < 1:
        parser.error("--mem-length must be positive")
    args.checkpoint = args.checkpoint.resolve()
    args.model_repo = args.model_repo.resolve()
    evaluate(args)


if __name__ == "__main__":
    main()
