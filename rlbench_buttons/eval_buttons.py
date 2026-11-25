"""Evaluate a MemoryVLA policy on the saved RLBench button test layouts."""

import argparse
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
MAX_ACTIONS = 800
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


def policy_worker(checkpoint):
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
            model = load_vla(
                str(checkpoint), hf_token=None, load_for_training=False,
                **{key: config[key] for key in keys if key in config},
            ).to(device="cuda", dtype=torch.float32).eval()
            model.get_action_stats(UNNORM_KEY)
            print(json.dumps({"ready": True}), file=responses)
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
                        or len(actions) < CHUNK_STEPS or not np.isfinite(actions).all()):
                    raise ValueError(f"Expected a finite action chunk with shape (N, 7), got {actions.shape}")
                print(json.dumps(actions[:CHUNK_STEPS].tolist()), file=responses)


def read_policy(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("MemoryVLA stopped responding. See the evaluation log for the traceback.")
    return json.loads(line)


def evaluate(args):
    import base64
    import io
    import numpy as np
    from PIL import Image
    from rlbench.backend.exceptions import InvalidActionError
    from collect import PushButtons as CollectedPushButtons
    from collect import apply_delta, button_positions, make_environment
    from demo_reset import reset_demo

    class PushButtons(CollectedPushButtons):
        # Keep the class name for RLBench's asset lookup. During the policy
        # rollout, allow all three scored positions even after an early error.
        scoring_rollout = False

        def success(self):
            success, done = super().success()
            if self.scoring_rollout and len(self.events) < 3:
                return False, False
            return success, done

    test_dir = args.data / "test"
    cases = [(order, idx, test_dir / f"order{order}_episode{idx:05}.npz")
             for order in range(len(ORDERS)) for idx in range(args.per_order)]
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
                     "--policy-worker", str(args.checkpoint)]
    rows = []
    with (args.output / "episodes.jsonl").open("x", encoding="utf-8") as episodes:
        with subprocess.Popen(
            model_command, cwd=args.model_repo, env=policy_env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
        ) as policy:
            try:
                if read_policy(policy) != {"ready": True}:
                    raise RuntimeError("Unexpected MemoryVLA startup response")
                env = make_environment(224)
                env.launch()
                try:
                    env_task = env.get_task(PushButtons)
                    task = env_task._task
                    for order, idx, path in cases:
                        expected_order = ORDERS[order]
                        with np.load(path, allow_pickle=False) as demo_file:
                            seed = int(demo_file["seed"].item())
                            layout = demo_file["layout"]
                            saved_order = tuple(int(button) for button in demo_file["order"])
                            saved_instruction = str(demo_file["instruction"].item())
                        if saved_order != expected_order:
                            raise RuntimeError(
                                f"{path.name}: saved order {saved_order} does not match {expected_order}"
                            )
                        print(f"Starting {path.stem}, layout seed {seed}", flush=True)
                        # Reproduce the collector's reset, including any planner retry.
                        np.random.seed(seed)
                        env_task.set_variation(order)
                        task.scoring_rollout = False
                        demo = env_task.get_demos(1, live_demos=True, max_attempts=2)[0]
                        descriptions, obs = reset_demo(env_task, demo, order, expected_order)
                        del demo
                        if not np.allclose(button_positions(task), layout, atol=1e-4, rtol=0):
                            raise RuntimeError(f"{path.name}: saved test layout could not be restored")
                        if descriptions[0] != saved_instruction:
                            raise RuntimeError(f"{path.name}: saved instruction did not match after reset")
                        print(f"Verified {path.stem}: {descriptions[0]}", flush=True)
                        task.scoring_rollout = True

                        reward, done, steps = 0, False, 0
                        error = None
                        policy_seed = POLICY_SEED + order * 20 + idx
                        while steps < MAX_ACTIONS and not done:
                            with io.BytesIO() as image:
                                rgb = np.asarray(obs.front_rgb, dtype=np.uint8)
                                Image.fromarray(rgb).save(image, format="PNG", compress_level=1)
                                request = {
                                    "image": base64.b64encode(image.getvalue()).decode("ascii"),
                                    "instruction": descriptions[0],
                                    "first_frame": steps == 0,
                                    "policy_seed": policy_seed,
                                }
                            policy.stdin.write(json.dumps(request) + "\n")
                            actions = read_policy(policy)
                            for action in actions[:MAX_ACTIONS - steps]:
                                steps += 1
                                try:
                                    obs, reward, done = apply_delta(env_task, action)
                                except InvalidActionError as exc:
                                    error = str(exc)
                                    done = True
                                if done:
                                    break
                        row = {
                            "episode_id": path.stem, "order_id": order,
                            "target": [COLORS[i] for i in expected_order],
                            "press_events": [[t, COLORS[i]] for t, i in task.events],
                            "layout_seed": seed, "policy_seed": policy_seed,
                            "steps": steps, "success": bool(reward),
                            "score": score_presses(expected_order, task.events),
                            "end_reason": (
                                "success" if reward else "invalid_action" if error is not None
                                else "wrong_or_simultaneous_press" if done else "action_limit"
                            ),
                            "error": error,
                        }
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
        "unnorm_key": UNNORM_KEY, "chunk_steps": CHUNK_STEPS,
        "max_actions": MAX_ACTIONS, "policy_seed_base": POLICY_SEED,
        "cfg_scale": CFG_SCALE, "ddim_steps": DDIM_STEPS, "precision": "fp32",
        "press_detection": "RLBench button joint displacement >= 0.003 m",
        "scoring": "30 per position matched by a distinct-time press, plus 10 for all three",
        **summarize(rows),
        "per_order": {str(order): summarize([r for r in rows if r["order_id"] == order])
                      for order in range(len(ORDERS))},
    }
    summary_file = args.output / "summary.json"
    temporary = summary_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_file)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--policy-worker":
        policy_worker(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="NPZ root containing test/")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--model-gpu", required=True, help="CUDA device UUID")
    parser.add_argument("--per-order", type=int, choices=range(1, 21), default=20, metavar="1..20")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.model_repo = args.model_repo.resolve()
    evaluate(args)


if __name__ == "__main__":
    main()
