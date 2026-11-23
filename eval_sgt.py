"""Evaluate the MIKASA MemoryVLA checkpoint on the legacy ShellGameTouch task."""

import argparse
import hashlib
import io
import json
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from sgt_contacts import CupTracker
from sgt_metrics import summarize_pairs, summarize_results


TASK = "ShellGameTouch-v0"
PROMPT = "Memorize the position of the cup covering the ball, then pick that cup"
CONTROL_MODE = "pd_ee_delta_pose"
CHUNK_SIZE = 4
MAX_STEPS = 90
CUE_STEPS = 5


def rgb_frame(obs):
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Expected an RGB uint8 image, got {rgb.shape} {rgb.dtype}")
    return rgb


def request_actions(session, url, rgb, first_frame, *, episode_seed=None, clear_memory=False):
    png = io.BytesIO()
    Image.fromarray(rgb).save(png, format="PNG")
    data = {"text": PROMPT, "episode_first_frame": str(first_frame)}
    if episode_seed is not None:
        data["episode_seed"] = str(episode_seed)
    if clear_memory:
        data["clear_memory"] = "True"
    response = session.post(
        url,
        files={"image": ("frame.png", png.getvalue(), "image/png")},
        data=data,
        timeout=(10, 180),
    )
    response.raise_for_status()
    payload = response.json()
    if episode_seed is not None and payload.get("episode_seed") != episode_seed:
        raise RuntimeError("SGT server did not confirm the episode seed")
    if clear_memory and payload.get("memory_cleared") is not True:
        raise RuntimeError("SGT server did not confirm the memory clear")
    rows = [row.split() for row in payload["response"].split(";")]
    if len(rows) < CHUNK_SIZE or any(len(row) != 7 for row in rows):
        raise ValueError(
            f"Expected at least {CHUNK_SIZE} rows of 7 values from {url}, "
            f"got {len(rows)} rows with widths {[len(row) for row in rows]}"
        )
    actions = np.asarray(rows, dtype=np.float32)
    invalid = ~np.isfinite(actions)
    if invalid.any():
        raise ValueError(
            f"Non-finite actions from {url}: {actions[invalid].tolist()} "
            f"at zero-based (step, axis) indices {np.argwhere(invalid).tolist()}"
        )
    # Replan after four simulator steps even if the service returns more actions.
    return actions[:CHUNK_SIZE]


def rollout(env, policy, seed, video, memory_condition=None):
    obs, _ = env.reset(seed=seed)
    cups = CupTracker(env)
    rgb = rgb_frame(obs)
    if video is not None:
        video.append_data(rgb)
    steps = 0
    clear_step = None
    # Compare the observations and executed actions shared by both arms before clearing.
    trace = hashlib.sha256()
    reached_post_cue_request = False
    while steps < MAX_STEPS:
        if not reached_post_cue_request:
            trace.update(rgb.tobytes())
        post_cue_request = not reached_post_cue_request and steps >= CUE_STEPS
        clear_now = memory_condition == "clear" and post_cue_request
        actions = policy(
            rgb, steps == 0,
            episode_seed=seed if memory_condition is not None and steps == 0 else None,
            clear_memory=clear_now,
        )
        if clear_now:
            clear_step = steps
        reached_post_cue_request |= post_cue_request
        for action in actions:
            if not reached_post_cue_request:
                trace.update(action.tobytes())
            with cups.track_contacts(steps + 1):
                obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            rgb = rgb_frame(obs)
            if video is not None:
                video.append_data(rgb)
            if steps >= CUE_STEPS:
                cups.observe(steps)
            success = steps >= CUE_STEPS and bool(info["success"])
            if success or bool(terminated) or bool(truncated) or steps == MAX_STEPS:
                return {
                    "seed": seed, "steps": steps, "success": success, **cups.as_result(),
                    "memory_condition": memory_condition or "keep",
                    "memory_clear_step": clear_step,
                    "first_post_cue_request_reached": reached_post_cue_request,
                    "pre_clear_trace_sha256": trace.hexdigest(),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8052/process_frame")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-backend", default="gpu", help="ManiSkill render device: gpu, cuda:N, or cpu")
    parser.add_argument("--paired-memory-ablation", action="store_true",
                        help="Rerun every seed with intact memory and a one-time post-cue clear")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.paired_memory_ablation and not 0 <= args.seed <= args.seed + args.episodes - 1 < 2**32:
        parser.error("paired evaluation requires episode seeds in [0, 2**32)")

    import gymnasium as gym
    import imageio.v2 as imageio
    import mikasa_robo_suite  # Registers the legacy tasks.

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": TASK, "instruction": PROMPT, "seed_start": args.seed,
        "num_eval": args.episodes, "max_steps": MAX_STEPS, "action_scale": 1.0,
        "action_chunking_window": CHUNK_SIZE, "control_mode": CONTROL_MODE,
        "camera": "base_camera", "sim_backend": "cpu", "render_backend": args.render_backend,
        "url": args.url, "cue_steps": CUE_STEPS, "paired_memory_ablation": args.paired_memory_ablation,
        "versions": {name: version(name) for name in ("mikasa-robo-suite", "mani-skill", "sapien")},
    }
    with (args.out / "config.json").open("x", encoding="utf-8") as config_file:
        json.dump(manifest, config_file, indent=2)
    print(f"Initializing {TASK} with CPU physics and render_backend={args.render_backend}", flush=True)
    env = gym.make(
        TASK, num_envs=1, obs_mode="rgb", control_mode=CONTROL_MODE,
        sim_backend="cpu", render_backend=args.render_backend, reward_mode="dense",
    )
    print("SGT environment ready; starting rollouts", flush=True)
    try:
        with requests.Session() as session, (args.out / "episodes.jsonl").open("x", encoding="utf-8") as log:
            def policy(rgb, first_frame, **options):
                return request_actions(session, args.url, rgb, first_frame, **options)

            groups = ("keep", "clear") if args.paired_memory_ablation else (None,)
            results = {group: [] for group in groups}
            for episode in range(args.episodes):
                for group in groups:
                    suffix = f"_{group}" if group else ""
                    clip = args.out / f"episode_000{suffix}.mp4"
                    writer = imageio.get_writer(clip, fps=20) if episode == 0 else nullcontext()
                    with writer as video:
                        result = rollout(env, policy, args.seed + episode, video, group)
                    results[group].append(result)
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                    print(f"{episode + 1}/{args.episodes} {result}", flush=True)

        summary = dict(manifest)
        if args.paired_memory_ablation:
            summary.update({group: summarize_results(results[group]) for group in groups})
            summary["paired"] = summarize_pairs(results["keep"], results["clear"])
        else:
            summary.update(summarize_results(results[None]))
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if args.paired_memory_ablation:
            for group in groups:
                counts = summary[group]
                print(f"{group}: official={counts['successes']}/{args.episodes}, "
                      f"first_touch={counts['first_touch_successes']}/{args.episodes}, "
                      f"first_reach={counts['first_reach_successes']}/{args.episodes}", flush=True)
            print(f"EVAL_COMPLETE {args.episodes} seed pair(s)", flush=True)
        else:
            print(f"EVAL_COMPLETE {summary['successes']}/{args.episodes} ({summary['success_rate']:.1%})", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
