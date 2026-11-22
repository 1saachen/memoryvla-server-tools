"""Evaluate the MIKASA MemoryVLA checkpoint on the legacy ShellGameTouch task."""

import argparse
import io
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import requests
from PIL import Image


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


def request_actions(session, url, rgb, first_frame):
    png = io.BytesIO()
    Image.fromarray(rgb).save(png, format="PNG")
    response = session.post(
        url,
        files={"image": ("frame.png", png.getvalue(), "image/png")},
        data={"text": PROMPT, "episode_first_frame": str(first_frame)},
        timeout=(10, 180),
    )
    response.raise_for_status()
    actions = np.asarray(
        [row.split() for row in response.json()["response"].split(";")],
        dtype=np.float32,
    )
    if actions.shape != (CHUNK_SIZE, 7) or not np.isfinite(actions).all():
        raise ValueError(f"Expected {CHUNK_SIZE} finite 7D actions, got {actions.shape}")
    return actions


def rollout(env, policy, seed, video):
    obs, _ = env.reset(seed=seed)
    rgb = rgb_frame(obs)
    if video is not None:
        video.append_data(rgb)
    steps = 0
    while steps < MAX_STEPS:
        for action in policy(rgb, steps == 0):
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            rgb = rgb_frame(obs)
            if video is not None:
                video.append_data(rgb)
            success = steps >= CUE_STEPS and bool(info["success"])
            if success or bool(terminated) or bool(truncated) or steps == MAX_STEPS:
                return {"seed": seed, "steps": steps, "success": success}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8052/process_frame")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-backend", default="gpu", help="ManiSkill render device: gpu, cuda:N, or cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    import gymnasium as gym
    import imageio.v2 as imageio
    import mikasa_robo_suite  # Registers the legacy tasks.

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": TASK, "instruction": PROMPT, "seed_start": args.seed,
        "num_eval": args.episodes, "max_steps": MAX_STEPS, "action_scale": 1.0,
        "action_chunking_window": CHUNK_SIZE, "control_mode": CONTROL_MODE,
        "camera": "base_camera", "sim_backend": "cpu", "render_backend": args.render_backend,
        "url": args.url,
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
            def policy(rgb, first_frame):
                return request_actions(session, args.url, rgb, first_frame)

            successes = 0
            for episode in range(args.episodes):
                if episode == 0:
                    with imageio.get_writer(args.out / "episode_000.mp4", fps=20) as video:
                        result = rollout(env, policy, args.seed + episode, video)
                else:
                    result = rollout(env, policy, args.seed + episode, None)
                successes += int(result["success"])
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(f"{episode + 1}/{args.episodes} {result} running_success={successes}/{episode + 1}", flush=True)
        summary = {**manifest, "successes": successes, "success_rate": successes / args.episodes}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"EVAL_COMPLETE {successes}/{args.episodes} ({summary['success_rate']:.1%})", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
