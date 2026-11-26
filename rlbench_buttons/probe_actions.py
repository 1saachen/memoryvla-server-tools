"""Compare first-button actions on recorded frames, without running the simulator."""

import argparse
import base64
import io
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from eval_buttons import COLORS, read_policy


def predict_frames(policy, frames, instruction, seed):
    predictions = []
    for i, frame in enumerate(frames):
        with io.BytesIO() as image:
            Image.fromarray(frame).save(image, format="PNG", compress_level=1)
            request = {
                "image": base64.b64encode(image.getvalue()).decode("ascii"),
                "instruction": instruction, "first_frame": i == 0, "policy_seed": seed,
            }
        policy.stdin.write(json.dumps(request) + "\n")
        predictions.append(read_policy(policy)[0])
    return np.asarray(predictions)


def probe(policy, path, seed):
    with np.load(path, allow_pickle=False) as episode:
        instruction = str(episode["instruction"].item())
        frames = episode["image"]
        expert = episode["action"].astype(np.float64)
        ticks = episode["sim_ticks"]
        first_press = int(episode["press_events"][0, 0])
    # Action i is sent after frame i. Include the action that reaches the press tick.
    stop = int(np.searchsorted(ticks, first_press, side="left"))
    if not 0 < stop <= len(expert) or len(frames) != len(expert) + 1:
        raise ValueError(f"Invalid first-press alignment: {path}")
    print(f"Probing {path.parent.name}/{path.stem}: {instruction} ({stop} frames)", flush=True)
    return expert[:stop], predict_frames(policy, frames[:stop], instruction, seed)


def compare_first_colors(policy, path, seed):
    with np.load(path, allow_pickle=False) as episode:
        ticks = episode["sim_ticks"]
        stop = int(np.searchsorted(ticks, episode["press_events"][0, 0]))
        frames = episode["image"][:min(stop, 40)]
        layout_xy = episode["layout"][:, :2]
        original_order = [int(index) for index in episode["order"]]
    if not 0 < stop < len(ticks) or len(frames) != min(stop, 40):
        raise ValueError(f"Invalid first-press alignment: {path}")

    print(f"Comparing first colors on {path.parent.name}/{path.stem} ({len(frames)} frames)", flush=True)
    sums_mm = {}
    for first in range(len(COLORS)):
        second, third = (index for index in original_order if index != first)
        a, b, c = (COLORS[index] for index in (first, second, third))
        instruction = f"Press the {a} button, then the {b} button, then the {c} button."
        actions = predict_frames(policy, frames, instruction, seed)
        sums_mm[first] = 1000 * actions[:, :2].sum(axis=0)

    toward_target_mm = {}
    for a, b in combinations(range(len(COLORS)), 2):
        separation = layout_xy[a] - layout_xy[b]
        distance = np.linalg.norm(separation)
        if not np.isfinite(distance) or distance == 0:
            raise ValueError(f"Invalid horizontal button separation in {path}")
        # A positive projection means the two prompts move the commands in the expected relative direction.
        projected = np.dot(sums_mm[a] - sums_mm[b], separation / distance)
        toward_target_mm[f"{COLORS[a]}_vs_{COLORS[b]}"] = round(float(projected), 2)
    return {
        "split": path.parent.name,
        "episode_id": path.stem,
        "frames": len(frames),
        "xy_cmd_sum_mm": {COLORS[index]: value.round(2).tolist() for index, value in sums_mm.items()},
        "pairwise_toward_target_mm": toward_target_mm,
    }


def summarize_actions(expert, predicted):
    xyz_error = np.linalg.norm(predicted[:, :3] - expert[:, :3], axis=1)
    # Compare the actual commands around contact as well as the approach.
    expert_contact = expert[-5:, 2]
    predicted_contact = predicted[-5:, 2]
    return {
        "frames": len(expert),
        "median_xyz_error_mm": float(1000 * np.median(xyz_error)),
        "expert_xyz_sum_mm": (1000 * expert[:, :3].sum(axis=0)).round(3).tolist(),
        "predicted_xyz_sum_mm": (1000 * predicted[:, :3].sum(axis=0)).round(3).tolist(),
        "expert_median_move_mm": float(1000 * np.median(np.linalg.norm(expert[:, :3], axis=1))),
        "predicted_median_move_mm": float(1000 * np.median(np.linalg.norm(predicted[:, :3], axis=1))),
        "expert_last_z_steps_mm": (1000 * expert_contact).round(3).tolist(),
        "predicted_last_z_steps_mm": (1000 * predicted_contact).round(3).tolist(),
        "gripper_agreement_pct": float(100 * np.mean((expert[:, 6] > .5) == (predicted[:, 6] > .5))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path, help="Original NPZ split directories")
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"),
                        help="Probe one split (default: train and val, or val for first-color comparison)")
    parser.add_argument("--compare-first-color", action="store_true",
                        help="Compare three first-button prompts on identical approach observations")
    args = parser.parse_args()
    checkpoint, data = args.checkpoint.resolve(), args.data.resolve()
    if args.compare_first_color:
        cases = [data / (args.split or "val") / f"order{order}_episode00000.npz" for order in (0, 2, 4)]
    else:
        splits = (args.split,) if args.split else ("train", "val")
        cases = [data / split / "order2_episode00000.npz" for split in splits]
        stats_path = checkpoint.parents[1] / "dataset_statistics.json"
        bounds = json.loads(stats_path.read_text(encoding="utf-8"))["custom_finetuning"]["action"]
        low, high = np.asarray(bounds["q01"][:6]), np.asarray(bounds["q99"][:6])
    for path in [checkpoint, *cases]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("eval_buttons.py").resolve()),
        "--policy-worker", str(checkpoint), "1", "0",
    ]
    results = {}
    color_results = []
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", bufsize=1) as policy:
        try:
            info = read_policy(policy)
            if info.get("ready") is not True:
                raise RuntimeError("Unexpected policy startup response")
            print(f"Memory capacity: {info['mem_length']}", flush=True)
            for path in cases:
                if args.compare_first_color:
                    comparison = compare_first_colors(policy, path, seed=47)
                    color_results.append(comparison)
                    print(f"first_color {json.dumps(comparison)}", flush=True)
                    continue
                expert, predicted = probe(policy, path, seed=47)
                split = path.parent.name
                np.savez_compressed(args.output / f"{split}_actions.npz",
                                    expert=expert, predicted=predicted)
                results[split] = summarize_actions(expert, predicted)
                clipped = (expert[:, :6] < low) | (expert[:, :6] > high)
                results[split]["expert_outside_q01_q99_pct_by_axis"] = (
                    100 * clipped.mean(axis=0)
                ).round(2).tolist()
                expert_xyz = expert[:, :3]
                prediction_error = predicted[:, :3] - expert_xyz
                clipping_error = np.clip(expert_xyz, low[:3], high[:3]) - expert_xyz
                results[split]["approach_prefixes"] = {
                    count: {
                        "pred_minus_expert_xyz_mm": (1000 * prediction_error[:count].sum(axis=0)).round(2).tolist(),
                        "clipped_minus_expert_xyz_mm": (1000 * clipping_error[:count].sum(axis=0)).round(2).tolist(),
                    }
                    for count in (10, 20, 40) if count <= len(expert)
                }
                print(f"{split} {json.dumps(results[split])}", flush=True)
        finally:
            policy.stdin.close()
            if policy.poll() is None:
                policy.terminate()
    summary = {
        "checkpoint": str(checkpoint), "mem_length": info["mem_length"],
        "axis_order": ["x", "y", "z", "roll", "pitch", "yaw"], "policy_seed": 47,
    }
    if args.compare_first_color:
        summary["first_color_comparison"] = color_results
    else:
        summary["splits"] = results
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
