"""Compare first-button actions on recorded frames, without running the simulator."""

import argparse
import base64
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from eval_buttons import read_policy


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
    predictions = []
    for i in range(stop):
        with io.BytesIO() as image:
            Image.fromarray(frames[i]).save(image, format="PNG", compress_level=1)
            request = {
                "image": base64.b64encode(image.getvalue()).decode("ascii"),
                "instruction": instruction, "first_frame": i == 0, "policy_seed": seed,
            }
        policy.stdin.write(json.dumps(request) + "\n")
        predictions.append(read_policy(policy)[0])
    return expert[:stop], np.asarray(predictions)


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
                        help="Probe one split (default: train and val)")
    args = parser.parse_args()
    checkpoint, data = args.checkpoint.resolve(), args.data.resolve()
    splits = (args.split,) if args.split else ("train", "val")
    cases = [data / split / "order2_episode00000.npz" for split in splits]
    for path in [checkpoint, *cases]:
        if not path.is_file():
            raise FileNotFoundError(path)
    stats_path = checkpoint.parents[1] / "dataset_statistics.json"
    bounds = json.loads(stats_path.read_text(encoding="utf-8"))["custom_finetuning"]["action"]
    low, high = np.asarray(bounds["q01"][:6]), np.asarray(bounds["q99"][:6])
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("eval_buttons.py").resolve()),
        "--policy-worker", str(checkpoint), "1", "0",
    ]
    results = {}
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", bufsize=1) as policy:
        try:
            info = read_policy(policy)
            if info.get("ready") is not True:
                raise RuntimeError("Unexpected policy startup response")
            print(f"Memory capacity: {info['mem_length']}", flush=True)
            for path in cases:
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
    (args.output / "summary.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "mem_length": info["mem_length"],
                    "axis_order": ["x", "y", "z", "roll", "pitch", "yaw"],
                    "policy_seed": 47, "splits": results}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
