"""Export replay-validated button episodes to MemoryVLA's custom_finetuning dataset."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


class CustomFinetuning(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def __init__(self, *, source, image_size, **kwargs):
        self.source = source
        self.image_size = image_size
        super().__init__(**kwargs)

    def _info(self):
        tensor = tfds.features.Tensor
        return tfds.core.DatasetInfo(builder=self, features=tfds.features.FeaturesDict({
            "steps": tfds.features.Dataset({
                "observation": {
                    "image": tfds.features.Image(
                        shape=(self.image_size, self.image_size, 3), encoding_format="png"
                    ),
                    "base_pose_tool_reached": tensor(shape=(7,), dtype=np.float32),
                    "gripper_closed": tensor(shape=(1,), dtype=np.float32),
                },
                "action": tensor(shape=(7,), dtype=np.float32),
                "language_instruction": tfds.features.Text(),
                "is_first": tfds.features.Scalar(dtype=np.bool_),
                "is_last": tfds.features.Scalar(dtype=np.bool_),
                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
            }),
            "episode_metadata": {"episode_id": tfds.features.Text()},
        }))

    def _split_generators(self, dl_manager):
        # MemoryVLA computes normalization over "all"; export only train here.
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        paths = sorted((self.source / "train").glob("order*_episode*.npz"))
        if not paths:
            raise ValueError(f"No training episodes in {self.source / 'train'}")
        for path in paths:
            with np.load(path, allow_pickle=False) as episode:
                action = episode["action"]
                image = episode["image"]
                pose = episode["pose"]
                closed = episode["gripper_closed"]
                order = episode["order"]
                events = episode["press_events"]
                instruction = str(episode["instruction"].item())
            if len(action) < 16 or action.shape != (len(image) - 1, 7):
                raise ValueError(f"{path}: expected >=16 actions and one more observation than actions")
            if sorted(order.tolist()) != [0, 1, 2] or events.shape != (3, 2):
                raise ValueError(f"{path}: expected three distinct target buttons and three presses")
            if not np.array_equal(events[:, 1], order) or np.any(np.diff(events[:, 0]) <= 0):
                raise ValueError(f"{path}: the recorded press sequence is invalid")
            if pose.shape != (len(image), 7) or closed.shape != (len(image), 1):
                raise ValueError(f"{path}: invalid proprioception shape")
            if not all(np.isfinite(x).all() for x in (pose, action, closed)) or not instruction:
                raise ValueError(f"{path}: missing instruction or non-finite numeric fields")
            if image.shape[1:] != (self.image_size, self.image_size, 3) or image.dtype != np.uint8:
                raise ValueError(f"{path}: unexpected RGB shape or dtype")

            terminal_action = np.zeros(7, dtype=np.float32)
            terminal_action[6] = action[-1, 6]
            all_actions = np.vstack([action, terminal_action])
            steps = []
            for i, control in enumerate(all_actions):
                last = i == len(all_actions) - 1
                steps.append({
                    "observation": {
                        "image": image[i],
                        "base_pose_tool_reached": pose[i],
                        "gripper_closed": closed[i],
                    },
                    "action": control,
                    "language_instruction": instruction,
                    "is_first": i == 0,
                    "is_last": last,
                    "is_terminal": last,
                })
            yield path.stem, {"steps": steps, "episode_metadata": {"episode_id": path.stem}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    builder = CustomFinetuning(
        source=args.source, data_dir=str(args.data_root), image_size=args.image_size
    )
    if Path(builder.data_dir).exists():
        raise FileExistsError(f"{builder.data_dir} exists. Choose a fresh --data-root.")
    builder.download_and_prepare()
    reader = tfds.builder_from_directory(builder.data_dir)
    example = next(iter(reader.as_dataset(split="train").take(1)))
    steps = list(itertools.islice(example["steps"].as_numpy_iterator(), 2))
    if len(steps) != 2 or not steps[0]["is_first"] or steps[0]["action"].shape != (7,):
        raise RuntimeError("TFDS round-trip did not preserve the expected step schema.")
    print(f"{builder.data_dir}: {reader.info.splits['train'].num_examples} training episodes")


if __name__ == "__main__":
    main()
