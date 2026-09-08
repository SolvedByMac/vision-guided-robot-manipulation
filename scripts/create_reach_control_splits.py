import json
from pathlib import Path

import numpy as np


root = Path("data/reach_pose_control")

with open(root / "metadata.json") as file:
    samples = json.load(file)

rng = np.random.default_rng(0)

indices = np.arange(len(samples))
rng.shuffle(indices)

train_end = int(0.8 * len(indices))
val_end = int(0.9 * len(indices))

splits = {
    "train": indices[:train_end],
    "val": indices[train_end:val_end],
    "test": indices[val_end:],
}

for split_name, split_indices in splits.items():
    split_samples = [
        samples[int(index)]
        for index in split_indices
    ]

    with open(root / f"{split_name}.json", "w") as file:
        json.dump(split_samples, file, indent=2)

    print(f"{split_name}: {len(split_samples)}")