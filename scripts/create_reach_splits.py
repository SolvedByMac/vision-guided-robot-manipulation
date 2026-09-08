import json
from pathlib import Path

import numpy as np


metadata_path = Path("data/reach_pose/metadata.json")
output_dir = Path("data/reach_pose")

with open(metadata_path) as file:
    samples = json.load(file)

indices = np.arange(len(samples))
rng = np.random.default_rng(0)
rng.shuffle(indices)

train_end = int(0.8 * len(indices))
val_end = int(0.9 * len(indices))

splits = {
    "train": indices[:train_end].tolist(),
    "val": indices[train_end:val_end].tolist(),
    "test": indices[val_end:].tolist(),
}

for name, split_indices in splits.items():
    split_samples = [samples[index] for index in split_indices]

    with open(output_dir / f"{name}.json", "w") as file:
        json.dump(split_samples, file, indent=2)

    print(f"{name}: {len(split_samples)}")