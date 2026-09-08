import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize


class ReachPoseDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)

        with open(self.root / f"{split}.json") as file:
            self.samples = json.load(file)

        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = cv2.imread(str(self.root / sample["image"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)

        target = torch.tensor(
            sample["goal_position"],
            dtype=torch.float32,
        )

        return image, target