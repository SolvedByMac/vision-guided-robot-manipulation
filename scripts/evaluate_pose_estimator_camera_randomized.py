import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.perception.dataset import ReachPoseDataset
from src.perception.model import PoseRegressor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ReachPoseDataset(
    "data/reach_pose_camera_randomized",
    "test",
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

model = PoseRegressor().to(device)

model.load_state_dict(
    torch.load(
        "models/reach_pose_resnet18_camera_randomized_10k.pt",
        map_location=device,
    )
)

model.eval()

predictions = []
targets = []

with torch.no_grad():
    for images, batch_targets in loader:
        images = images.to(device, non_blocking=True)

        batch_predictions = model(images).cpu().numpy()

        predictions.append(batch_predictions)
        targets.append(batch_targets.numpy())

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

errors = predictions - targets
distances = np.linalg.norm(errors, axis=1)

mae_per_axis = np.mean(np.abs(errors), axis=0)
rmse_per_axis = np.sqrt(np.mean(errors ** 2, axis=0))

results = {
    "samples": len(dataset),
    "mean_error_m": float(np.mean(distances)),
    "median_error_m": float(np.median(distances)),
    "p90_error_m": float(np.percentile(distances, 90)),
    "p95_error_m": float(np.percentile(distances, 95)),
    "max_error_m": float(np.max(distances)),
    "mae_per_axis_m": mae_per_axis.tolist(),
    "rmse_per_axis_m": rmse_per_axis.tolist(),
}

Path("results").mkdir(exist_ok=True)

with open(
    "results/reach_pose_test_camera_randomized_10k.json",
    "w",
) as file:
    json.dump(results, file, indent=2)

print("model: camera-randomized 10k")
print("test samples:", results["samples"])
print("mean error:", round(results["mean_error_m"] * 100, 2), "cm")
print("median error:", round(results["median_error_m"] * 100, 2), "cm")
print("p90 error:", round(results["p90_error_m"] * 100, 2), "cm")
print("p95 error:", round(results["p95_error_m"] * 100, 2), "cm")
print("max error:", round(results["max_error_m"] * 100, 2), "cm")
print("axis MAE:", np.round(mae_per_axis * 100, 2), "cm")
print("axis RMSE:", np.round(rmse_per_axis * 100, 2), "cm")
print(
    "saved: results/reach_pose_test_camera_randomized_10k.json"
)