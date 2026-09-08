import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.envs.camera import get_camera_matrices
from src.perception.dataset import ReachPoseDataset
from src.perception.model import PoseRegressor


def project_point(point, view_matrix, projection_matrix, width, height):
    point = np.array([point[0], point[1], point[2], 1.0])

    view = np.array(view_matrix).reshape(4, 4, order="F")
    projection = np.array(projection_matrix).reshape(4, 4, order="F")

    clip = projection @ view @ point
    ndc = clip[:3] / clip[3]

    x = int((ndc[0] + 1.0) * 0.5 * width)
    y = int((1.0 - ndc[1]) * 0.5 * height)

    return x, y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ReachPoseDataset("data/reach_pose", "test")

model = PoseRegressor().to(device)

model.load_state_dict(
    torch.load(
        "models/reach_pose_resnet18_tightcam_10k.pt",
        map_location=device,
    )
)

model.eval()

with open("data/reach_pose/test.json") as file:
    samples = json.load(file)

width = 256
height = 256

view_matrix, projection_matrix = get_camera_matrices(width, height)

records = []

with torch.no_grad():
    for index in range(len(dataset)):
        image, target = dataset[index]

        prediction = (
            model(image.unsqueeze(0).to(device))
            .cpu()
            .numpy()[0]
        )

        target = target.numpy()
        error = float(np.linalg.norm(prediction - target))

        records.append(
            {
                "index": index,
                "prediction": prediction,
                "target": target,
                "error": error,
            }
        )

records.sort(key=lambda item: item["error"])

selected = {
    "best": records[0],
    "median": records[len(records) // 2],
    "worst": records[-1],
}

output_dir = Path("results/pose_overlays_tightcam")
output_dir.mkdir(parents=True, exist_ok=True)

for name, record in selected.items():
    sample = samples[record["index"]]

    image = cv2.imread(
        str(Path("data/reach_pose") / sample["image"])
    )

    true_pixel = project_point(
        record["target"],
        view_matrix,
        projection_matrix,
        width,
        height,
    )

    predicted_pixel = project_point(
        record["prediction"],
        view_matrix,
        projection_matrix,
        width,
        height,
    )

    cv2.circle(image, true_pixel, 5, (0, 255, 0), -1)
    cv2.circle(image, predicted_pixel, 5, (0, 0, 255), -1)

    cv2.line(
        image,
        true_pixel,
        predicted_pixel,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        image,
        f"error: {record['error'] * 100:.2f} cm",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_path = output_dir / f"{name}.png"

    cv2.imwrite(str(output_path), image)

    print(name)
    print("true:", np.round(record["target"], 4))
    print("pred:", np.round(record["prediction"], 4))
    print("error:", round(record["error"] * 100, 2), "cm")
    print("saved:", output_path)
    print()