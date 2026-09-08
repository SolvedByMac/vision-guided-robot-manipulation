import json
from pathlib import Path

import cv2

from src.envs.camera import render_camera
from src.envs.reach_env import make_reach_env


output_dir = Path("data/reach_pose")
images_dir = output_dir / "images"

images_dir.mkdir(parents=True, exist_ok=True)

env = make_reach_env()

samples = []
num_samples = 10000

for index in range(num_samples):
    obs, info = env.reset(seed=index)

    image, _, _ = render_camera()

    rgb = image[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    image_name = f"{index:06d}.png"
    image_path = images_dir / image_name

    cv2.imwrite(str(image_path), bgr)

    samples.append(
        {
            "image": f"images/{image_name}",
            "goal_position": obs["desired_goal"].tolist(),
        }
    )

    if (index + 1) % 100 == 0:
        print(f"generated {index + 1}/{num_samples}")

env.close()

with open(output_dir / "metadata.json", "w") as file:
    json.dump(samples, file, indent=2)

print(f"saved {num_samples} samples to {output_dir}")