import json
from pathlib import Path

import cv2
from stable_baselines3 import PPO

from src.envs.camera import render_camera
from src.envs.reach_env import make_reach_env


output_dir = Path("data/reach_pose_control")
images_dir = output_dir / "images"

output_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)

policy = PPO.load("models/reach_ppo_seed0")
env = make_reach_env()

num_samples = 10000
samples = []

for index in range(num_samples):
    obs, info = env.reset(seed=index)

    rollout_steps = index % 5

    for _ in range(rollout_steps):
        action, _ = policy.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    image, _, _ = render_camera()

    image_path = images_dir / f"{index:06d}.png"

    cv2.imwrite(
        str(image_path),
        cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR),
    )

    samples.append(
        {
            "image": f"images/{index:06d}.png",
            "goal_position": obs["desired_goal"].tolist(),
            "rollout_steps": rollout_steps,
        }
    )

    if (index + 1) % 100 == 0:
        print(f"generated {index + 1}/{num_samples}")

env.close()

with open(output_dir / "metadata.json", "w") as file:
    json.dump(samples, file, indent=2)

print(f"saved {num_samples} samples to {output_dir}")