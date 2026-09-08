import json
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from src.envs.reach_env import make_reach_env


def render_randomized_camera(rng, width=256, height=256):
    eye = [
        rng.uniform(0.76, 0.84),
        rng.uniform(0.76, 0.84),
        rng.uniform(0.62, 0.68),
    ]

    fov = rng.uniform(43.0, 47.0)

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=[0.0, 0.0, 0.15],
        cameraUpVector=[0.0, 0.0, 1.0],
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=0.1,
        farVal=2.0,
    )

    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )

    rgba = np.asarray(
        rgba,
        dtype=np.uint8,
    ).reshape(height, width, 4)

    return rgba, eye, fov


output_dir = Path("data/reach_pose_camera_randomized")
images_dir = output_dir / "images"

output_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)

policy = PPO.load("models/reach_ppo_seed0")
env = make_reach_env()

num_samples = 10000
rng = np.random.default_rng(0)

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

    image, eye, fov = render_randomized_camera(rng)

    image_path = images_dir / f"{index:06d}.png"

    cv2.imwrite(
        str(image_path),
        cv2.cvtColor(
            image[:, :, :3],
            cv2.COLOR_RGB2BGR,
        ),
    )

    samples.append(
        {
            "image": f"images/{index:06d}.png",
            "goal_position": obs["desired_goal"].tolist(),
            "rollout_steps": rollout_steps,
            "camera_eye": eye,
            "camera_fov": float(fov),
        }
    )

    if (index + 1) % 100 == 0:
        print(f"generated {index + 1}/{num_samples}")

env.close()

with open(output_dir / "metadata.json", "w") as file:
    json.dump(samples, file, indent=2)

print(f"saved {num_samples} samples to {output_dir}")