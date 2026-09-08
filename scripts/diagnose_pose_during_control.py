import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from src.envs.camera import render_camera
from src.envs.reach_env import make_reach_env
from src.perception.model import PoseRegressor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pose_model = PoseRegressor().to(device)

pose_model.load_state_dict(
    torch.load(
        "models/reach_pose_resnet18_tightcam_10k.pt",
        map_location=device,
    )
)

pose_model.eval()

policy = PPO.load("models/reach_ppo_seed0")

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)

episodes = 100
records = []

env = make_reach_env()

for episode in range(episodes):
    obs, info = env.reset(seed=episode)

    true_goal = obs["desired_goal"].copy()

    terminated = False
    truncated = False
    step = 0

    while not terminated and not truncated:
        image, _, _ = render_camera()
        rgb = image[:, :, :3]

        image_tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_goal = (
                pose_model(image_tensor)
                .cpu()
                .numpy()[0]
            )

        pose_error = float(
            np.linalg.norm(predicted_goal - true_goal)
        )

        records.append(
            {
                "episode": episode,
                "step": step,
                "pose_error_m": pose_error,
            }
        )

        action, _ = policy.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        step += 1

env.close()

step_groups = {}

for record in records:
    step = record["step"]
    step_groups.setdefault(step, [])
    step_groups[step].append(record["pose_error_m"])

summary = {}

for step, errors in sorted(step_groups.items()):
    errors = np.array(errors)

    summary[str(step)] = {
        "samples": len(errors),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "p90_error_m": float(np.percentile(errors, 90)),
    }

Path("results").mkdir(exist_ok=True)

with open("results/pose_error_during_privileged_control.json", "w") as file:
    json.dump(summary, file, indent=2)

print("pose error during privileged control")

for step, values in summary.items():
    print(
        f"step {step}: "
        f"n={values['samples']} "
        f"mean={values['mean_error_m'] * 100:.2f} cm "
        f"median={values['median_error_m'] * 100:.2f} cm "
        f"p90={values['p90_error_m'] * 100:.2f} cm"
    )

print("saved: results/pose_error_during_privileged_control.json")