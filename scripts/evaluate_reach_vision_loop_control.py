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
        "models/reach_pose_resnet18_control_10k.pt",
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

successes = 0
total_steps = 0

final_distances = []
pose_errors = []
episode_pose_errors = []

env = make_reach_env()

for episode in range(episodes):
    obs, info = env.reset(seed=episode)

    true_goal = obs["desired_goal"].copy()

    terminated = False
    truncated = False
    steps = 0

    current_episode_errors = []

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

        pose_errors.append(pose_error)
        current_episode_errors.append(pose_error)

        policy_obs = {
            "observation": obs["observation"],
            "achieved_goal": obs["achieved_goal"],
            "desired_goal": predicted_goal.astype(np.float32),
        }

        action, _ = policy.predict(
            policy_obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1

    successes += int(info.get("is_success", False))
    total_steps += steps

    final_distance = float(
        np.linalg.norm(
            true_goal - obs["achieved_goal"]
        )
    )

    final_distances.append(final_distance)

    episode_pose_errors.append(
        float(np.mean(current_episode_errors))
    )

env.close()

results = {
    "episodes": episodes,
    "success_rate": successes / episodes,
    "mean_steps": total_steps / episodes,
    "mean_final_distance_m": float(np.mean(final_distances)),
    "mean_pose_error_m": float(np.mean(pose_errors)),
    "median_pose_error_m": float(np.median(pose_errors)),
    "p90_pose_error_m": float(np.percentile(pose_errors, 90)),
    "p95_pose_error_m": float(np.percentile(pose_errors, 95)),
    "mean_episode_pose_error_m": float(
        np.mean(episode_pose_errors)
    ),
}

Path("results").mkdir(exist_ok=True)

with open(
    "results/reach_vision_loop_control_seed0.json",
    "w",
) as file:
    json.dump(results, file, indent=2)

print("model: control-state 10k")
print("episodes:", results["episodes"])
print("success rate:", round(results["success_rate"], 3))
print("mean steps:", round(results["mean_steps"], 2))
print(
    "mean final distance:",
    round(results["mean_final_distance_m"] * 100, 2),
    "cm",
)
print(
    "mean pose error:",
    round(results["mean_pose_error_m"] * 100, 2),
    "cm",
)
print(
    "median pose error:",
    round(results["median_pose_error_m"] * 100, 2),
    "cm",
)
print(
    "p90 pose error:",
    round(results["p90_pose_error_m"] * 100, 2),
    "cm",
)
print(
    "p95 pose error:",
    round(results["p95_pose_error_m"] * 100, 2),
    "cm",
)
print(
    "mean episode pose error:",
    round(results["mean_episode_pose_error_m"] * 100, 2),
    "cm",
)
print(
    "saved: results/reach_vision_loop_control_seed0.json"
)