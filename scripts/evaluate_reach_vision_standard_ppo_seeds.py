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

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)

seeds = [0, 1, 2, 3, 4]
episodes = 100

seed_results = []

for seed in seeds:
    policy = PPO.load(f"models/reach_ppo_seed{seed}")
    env = make_reach_env()

    successes = 0
    total_steps = 0
    final_distances = []
    pose_errors = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        true_goal = obs["desired_goal"].copy()

        terminated = False
        truncated = False
        steps = 0

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

    env.close()

    result = {
        "seed": seed,
        "success_rate": successes / episodes,
        "mean_steps": total_steps / episodes,
        "mean_final_distance_m": float(
            np.mean(final_distances)
        ),
        "mean_pose_error_m": float(
            np.mean(pose_errors)
        ),
    }

    seed_results.append(result)

    print(
        f"seed {seed}: "
        f"success={result['success_rate']:.3f} "
        f"steps={result['mean_steps']:.2f} "
        f"distance={result['mean_final_distance_m'] * 100:.2f} cm "
        f"pose_error={result['mean_pose_error_m'] * 100:.2f} cm"
    )

success_rates = np.array(
    [result["success_rate"] for result in seed_results]
)

mean_steps = np.array(
    [result["mean_steps"] for result in seed_results]
)

final_distances = np.array(
    [result["mean_final_distance_m"] for result in seed_results]
)

pose_errors = np.array(
    [result["mean_pose_error_m"] for result in seed_results]
)

summary = {
    "episodes_per_seed": episodes,
    "seeds": seed_results,
    "success_rate_mean": float(np.mean(success_rates)),
    "success_rate_std": float(np.std(success_rates)),
    "mean_steps_mean": float(np.mean(mean_steps)),
    "mean_steps_std": float(np.std(mean_steps)),
    "final_distance_mean": float(np.mean(final_distances)),
    "final_distance_std": float(np.std(final_distances)),
    "pose_error_mean": float(np.mean(pose_errors)),
    "pose_error_std": float(np.std(pose_errors)),
}

Path("results").mkdir(exist_ok=True)

with open(
    "results/reach_vision_standard_ppo_5seed.json",
    "w",
) as file:
    json.dump(summary, file, indent=2)

print()
print(
    "success:",
    f"{summary['success_rate_mean']:.3f}",
    "±",
    f"{summary['success_rate_std']:.3f}",
)
print(
    "mean steps:",
    f"{summary['mean_steps_mean']:.3f}",
    "±",
    f"{summary['mean_steps_std']:.3f}",
)
print(
    "final distance:",
    f"{summary['final_distance_mean'] * 100:.2f}",
    "±",
    f"{summary['final_distance_std'] * 100:.2f}",
    "cm",
)
print(
    "pose error:",
    f"{summary['pose_error_mean'] * 100:.2f}",
    "±",
    f"{summary['pose_error_std'] * 100:.2f}",
    "cm",
)
print("saved: results/reach_vision_standard_ppo_5seed.json")