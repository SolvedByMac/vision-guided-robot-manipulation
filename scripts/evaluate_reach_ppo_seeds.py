import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.envs.reach_env import make_reach_env


def evaluate_model(model_path, episodes=100):
    env = make_reach_env()
    model = PPO.load(model_path)

    successes = 0
    total_steps = 0
    final_distances = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        successes += int(info.get("is_success", False))
        total_steps += steps

        distance = np.linalg.norm(
            obs["desired_goal"] - obs["achieved_goal"]
        )
        final_distances.append(float(distance))

    env.close()

    return {
        "success_rate": successes / episodes,
        "mean_steps": total_steps / episodes,
        "mean_final_distance": float(np.mean(final_distances)),
    }


seeds = [0, 1, 2, 3, 4]
results = {}

for seed in seeds:
    path = f"models/reach_ppo_seed{seed}"
    metrics = evaluate_model(path)

    results[str(seed)] = metrics

    print(f"seed {seed}")
    print("success rate:", round(metrics["success_rate"], 3))
    print("mean steps:", round(metrics["mean_steps"], 2))
    print("mean final distance:", round(metrics["mean_final_distance"], 4))
    print()

success_rates = [results[str(seed)]["success_rate"] for seed in seeds]
mean_steps = [results[str(seed)]["mean_steps"] for seed in seeds]
final_distances = [
    results[str(seed)]["mean_final_distance"] for seed in seeds
]

summary = {
    "success_rate_mean": float(np.mean(success_rates)),
    "success_rate_std": float(np.std(success_rates)),
    "mean_steps_mean": float(np.mean(mean_steps)),
    "mean_steps_std": float(np.std(mean_steps)),
    "final_distance_mean": float(np.mean(final_distances)),
    "final_distance_std": float(np.std(final_distances)),
}

output = {
    "episodes_per_seed": 100,
    "seeds": results,
    "summary": summary,
}

Path("results").mkdir(exist_ok=True)

with open("results/reach_ppo_5seed.json", "w") as file:
    json.dump(output, file, indent=2)

print("summary")
print(
    "success rate:",
    round(summary["success_rate_mean"], 3),
    "+/-",
    round(summary["success_rate_std"], 3),
)
print(
    "mean steps:",
    round(summary["mean_steps_mean"], 2),
    "+/-",
    round(summary["mean_steps_std"], 2),
)
print(
    "final distance:",
    round(summary["final_distance_mean"], 4),
    "+/-",
    round(summary["final_distance_std"], 4),
)

print("saved: results/reach_ppo_5seed.json")