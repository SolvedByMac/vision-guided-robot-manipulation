import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.envs.noisy_reach_env import make_noisy_reach_env


seeds = [0, 1, 2, 3, 4]
episodes = 100

seed_results = []

for seed in seeds:
    model = PPO.load(f"models/reach_ppo_noisy_seed{seed}")
    env = make_noisy_reach_env(seed=seed)

    successes = 0
    total_steps = 0
    final_distances = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        true_goal = env.env.unwrapped.task.get_goal().copy()

        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        successes += int(info.get("is_success", False))
        total_steps += steps

        final_distance = np.linalg.norm(
            true_goal - obs["achieved_goal"]
        )

        final_distances.append(float(final_distance))

    env.close()

    result = {
        "seed": seed,
        "success_rate": successes / episodes,
        "mean_steps": total_steps / episodes,
        "mean_final_distance_m": float(
            np.mean(final_distances)
        ),
    }

    seed_results.append(result)

    print(
        f"seed {seed}: "
        f"success={result['success_rate']:.3f} "
        f"steps={result['mean_steps']:.2f} "
        f"distance={result['mean_final_distance_m'] * 100:.2f} cm"
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

summary = {
    "episodes_per_seed": episodes,
    "seeds": seed_results,
    "success_rate_mean": float(np.mean(success_rates)),
    "success_rate_std": float(np.std(success_rates)),
    "mean_steps_mean": float(np.mean(mean_steps)),
    "mean_steps_std": float(np.std(mean_steps)),
    "final_distance_mean": float(np.mean(final_distances)),
    "final_distance_std": float(np.std(final_distances)),
}

Path("results").mkdir(exist_ok=True)

with open("results/reach_ppo_noisy_5seed.json", "w") as file:
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
print("saved: results/reach_ppo_noisy_5seed.json")