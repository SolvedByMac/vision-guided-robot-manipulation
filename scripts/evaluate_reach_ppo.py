import json
from pathlib import Path

from stable_baselines3 import PPO

from src.envs.reach_env import make_reach_env


env = make_reach_env()
model = PPO.load("models/reach_ppo_seed0")

episodes = 100
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

    distance = ((obs["desired_goal"] - obs["achieved_goal"]) ** 2).sum() ** 0.5
    final_distances.append(float(distance))

env.close()

results = {
    "episodes": episodes,
    "success_rate": successes / episodes,
    "mean_steps": total_steps / episodes,
    "mean_final_distance": sum(final_distances) / episodes,
}

Path("results").mkdir(exist_ok=True)

with open("results/reach_ppo_seed0.json", "w") as file:
    json.dump(results, file, indent=2)

print("success rate:", round(results["success_rate"], 3))
print("mean steps:", round(results["mean_steps"], 2))
print("mean final distance:", round(results["mean_final_distance"], 4))
print("saved: results/reach_ppo_seed0.json")