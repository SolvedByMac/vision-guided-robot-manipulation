import json
from pathlib import Path

from src.envs.reach_env import make_reach_env
from src.eval.reach_baselines import oracle_policy, random_policy


def evaluate(policy, episodes=100):
    env = make_reach_env()

    successes = 0
    total_steps = 0
    final_distances = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        successes += int(info.get("is_success", False))
        total_steps += steps

        distance = ((obs["desired_goal"] - obs["achieved_goal"]) ** 2).sum() ** 0.5
        final_distances.append(float(distance))

    env.close()

    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "mean_steps": total_steps / episodes,
        "mean_final_distance": sum(final_distances) / episodes,
    }


results = {
    "random": evaluate(random_policy),
    "oracle": evaluate(oracle_policy),
}

Path("results").mkdir(exist_ok=True)

with open("results/reach_baselines.json", "w") as file:
    json.dump(results, file, indent=2)

for name, metrics in results.items():
    print(name)
    print("success rate:", round(metrics["success_rate"], 3))
    print("mean steps:", round(metrics["mean_steps"], 2))
    print("mean final distance:", round(metrics["mean_final_distance"], 4))
    print()

print("saved: results/reach_baselines.json")