import numpy as np

from src.envs.reach_env import make_reach_env


env = make_reach_env()

obs, info = env.reset(seed=0)

print("observation keys:", obs.keys())
print("observation:", obs["observation"])
print("achieved goal:", obs["achieved_goal"])
print("desired goal:", obs["desired_goal"])
print("action space:", env.action_space)

for _ in range(100):
    action = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

print("completed 100 simulation steps")

env.close()