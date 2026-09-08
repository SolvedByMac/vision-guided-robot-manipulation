from pathlib import Path

from src.envs.reach_env import make_reach_env
from src.policy.train_ppo import train_ppo


Path("models").mkdir(exist_ok=True)

env = make_reach_env()

model = train_ppo(
    env,
    total_timesteps=100_000,
    seed=0,
)

model.save("models/reach_ppo_seed0")

env.close()

print("saved: models/reach_ppo_seed0.zip")