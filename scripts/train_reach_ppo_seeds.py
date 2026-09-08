from pathlib import Path

from src.envs.reach_env import make_reach_env
from src.policy.train_ppo import train_ppo


seeds = [1, 2, 3, 4]

Path("models").mkdir(exist_ok=True)

for seed in seeds:
    print()
    print(f"training seed {seed}")

    env = make_reach_env()

    model = train_ppo(
        env,
        total_timesteps=100_000,
        seed=seed,
    )

    path = f"models/reach_ppo_seed{seed}"
    model.save(path)

    env.close()

    print(f"saved: {path}.zip")