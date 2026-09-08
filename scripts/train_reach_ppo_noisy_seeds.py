from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.envs.noisy_reach_env import make_noisy_reach_env


Path("models").mkdir(exist_ok=True)

seeds = [0, 1, 2, 3, 4]
total_timesteps = 100_000

for seed in seeds:
    print(f"training noisy PPO seed {seed}")

    env = make_noisy_reach_env(seed=seed)
    env.reset(seed=seed)

    env = Monitor(env)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=seed,
        tensorboard_log="runs/",
    )

    model.learn(total_timesteps=total_timesteps)

    model_path = f"models/reach_ppo_noisy_seed{seed}"

    model.save(model_path)
    env.close()

    print(f"saved: {model_path}.zip")