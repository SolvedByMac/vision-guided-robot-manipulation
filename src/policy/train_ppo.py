from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def train_ppo(env, total_timesteps=100_000, seed=0):
    env = Monitor(env)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=seed,
        tensorboard_log="runs/",
    )

    model.learn(total_timesteps=total_timesteps)

    return model