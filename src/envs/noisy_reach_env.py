import gymnasium as gym
import numpy as np

from src.envs.reach_env import make_reach_env


class NoisyGoalObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        noise_std=(0.0185, 0.0178, 0.0127),
        seed=0,
    ):
        super().__init__(env)

        self.noise_std = np.array(
            noise_std,
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)

    def observation(self, observation):
        noisy_observation = {
            key: value.copy()
            for key, value in observation.items()
        }

        noise = self.rng.normal(
            loc=0.0,
            scale=self.noise_std,
        ).astype(np.float32)

        noisy_observation["desired_goal"] = (
            noisy_observation["desired_goal"] + noise
        ).astype(np.float32)

        return noisy_observation


def make_noisy_reach_env(seed=0):
    env = make_reach_env()

    return NoisyGoalObservationWrapper(
        env,
        seed=seed,
    )