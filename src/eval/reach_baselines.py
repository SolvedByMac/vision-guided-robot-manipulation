import numpy as np


def random_policy(obs):
    return np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)


def oracle_policy(obs, gain=10.0):
    achieved = obs["achieved_goal"]
    desired = obs["desired_goal"]

    action = gain * (desired - achieved)
    return np.clip(action, -1.0, 1.0).astype(np.float32)