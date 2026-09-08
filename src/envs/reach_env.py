import gymnasium as gym
import panda_gym


def make_reach_env():
    return gym.make("PandaReach-v3")