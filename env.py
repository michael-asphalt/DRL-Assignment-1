import gym
import numpy as np
import random
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")
obs, info = env.reset()

print(dir(env))
help(env)
# print("Action Space:", env.action_space)
# print("Observation Space:", env.observation_space)
# print("Reward Range:", env.reward_range)
# print("Metadata:", env.metadata)