# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

grid_size = 10
file_name = f"q_table-{grid_size}.pkl"
try:
    with open(file_name, "rb") as f:
        Q_table = pickle.load(f)
except FileNotFoundError:
    Q_table = {}

def get_state(obs):
    return obs

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    state = get_state(obs)
    if state in Q_table:
        return int(np.argmax(Q_table[state]))
    else:
        # Fallback to a random action if state is unseen
        return random.choice([0, 1, 2, 3])
    # You can submit this random agent to evaluate the performance of a purely random strategy.

