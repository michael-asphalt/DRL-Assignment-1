# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

pickup = False
pickup_pos_idx = 0
drop_pos_idx = 0
xxx = 0

file_name = f"q_table.pkl"
try:
    with open(file_name, "rb") as f:
        Q_table = pickle.load(f)
except FileNotFoundError:
    Q_table = {}

def get_state(obs, pickup, pickup_pos_idx, drop_pos_idx):
    taxi_pos = obs[0], obs[1]
    station_1 = obs[2], obs[3]
    station_2 = obs[4], obs[5]
    station_3 = obs[6], obs[7]
    station_4 = obs[8], obs[9]
    stations_list = [station_1, station_2, station_3, station_4]
    if not pickup:
        next_station = stations_list[pickup_pos_idx][0], stations_list[pickup_pos_idx][1]
    else:
        next_station = stations_list[drop_pos_idx][0], stations_list[drop_pos_idx][1]
    ret = []
    ret.append(pickup)
    ret.append(obs[10])
    ret.append(obs[11])
    ret.append(obs[12])
    ret.append(obs[13])
    ret.append(obs[14])
    ret.append(obs[15])
    ret.append(np.sign(taxi_pos[0] - next_station[0]))
    ret.append(np.sign(taxi_pos[1] - next_station[1]))

    return tuple(ret)


def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    global pickup
    global pickup_pos_idx
    global drop_pos_idx
    global action
    state = get_state(obs, pickup, pickup_pos_idx, drop_pos_idx)
    if not pickup:
        # flag = at one of the station
        flag = ((obs[0] == obs[2] and obs[1] == obs[3]) or \
            (obs[0] == obs[4] and obs[1] == obs[5]) or \
            (obs[0] == obs[6] and obs[1] == obs[7]) or \
            (obs[0] == obs[8] and obs[1] == obs[9]))
        if (flag and state[5] != 1) and pickup_pos_idx < 3:
            pickup_pos_idx += 1
        elif flag and state[5] == 1 and action == 4:
            pickup = True
    else:
        flag = ((obs[0] == obs[2] and obs[1] == obs[3]) or \
            (obs[0] == obs[4] and obs[1] == obs[5]) or \
            (obs[0] == obs[6] and obs[1] == obs[7]) or \
            (obs[0] == obs[8] and obs[1] == obs[9]))
        if flag and state[6] != 1:
            drop_pos_idx += 1
            drop_pos_idx %= 4
        elif flag and state[6] == 1 and action == 5:
            pickup = False

    if state in Q_table:
        action = np.argmax(Q_table[state])
        return action
    else:
        # Fallback to a random action if state is unseen
        if obs[10] == 0:
            return 1
        elif obs[11] == 0:
            return 0
        elif obs[12] == 0:
            return 2
        elif obs[13] == 0:
            return 3
        return action
   
    # You can submit this random agent to evaluate the performance of a purely random strategy.

