import numpy as np
import random
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

from simple_custom_taxi_env import SimpleTaxiEnv
# 0 taxi_row
# 1 taxi_col
# 2 station_0_0
# 3 station_0_1
# 4 station_1_0
# 5 station_1_1
# 6 station_2_0
# 7 station_2_1
# 8 station_3_0
# 9 station_3_1
# 10 obstacle_north
# 11 obstacle_south
# 12 obstacle_east
# 13 obstacle_west
# 14 passenger_look
# 15 destination_look

# state
# 0 pickup
# 1 obstacle_north
# 2 obstacle_south
# 3 obstacle_east
# 4 obstacle_west
# 5 passenger_look
# 6 destination_look
# 7 sign(taxi_row - next_station_row)
# 8 sign(taxi_col - next_station_col)

class ActionSpace:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return random.randint(0, self.n - 1)

def get_state(obs, pickup, pickup_pos_idx, drop_pos_idx):
    taxi_pos = obs[0], obs[1]
    station_1 = obs[2], obs[3]
    station_2 = obs[4], obs[5]
    station_3 = obs[6], obs[7]
    station_4 = obs[8], obs[9]
    stations_list = [station_1, station_2, station_3, station_4]
    if not pickup:
        next_station = stations_list[pickup_pos_idx]
    else:
        next_station = stations_list[drop_pos_idx]
    
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

def tabular_q_learning(episodes=5000, alpha=0.1, gamma=0.99,
                         epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.9997,
                         fuel_limit=5000, grid_size=5):
    env = SimpleTaxiEnv(grid_size=grid_size, fuel_limit=fuel_limit)
    env.action_space = ActionSpace(6)
    
    obs, _ = env.reset()
    q_table = defaultdict(lambda: np.zeros(6))  
    rewards_per_episode = []
    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _ = env.reset()
        pickup_pos_idx = 0 
        drop_pos_idx = 0
        state = get_state(obs, False, pickup_pos_idx, drop_pos_idx)
        done = False
        total_reward = 0
        has_pickuped = 0
        has_dropped = 0
        while not done:
            # intitialize Q-table for unseen states
            pickup = state[0]
            if state not in q_table:
                q_table[state] = np.zeros(6)
        
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
            
            obs, reward, done, info = env.step(action)
            shaped_reward = 0
            # whether pickup or not
            # decide pickup_pos_idx, drop_pos_idx (0, 1, 2, 3)
            
            if not pickup:
                flag = (obs[0] == obs[2 + 2 * pickup_pos_idx] and obs[1] == obs[3 + 2 * pickup_pos_idx])
                if flag and obs[14] != 1:
                    if action == 4 or action == 5:
                        shaped_reward -= 20

                    if pickup_pos_idx < 3:
                        pickup_pos_idx += 1
                elif flag and obs[14] == 1:
                    if action == 4:
                        pickup = True
                        if has_pickuped == 0:
                            shaped_reward += 50
                            has_pickuped = 1
                    elif action == 5:
                        shaped_reward -= 20
            else:
                flag = (obs[0] == obs[2 + 2 * drop_pos_idx] and obs[1] == obs[3 + 2 * drop_pos_idx])
                if flag and obs[15] != 1:
                    if action == 4:
                        shaped_reward -= 40
                    elif action == 5:
                        pickup = False
                        pickup_pos_idx = 0
                        drop_pos_idx = 0
                        shaped_reward -= 40
                    
                    elif  drop_pos_idx < 3:
                        drop_pos_idx += 1
                elif flag and obs[15] == 1:
                    if action == 5:
                        pickup = False  
                        pickup_pos_idx = 0
                        drop_pos_idx = 0
                        if has_dropped == 0:
                            shaped_reward += 100
                            has_dropped = 1
                    elif action == 4:
                        shaped_reward -= 40


            next_state = get_state(obs, pickup, pickup_pos_idx, drop_pos_idx)
            
            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)
            

            if state[1] == 1 and action == 1:
                shaped_reward -= 20
            if state[2] == 1 and action == 0:
                shaped_reward -= 20
            if state[3] == 1 and action == 2:
                shaped_reward -= 20
            if state[4] == 1 and action == 3:
                shaped_reward -= 20
            
            reward += shaped_reward
            total_reward += reward
            best_next_action = int(np.argmax(q_table[next_state]))
            q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])
            
            state = next_state
        
        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")
    
    return q_table, rewards_per_episode

if __name__ == "__main__":
    episodes = 10000 
    all_q_tables = {}
    all_rewards = {}

    for grid_size in range(5, 11):
        print(f"start training grid_size = {grid_size}")
        q_table, rewards = tabular_q_learning(episodes=episodes, grid_size=grid_size)
        all_q_tables[grid_size] = dict(q_table)
        all_rewards[grid_size] = rewards

        file_name = f"q_table_{grid_size}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(dict(q_table), f)
        print(f"Q-table of gridsize = {grid_size} have been saved to {file_name}\n")
    
    # merge q_tables
    merged_q_table = {}
    for grid_size in sorted(all_q_tables.keys()):
        q_table = all_q_tables[grid_size]
        for state, q_values in q_table.items():
            if state not in merged_q_table:
                merged_q_table[state] = (grid_size, q_values)
            else:
                stored_grid_size, _ = merged_q_table[state]
                if grid_size > stored_grid_size:
                    merged_q_table[state] = (grid_size, q_values)


    final_q_table = {state: q_values for state, (grid_size, q_values) in merged_q_table.items()}

    # 合併所有資料，並將合併後的 final_q_table 加入其中
    merged_file_name = "q_table.pkl"
    with open(merged_file_name, "wb") as f:
        pickle.dump(final_q_table, f)
    print(f"所有 grid_size 的 Q-table 已合併，並儲存至 {merged_file_name}")

    # plt.plot(rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title(f"Rewards per Episode (grid_size=10)")
    # plt.show()
