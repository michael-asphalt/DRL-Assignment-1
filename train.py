import gym
import numpy as np
import random
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

def tabular_q_learning(env_name="Taxi-v3", episodes=5000, alpha=0.1, gamma=0.99,
                       epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.9997):
    """
    Implementing Tabular Q-Learning with Epsilon Decay for Taxi-v3.
    - Uses a Q-table to store action values for each state.
    - Updates Q-values using the Bellman equation.
    - Implements ε-greedy exploration for action selection.
    """
    env = gym.make(env_name)
    obs, _ = env.reset()
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards_per_episode = []
    epsilon = epsilon_start

    def get_state(env, obs):
        taxi_row, taxi_col, passenger_idx, destination_idx = env.unwrapped.decode(obs)
        return taxi_row, taxi_col, passenger_idx, destination_idx

    for episode in range(episodes):
        obs, _ = env.reset()
        state = get_state(env, obs)
        done = False
        total_reward = 0

        while not done:
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, ended, truncated, info = env.step(action)
            done = ended or truncated
            next_state = get_state(env, next_obs)

            shaped_reward = 0
            # 檢查撞到障礙物
            if info.get("obstacle_hit", False):
                shaped_reward -= 10000

            taxi_row, taxi_col, passenger_location, destination = next_state

            if action == 4:  # PICKUP
                # 若乘客已在 taxi 中，則錯誤的 pickup
                if passenger_location == 4:
                    shaped_reward -= 10000
                else:
                    # 正確的 pickup: taxi 必須在正確的 pickup 地點
                    correct_pickup_loc = env.unwrapped.locs[passenger_location]
                    if (taxi_row, taxi_col) == tuple(correct_pickup_loc):
                        shaped_reward += 100
                    else:
                        shaped_reward -= 10000

            elif action == 5:  # DROPOFF
                # 若乘客不在 taxi 中，則錯誤的 dropoff
                if passenger_location != 4:
                    shaped_reward -= 10000
                else:
                    # 若乘客在 taxi 中，檢查 taxi 是否在正確的 dropoff 地點
                    correct_dropoff_loc = env.unwrapped.locs[destination]
                    if (taxi_row, taxi_col) != tuple(correct_dropoff_loc):
                        shaped_reward -= 10000

            # 更新總獎勵，將原始 reward 與 shaping reward 結合
            total_reward += (reward + shaped_reward)

            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)

            best_next_action = int(np.argmax(q_table[next_state]))
            # 在 Q-learning 更新中使用 reward + shaped_reward
            q_table[state][action] += alpha * ((reward + shaped_reward) + gamma * q_table[next_state][best_next_action] - q_table[state][action])
            state = next_state

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")

    env.close()
    return q_table, rewards_per_episode

if __name__ == "__main__":
    q_table, rewards = tabular_q_learning(episodes=10000)

    q_table = dict(q_table)
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Q-table saved to q_table.pkl")

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards per Episode")
    plt.show()
