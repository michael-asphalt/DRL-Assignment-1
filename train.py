import numpy as np
import random
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

# 引入自定義的 Taxi 環境
from simple_custom_taxi_env import SimpleTaxiEnv

# 建立一個簡單的 action_space 物件，提供 .n 與 sample() 方法
class ActionSpace:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return random.randint(0, self.n - 1)

def get_state(obs):
    """
    由於自定義環境的 obs 已經是一個 tuple，
    直接回傳即可作為 Q-table 的 key。
    """
    return tuple(obs)

def tabular_q_learning(episodes=5000, alpha=0.1, gamma=0.99,
                         epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.9997,
                         fuel_limit=5000, grid_size=5):
    """
    使用 Tabular Q-Learning 訓練自定義的 Taxi 環境。
    主要修改重點：
      - 使用 SimpleTaxiEnv 而非 Gym 的 Taxi-v3。
      - 設定自定義 action_space (6 個動作)。
      - 直接使用環境回傳的 reward，不做額外 shaping。
    """
    # 建立環境，根據 spec 可調整 grid_size 與 fuel_limit
    env = SimpleTaxiEnv(grid_size=grid_size, fuel_limit=fuel_limit)
    # 為環境補上 action_space 屬性
    env.action_space = ActionSpace(6)
    
    obs, _ = env.reset()
    q_table = defaultdict(lambda: np.zeros(6))  # 6 個動作
    rewards_per_episode = []
    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _ = env.reset()
        state = get_state(obs)
        done = False
        total_reward = 0

        while not done:
            # 若 state 尚未出現在 Q-table 中，先初始化
            if state not in q_table:
                q_table[state] = np.zeros(6)
            
            # ε-greedy 策略選擇動作
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
            
            next_obs, reward, done, info = env.step(action)
            next_state = get_state(next_obs)
            
            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)
            
            best_next_action = int(np.argmax(q_table[next_state]))
            # Q-learning 更新公式：直接使用環境提供的 reward
            q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])
            
            state = next_state
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")
    
    return q_table, rewards_per_episode

if __name__ == "__main__":
    episodes = 10000  # 可根據需要調整訓練輪數
    q_table, rewards = tabular_q_learning(episodes=episodes)
    q_table = dict(q_table)
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Q-table saved to q_table.pkl")

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards per Episode")
    plt.show()
