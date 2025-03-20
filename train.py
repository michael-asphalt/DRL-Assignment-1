import numpy as np
import random
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

from simple_custom_taxi_env import SimpleTaxiEnv

class ActionSpace:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return random.randint(0, self.n - 1)

def get_state(obs):
    return tuple(obs)

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
        state = get_state(obs)
        done = False
        total_reward = 0

        while not done:
            # intitialize Q-table for unseen states
            if state not in q_table:
                q_table[state] = np.zeros(6)
        
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
            
            next_obs, reward, done, info = env.step(action)
            next_state = get_state(next_obs)
            
            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)
            
            best_next_action = int(np.argmax(q_table[next_state]))
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
    episodes = 10000  
    q_table, rewards = tabular_q_learning(episodes=episodes, grid_size=10)
    q_table = dict(q_table)
    file_name = "q_table-10.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(q_table, f)
    print("Q-table saved to q_table-10.pkl")

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Rewards per Episode (grid_size=10)")
    plt.show()

    # for i in range(5, 11):
    #     q_table, rewards = tabular_q_learning(episodes=episodes, grid_size=i)
    #     q_table = dict(q_table)
    #     file_name = f"q_table-{i}.pkl"
    #     with open(file_name, "wb") as f:
    #         pickle.dump(q_table, f)
    #     print(f"Q-table saved to {file_name}")

    #     plt.plot(rewards)
    #     plt.xlabel("Episode")
    #     plt.ylabel("Total Reward")
    #     plt.title(f"Rewards per Episode (grid_size={i})")
    #     plt.show()
