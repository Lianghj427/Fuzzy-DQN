import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from dqn import DQNAgent
from fuzzy_dqn import FuzzyDQNAgent
import time

def train_agent(env, agent, episodes=500, max_steps=1000):
    rewards_history = []
    best_reward = -float('inf')
    start_time = time.time()
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            pole_angle = abs(next_state[2])
            position = abs(next_state[0])
            angle_reward = 0.1 * (0.5 - min(pole_angle, 0.5))
            position_penalty = 0.05 * min(position, 1.0)
            
            modified_reward = reward + angle_reward - position_penalty
            
            agent.memory.push(state, action, modified_reward, next_state, done)
            agent.update_model()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        agent.update_epsilon()
        rewards_history.append(episode_reward)
        
        current_avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
        
        if current_avg_reward > best_reward:
            best_reward = current_avg_reward
        
        if (episode + 1) % 10 == 0 or episode == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Avg Reward (last 100 ep): {current_avg_reward:.2f}, Best: {best_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Time: {elapsed_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    return rewards_history

def plot_comparison(dqn_rewards, fuzzy_dqn_rewards, window_size=10):

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    dqn_ma = moving_average(dqn_rewards, window_size)
    fuzzy_dqn_ma = moving_average(fuzzy_dqn_rewards, window_size)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_ma, label='DQN', color='blue', linewidth=2)
    plt.plot(fuzzy_dqn_ma, label='Fuzzy DQN', color='red', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward (over last 100 episodes)', fontsize=12)
    plt.title('DQN vs Fuzzy DQN Average Reward Comparison', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)
    
    max_dqn_idx = np.argmax(dqn_ma)
    max_fuzzy_idx = np.argmax(fuzzy_dqn_ma)
    
    plt.annotate(f'DQN Max: {dqn_ma[max_dqn_idx]:.2f}', 
                 xy=(max_dqn_idx, dqn_ma[max_dqn_idx]),
                 xytext=(max_dqn_idx-50, dqn_ma[max_dqn_idx]+20),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8))
    
    plt.annotate(f'Fuzzy DQN Max: {fuzzy_dqn_ma[max_fuzzy_idx]:.2f}', 
                 xy=(max_fuzzy_idx, fuzzy_dqn_ma[max_fuzzy_idx]),
                 xytext=(max_fuzzy_idx-50, fuzzy_dqn_ma[max_fuzzy_idx]-20),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))
    
    plt.tight_layout()
    plt.savefig('avg_reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_models():

    env = gym.make('CartPole-v1', render_mode=None)
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    episodes = 500
    
    dqn_agent = DQNAgent(state_dim, action_dim, device)
    dqn_agent.train()
    print("\nTraining standard DQN agent...")
    dqn_rewards = train_agent(env, dqn_agent, episodes=episodes)
    
    dqn_agent.save('model/dqn_model.pth')
    
    fuzzy_dqn_agent = FuzzyDQNAgent(state_dim, action_dim, device)
    fuzzy_dqn_agent.train()
    print("\nTraining Fuzzy DQN agent...")
    fuzzy_dqn_rewards = train_agent(env, fuzzy_dqn_agent, episodes=episodes)
    
    fuzzy_dqn_agent.save('model/fuzzy_dqn_model.pth')
    
    plot_comparison(dqn_rewards, fuzzy_dqn_rewards, window_size=10)
    
    env.close()

if __name__ == "__main__":

    train_models()
    print("Training and evaluation completed. Check 'avg_reward_comparison.png' for the performance graph.")
