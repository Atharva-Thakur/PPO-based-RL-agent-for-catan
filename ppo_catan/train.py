import gymnasium as gym
import catanatron_gym
import torch
import numpy as np
from .agent import PPOAgent
import os
from catanatron.models.player import RandomPlayer
from catanatron.game import Color

def train():
    # Configuration for players
    # Default is 1v1 (Agent vs Random Red). 
    # To play 4 players, uncomment the enemies list below.
    # enemies = [RandomPlayer(Color.RED)] 
    enemies = [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)]
    
    config = {
        "enemies": enemies
    }
    
    env_name = "catanatron_gym:catanatron-v1"
    env = gym.make(env_name, config=config)
    
    # Environment parameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Training with {len(enemies) + 1} players.")
    print(f"State dimension: {state_dim}")

    
    # Hyperparameters
    max_episodes = 10000
    max_timesteps = 1000
    update_timestep = 2000
    lr = 0.0003
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to: {device}")
    
    ppo_agent = PPOAgent(state_dim, action_dim, lr, gamma, K_epochs, eps_clip, device)
    
    time_step = 0
    
    # Logging
    log_interval = 20
    running_reward = 0
    avg_length = 0
    
    # Checkpoint
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    for i_episode in range(1, max_episodes + 1):
        state, info = env.reset()
        current_ep_reward = 0
        
        for t in range(max_timesteps):
            time_step += 1
            
            # Select action with masking
            valid_actions = info['valid_actions']
            action = ppo_agent.select_action(state, valid_actions)
            
            state, reward, terminated, truncated, info = env.step(action)
            
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(terminated)
            
            current_ep_reward += reward
            
            # Update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
                
            if terminated or truncated:
                break
        
        running_reward += current_ep_reward
        avg_length += t
        
        # Logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))
            
            print(f"Episode: {i_episode} \t Avg Length: {avg_length} \t Avg Reward: {running_reward}")
            running_reward = 0
            avg_length = 0
            
        # Save model
        if i_episode % 500 == 0:
            ppo_agent.save(os.path.join(checkpoint_dir, f"ppo_catan_{i_episode}.pth"))
            
    env.close()

if __name__ == '__main__':
    train()
