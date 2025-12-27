import gymnasium as gym
import catanatron_gym
import torch
import numpy as np
from ppo_catan.agent import PPOAgent
import os
from catanatron.models.player import RandomPlayer
from catanatron.game import Color

def evaluate(num_episodes=10, model_path=None, num_players=2):
    # Configure enemies based on num_players
    if num_players == 2:
        enemies = [RandomPlayer(Color.RED)]
    elif num_players == 3:
        enemies = [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE)]
    elif num_players == 4:
        enemies = [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)]
    else:
        raise ValueError("num_players must be 2, 3, or 4")

    config = {"enemies": enemies}
    env = gym.make("catanatron_gym:catanatron-v1", config=config)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device('cpu') # Use CPU for evaluation usually
    
    print(f"Evaluating with {num_players} players. State dim: {state_dim}")
    
    agent = PPOAgent(state_dim, action_dim, device=device)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            agent.load(model_path)
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            print("Mismatch in model architecture likely due to different number of players (state dimension).")
            return
    else:
        print("No model found or provided. Using random/untrained agent.")

    # Inspect env to find agent color
    # print("Env unwrapped attributes:", dir(env.unwrapped))
    
    wins = 0
    total_vps = 0
    
    print(f"Starting evaluation over {num_episodes} episodes...")
    
    for i in range(num_episodes):
        state, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            valid_actions = info['valid_actions']
            action = agent.select_action(state, valid_actions)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
        # Game over
        game = env.unwrapped.game
        winning_color = game.winning_color()
        
        # Agent is always p0 in catanatron_gym
        agent_color = env.unwrapped.p0.color
        num_players = len(game.state.players)
        
        is_winner = (winning_color == agent_color)
        if is_winner:
            wins += 1
            
        print(f"Episode {i+1}: Winner={winning_color}, AgentColor={agent_color}, Won={is_winner}, Steps={steps}, Players={num_players}")

    win_rate = wins / num_episodes
    print(f"\nEvaluation Complete.")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    
    env.close()

if __name__ == "__main__":
    # You can point to a specific checkpoint here
    evaluate(model_path="checkpoints/ppo_catan_1000.pth", num_players=4)
    # evaluate(num_players=2)
