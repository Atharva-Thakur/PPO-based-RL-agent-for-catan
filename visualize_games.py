"""
Visualize games with web GUI - integrates with evaluate_agent
"""
import gymnasium as gym
import catanatron_gym
import torch
import numpy as np
from ppo_catan.agent import PPOAgent
import os
import time
import threading
from catanatron.models.player import RandomPlayer
from catanatron.game import Color
from web_visualizer.server import socketio, broadcast_game_state, broadcast_game_end
from web_visualizer.game_state_extractor import extract_game_state

def visualize_games(num_episodes=10, model_path=None, num_players=4, delay=0.5):
    """
    Run game evaluation with real-time web visualization
    
    Args:
        num_episodes: Number of games to play
        model_path: Path to trained model checkpoint
        num_players: Number of players (2-4)
        delay: Delay in seconds between steps for visualization (0 for no delay)
    """
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
    device = torch.device('cpu')
    
    print(f"Visualizing with {num_players} players. State dim: {state_dim}")
    
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

    wins = 0
    
    print(f"Starting visualization over {num_episodes} episodes...")
    print("Open your browser to http://localhost:5000 to view the games!")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        step = 0
        
        # Broadcast initial game state
        game = env.unwrapped.game
        game_state = extract_game_state(game, episode=episode+1, step=step, action_taken="Game started")
        broadcast_game_state(game_state)
        time.sleep(delay * 2)  # Pause at start
        
        while not done:
            valid_actions = info['valid_actions']
            action = agent.select_action(state, valid_actions)
            
            # Get action description
            action_desc = f"Step {step}: Agent took action {action}"
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # Extract and broadcast game state
            game = env.unwrapped.game
            game_state = extract_game_state(game, episode=episode+1, step=step, action_taken=action_desc)
            broadcast_game_state(game_state)
            
            # Add delay for visualization
            if delay > 0:
                time.sleep(delay)
            
        # Game over
        game = env.unwrapped.game
        winning_color = game.winning_color()
        agent_color = env.unwrapped.p0.color
        
        is_winner = (winning_color == agent_color)
        if is_winner:
            wins += 1
        
        # Broadcast final state
        final_state = extract_game_state(game, episode=episode+1, step=step, 
                                        action_taken=f"Game Over! Winner: {winning_color.name}")
        broadcast_game_state(final_state)
        broadcast_game_end({'winner': winning_color.name, 'episode': episode+1})
        
        print(f"Episode {episode+1}: Winner={winning_color.name}, AgentColor={agent_color.name}, Won={is_winner}, Steps={step}")
        
        # Pause between games
        time.sleep(delay * 3)

    win_rate = wins / num_episodes
    print(f"\nVisualization Complete.")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    
    env.close()

def start_server_and_visualize(num_episodes=10, model_path=None, num_players=4, delay=0.5):
    """
    Start the Flask server and run game visualization
    """
    from web_visualizer.server import run_server
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    print("Server starting... waiting 2 seconds...")
    time.sleep(2)
    
    # Run visualization in main thread
    visualize_games(num_episodes, model_path, num_players, delay)

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "checkpoints/ppo_catan_500.pth"  # Change to your model
    NUM_EPISODES = 5
    NUM_PLAYERS = 4
    DELAY = 0.1  # Seconds between steps (lower = faster, 0 = no delay)
    
    start_server_and_visualize(
        num_episodes=NUM_EPISODES,
        model_path=MODEL_PATH,
        num_players=NUM_PLAYERS,
        delay=DELAY
    )
