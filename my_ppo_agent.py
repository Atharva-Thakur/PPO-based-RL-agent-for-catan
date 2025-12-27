# my_ppo_agent.py

import sys
from pathlib import Path

# Add PPO project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from ppo_player import PPOPlayer


def player():
    return PPOPlayer(
        checkpoint_path="checkpoints/ppo_catan_latest.pth",
        device="cpu"
    )
