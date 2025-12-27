# ppo_player.py

import torch
import numpy as np

from catanatron.players.player import Player
from ppo_catan.model import ActorCritic

# You *must* implement `decide(game, legal_actions)`
# Catanatron calls this every time it needs an action.
class PPOPlayer(Player):
    def __init__(self, checkpoint_path="checkpoints/best_model.pth", device="cpu"):
        super().__init__()
        self.device = device
        self.model = ActorCritic()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def decide(self, game, legal_actions):
        """
        game: Catanatron Game object
        legal_actions: list of ints
        """

        # Convert game -> observation vector
        # The `ppo_catan` training code already has this logic.
        # Inspect `ppo_catan/agent.py` for how obs is built during training.
        from ppo_catan.agent import preprocess_obs

        obs = preprocess_obs(game)  # shape: (obs_dim,)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(obs_tensor)

        logits = logits.squeeze(0).cpu().numpy()

        # mask out illegal actions
        mask = np.full_like(logits, -1e9)
        mask[legal_actions] = 0.0
        masked_logits = logits + mask

        # pick highest scoring legal action
        action = int(np.argmax(masked_logits))
        return action
