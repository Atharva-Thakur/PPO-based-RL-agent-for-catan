import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic head (Value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value

    def act(self, state, action_mask=None):
        """
        Selects an action based on the current state.
        Args:
            state: Tensor of shape (batch_size, input_dim)
            action_mask: Tensor of shape (batch_size, action_dim), 1 for valid, 0 for invalid
        """
        features = self.feature_extractor(state)
        action_logits = self.actor(features)
        
        if action_mask is not None:
            # Mask invalid actions with a large negative number
            # Ensure mask is boolean or 0/1
            # We want to set logits of invalid actions to -inf
            # action_mask: 1 is valid, 0 is invalid
            
            # Convert mask to float if it isn't
            if action_mask.dtype == torch.bool:
                action_mask = action_mask.float()
                
            # Apply mask: valid actions keep logits, invalid get -1e8
            # We use a large negative number instead of -inf to avoid NaNs in softmax sometimes
            huge_neg = torch.tensor(-1e8, device=action_logits.device, dtype=action_logits.dtype)
            action_logits = torch.where(action_mask > 0.5, action_logits, huge_neg)

        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach(), action_probs.detach()
    
    def evaluate(self, state, action, action_mask=None):
        """
        Evaluates the action for PPO update.
        """
        features = self.feature_extractor(state)
        action_logits = self.actor(features)
        state_values = self.critic(features)
        
        if action_mask is not None:
            if action_mask.dtype == torch.bool:
                action_mask = action_mask.float()
            huge_neg = torch.tensor(-1e8, device=action_logits.device, dtype=action_logits.dtype)
            action_logits = torch.where(action_mask > 0.5, action_logits, huge_neg)
            
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy
