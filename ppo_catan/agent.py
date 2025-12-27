import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .model import ActorCritic

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.action_masks = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.action_masks[:]

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2, device='cpu'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.action_dim = action_dim

    def select_action(self, state, valid_actions_indices):
        # Create mask
        mask = torch.zeros(self.action_dim).to(self.device)
        mask[valid_actions_indices] = 1.0
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, _ = self.policy_old.act(state, mask)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(self.policy_old.critic(self.policy_old.feature_extractor(state)))
        self.buffer.action_masks.append(mask)
        
        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        # Stack lists of tensors
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(self.device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(self.device)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().to(self.device)
        old_masks = torch.stack(self.buffer.action_masks, dim=0).detach().to(self.device)
        
        # Squeeze if necessary (e.g. if batch dim is 1)
        # But stack creates a new dimension, so we might not need squeeze if the original tensors were 1D or scalar
        # state: (input_dim) -> stack -> (batch, input_dim)
        # action: scalar -> stack -> (batch)
        
        # Ensure shapes are correct
        # old_states: (batch, input_dim)
        # old_actions: (batch)
        # old_logprobs: (batch)
        # old_state_values: (batch, 1)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_masks)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - old_state_values.detach().squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
