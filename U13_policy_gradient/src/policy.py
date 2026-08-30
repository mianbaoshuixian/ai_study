from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=4, hidden_dim=128, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def distribution(self, obs):
        return Categorical(logits=self.net(obs))

    def sample_action(self, obs):
        dist = self.distribution(obs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def greedy_action(self, obs):
        with torch.no_grad():
            return self.net(obs).argmax(dim=-1).item()


def load_policy(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device)
    policy = PolicyNet(
        obs_dim=checkpoint["obs_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        action_dim=checkpoint["action_dim"],
    ).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy

