from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from returns import compute_gae


class PolicyNet(nn.Module):
    """Actor: 输出每个动作的概率。"""

    def __init__(self, obs_dim=4, hidden_dim=128, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return torch.softmax(self.net(obs), dim=-1)


class ValueNet(nn.Module):
    """Critic: 估计状态价值 V(s)。"""

    def __init__(self, obs_dim=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs)


class ActorCriticAgent:
    """按课件第 2.4 节实现的一步 TD Actor-Critic。"""

    def __init__(
        self,
        obs_dim=4,
        hidden_dim=128,
        action_dim=2,
        gamma=0.98,
        actor_lr=2e-4,
        critic_lr=5e-4,
        device="cpu",
    ):
        self.gamma = gamma
        self.device = torch.device(device)
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(obs_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

    def get_action(self, state):
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        probs = self.actor(state_tensor).squeeze(0)
        action = Categorical(probs=probs).sample()
        return action.item(), probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        next_state_tensor = torch.as_tensor(
            next_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        reward_tensor = torch.as_tensor(
            reward, dtype=torch.float32, device=self.device
        )
        done_tensor = torch.as_tensor(
            done, dtype=torch.float32, device=self.device
        )

        # TD target: r_t + gamma * V(s_{t+1}); terminal state has no future value.
        with torch.no_grad():
            td_target = reward_tensor + self.gamma * self.critic(
                next_state_tensor
            ).squeeze(-1) * (1.0 - done_tensor)

        value = self.critic(state_tensor).squeeze(-1)
        critic_loss = 0.5 * (td_target - value).pow(2).mean()

        # delta is the one-step TD error and acts as the actor's advantage.
        delta = td_target - value
        actor_loss = -torch.log(action_prob.clamp_min(1e-8)) * delta.detach()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "td_error": float(delta.detach().cpu()),
        }

    def checkpoint(self):
        return {
            "obs_dim": self.actor.net[0].in_features,
            "hidden_dim": self.actor.net[0].out_features,
            "action_dim": self.actor.net[-1].out_features,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }


class GAEActorCriticAgent:
    """按课件第 2.6 节实现的 GAE Actor-Critic。

    与一步 TD 版本的区别只在更新时机和优势的算法：先攒满 rollout_steps 步
    (允许在 episode 中间截断)，再用 GAE 把这一窗口内的 TD error 按
    (gamma * lam)^l 加权成优势，然后做一次批量梯度更新。
    """

    def __init__(
        self,
        obs_dim=4,
        hidden_dim=128,
        action_dim=2,
        gamma=0.99,
        lam=0.95,
        rollout_steps=128,
        actor_lr=3e-4,
        critic_lr=1e-3,
        entropy_coef=0.01,
        device="cpu",
    ):
        self.gamma = gamma
        self.lam = lam
        self.rollout_steps = rollout_steps
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(obs_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )
        self._reset_buffer()

    def _reset_buffer(self):
        self.buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "next_obs": [],
            "terminated": [],
            "dones": [],
        }

    def get_action(self, state):
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state_tensor).squeeze(0)
        return Categorical(probs=probs).sample().item()

    def store(self, state, action, reward, next_state, terminated, done):
        self.buffer["obs"].append(np.asarray(state, dtype=np.float32))
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["next_obs"].append(np.asarray(next_state, dtype=np.float32))
        self.buffer["terminated"].append(float(terminated))
        self.buffer["dones"].append(float(done))

    def ready(self):
        return len(self.buffer["rewards"]) >= self.rollout_steps

    def update(self):
        obs = torch.as_tensor(
            np.stack(self.buffer["obs"]), device=self.device
        )
        next_obs = torch.as_tensor(
            np.stack(self.buffer["next_obs"]), device=self.device
        )
        actions = torch.as_tensor(
            self.buffer["actions"], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            self.buffer["rewards"], dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            self.buffer["terminated"], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            self.buffer["dones"], dtype=torch.float32, device=self.device
        )

        # 优势只作为权重使用，不需要梯度，所以用 no_grad 的价值估计来算。
        with torch.no_grad():
            values = self.critic(obs).squeeze(-1)
            next_values = self.critic(next_obs).squeeze(-1)

        advantages = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            dones,
            gamma=self.gamma,
            lam=self.lam,
        )
        # A_t = G_t^lambda - V(s_t)，移项就得到 Critic 的回归目标。
        value_targets = advantages + values

        value = self.critic(obs).squeeze(-1)
        critic_loss = 0.5 * (value_targets - value).pow(2).mean()

        weights = advantages
        if len(weights) > 1:
            weights = (weights - weights.mean()) / (
                weights.std(unbiased=False) + 1e-8
            )
        dist = Categorical(probs=self.actor(obs))
        entropy = dist.entropy().mean()
        actor_loss = -(dist.log_prob(actions) * weights).mean()
        actor_loss = actor_loss - self.entropy_coef * entropy

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        steps = len(rewards)
        self._reset_buffer()
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "advantage": float(advantages.abs().mean().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "steps": steps,
        }

    def checkpoint(self):
        return {
            "obs_dim": self.actor.net[0].in_features,
            "hidden_dim": self.actor.net[0].out_features,
            "action_dim": self.actor.net[-1].out_features,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }


def load_actor_critic(path, device="cpu"):
    # The checkpoint is produced by this project and contains plain model
    # state dicts; explicit weights_only=False keeps loading compatible with
    # PyTorch versions whose default is weights_only=True.
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    agent = ActorCriticAgent(
        obs_dim=checkpoint["obs_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        action_dim=checkpoint["action_dim"],
        device=device,
    )
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.actor.eval()
    agent.critic.eval()
    return agent
