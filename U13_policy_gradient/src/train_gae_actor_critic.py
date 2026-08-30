import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from actor_critic import GAEActorCriticAgent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(values, window):
    if len(values) < window:
        return np.asarray(values, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    env = gym.make("CartPole-v1")
    agent = GAEActorCriticAgent(
        obs_dim=env.observation_space.shape[0],
        hidden_dim=args.hidden_dim,
        action_dim=env.action_space.n,
        gamma=args.gamma,
        lam=args.lam,
        rollout_steps=args.rollout_steps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_coef=args.entropy_coef,
        device=device,
    )

    returns = []
    advantages = []
    episode = 0
    episode_return = 0.0
    obs, _ = env.reset(seed=args.seed)

    # 主循环按 env step 计数，不按 episode 计数：rollout 允许在 episode
    # 中间截断，更新时机由 rollout_steps 决定。
    for _ in range(args.total_steps):
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store(obs, action, reward, next_obs, terminated, done)
        episode_return += reward
        obs = next_obs

        if done:
            episode += 1
            returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
            if episode % args.log_every == 0:
                recent = returns[-args.log_every :]
                print(
                    f"episode={episode:4d} "
                    f"return={returns[-1]:6.1f} "
                    f"mean={np.mean(recent):6.1f} "
                    f"adv={advantages[-1] if advantages else 0.0:.4f}"
                )

        if agent.ready():
            stats = agent.update()
            advantages.append(stats["advantage"])

    env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.checkpoint(), output)

    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "returns": returns,
                "advantages": advantages,
                "gamma": args.gamma,
                "lam": args.lam,
                "rollout_steps": args.rollout_steps,
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
                "entropy_coef": args.entropy_coef,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    plt.figure(figsize=(10, 5))
    plt.plot(returns, alpha=0.25, label="episode return")
    if returns:
        window = min(args.smooth_window, len(returns))
        smoothed = moving_average(returns, window)
        plt.plot(
            range(window - 1, len(returns)),
            smoothed,
            label=f"{window}-episode mean",
        )
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"CartPole-v1 GAE Actor-Critic (lambda={args.lam})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output.with_suffix(".png"), dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--smooth-window", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="artifacts/gae_actor_critic_cartpole.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
