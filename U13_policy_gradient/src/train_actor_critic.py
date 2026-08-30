import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from actor_critic import ActorCriticAgent


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
    agent = ActorCriticAgent(
        obs_dim=env.observation_space.shape[0],
        hidden_dim=args.hidden_dim,
        action_dim=env.action_space.n,
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        device=device,
    )

    returns = []
    td_errors = []
    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        episode_return = 0.0
        episode_td_errors = []

        while not done:
            action, action_prob = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            stats = agent.update(obs, action_prob, reward, next_obs, done)
            episode_td_errors.append(stats["td_error"])
            episode_return += reward
            obs = next_obs

        returns.append(episode_return)
        td_errors.extend(episode_td_errors)
        if episode % args.log_every == 0:
            recent = returns[-args.log_every :]
            print(
                f"episode={episode:4d} "
                f"return={episode_return:6.1f} "
                f"mean={np.mean(recent):6.1f} "
                f"td={np.mean(np.abs(episode_td_errors)):.4f}"
            )

    env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.checkpoint(), output)

    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "returns": returns,
                "td_errors": td_errors,
                "gamma": args.gamma,
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
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
    plt.title("CartPole-v1 Actor-Critic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output.with_suffix(".png"), dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--smooth-window", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="artifacts/actor_critic_cartpole.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
