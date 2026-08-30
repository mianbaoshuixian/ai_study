import argparse

import gymnasium as gym
import torch

from actor_critic import load_actor_critic


def evaluate(model_path, episodes, seed, device):
    env = gym.make("CartPole-v1")
    agent = load_actor_critic(model_path, device=device)
    returns = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        while not done:
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action = agent.actor(obs_tensor).argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        returns.append(episode_return)

    env.close()
    print(
        f"episodes={episodes} "
        f"mean_return={sum(returns) / len(returns):.1f} "
        f"min={min(returns):.1f} max={max(returns):.1f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.episodes, args.seed, args.device)
