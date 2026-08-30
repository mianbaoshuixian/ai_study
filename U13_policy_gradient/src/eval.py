import argparse
import gymnasium as gym
import numpy as np
import torch
from policy import load_policy


def evaluate(model, episodes, seed):
    env = gym.make("CartPole-v1")
    policy = load_policy(model)
    scores = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        score = 0
        while not done:
            action = policy.greedy_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
        scores.append(score)
    env.close()
    print(f"episodes={episodes} mean={np.mean(scores):.1f} std={np.std(scores):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/policy.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    evaluate(**vars(parser.parse_args()))
