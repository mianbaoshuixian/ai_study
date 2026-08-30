import argparse
from pathlib import Path
import gymnasium as gym
import torch
from policy import load_policy


def record(model, out, seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=str(out), episode_trigger=lambda _: True)
    policy = load_policy(model)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = policy.greedy_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    print(f"视频已保存到：{out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/policy.pt")
    parser.add_argument("--out", default="artifacts/videos")
    parser.add_argument("--seed", type=int, default=2026)
    record(**vars(parser.parse_args()))

