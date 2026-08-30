import argparse
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from policy import PolicyNet
from returns import discount_returns


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args):
    """用 REINFORCE（蒙特卡洛策略梯度）在 CartPole-v1 上训练策略网络。

    核心思路：每回合完整跑完一条轨迹后，用整条轨迹的回报去更新策略——
    让「获得高回报的动作」的概率上升。损失取 ``-(log_prob * return)`` 的均值，
    对它做梯度下降，等价于对期望回报做梯度上升（策略梯度定理）。

    每个 episode 的三步：
      1. 采样：按当前策略与环境交互，记录每步的 log_prob、reward、熵；
      2. 计算回报：``discount_returns`` 做折扣累加并标准化（降低梯度方差）；
      3. 更新：反向传播策略梯度，裁剪梯度范数后用 Adam 更新参数。

    Args:
        args: 命令行参数，包含 episodes、gamma、lr、seed、solve_score、
            max_grad_norm、log_every、out、cpu 等字段。

    产出（写入 ``args.out`` 目录）：
        policy.pt（模型权重与结构超参）、training_curve.png（回报曲线）、
        run.json（本次运行的超参与耗时）。达到 solve_score 会提前停止。
    """
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    policy = PolicyNet().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    returns_history, losses, entropies, grad_norms = [], [], [], []
    start = time.perf_counter()

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        # 采样一条完整轨迹：REINFORCE 是蒙特卡洛方法，必须跑到回合结束才更新
        log_probs, rewards, entropy_values = [], [], []
        done = False
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, log_prob, entropy = policy.sample_action(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            entropy_values.append(entropy)
            rewards.append(reward)
            done = terminated or truncated

        # 折扣回报并标准化，作为每步动作的权重（advantage 的简化替代）
        discounted = discount_returns(rewards, args.gamma).to(device)
        # 策略梯度损失：-(log π(a|s) * G)，最小化它等价于最大化期望回报
        loss = -(torch.stack(log_probs) * discounted).mean()
        optimizer.zero_grad()
        loss.backward()
        # 裁剪梯度范数，防止个别大回报回合造成的更新爆炸
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        episode_return = float(sum(rewards))
        returns_history.append(episode_return)
        losses.append(loss.item())
        entropies.append(torch.stack(entropy_values).mean().item())
        grad_norms.append(float(grad_norm))

        if episode % args.log_every == 0:
            recent = np.mean(returns_history[-100:])
            elapsed = time.perf_counter() - start
            print(f"episode={episode:4d} return={episode_return:6.1f} "
                  f"mean100={recent:6.1f} entropy={entropies[-1]:.3f} "
                  f"grad_norm={grad_norms[-1]:.3f} time={elapsed:.1f}s")
        if len(returns_history) >= 100 and np.mean(returns_history[-100:]) >= args.solve_score:
            print(f"Solved at episode {episode}, mean100={np.mean(returns_history[-100:]):.1f}")
            break

    env.close()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "obs_dim": 4,
                "hidden_dim": 128, "action_dim": 2}, out / "policy.pt")
    plt.plot(returns_history, alpha=0.35, label="return")
    if len(returns_history) >= 10:
        moving = np.convolve(returns_history, np.ones(10) / 10, mode="valid")
        plt.plot(np.arange(9, len(returns_history)), moving, label="mean10")
    plt.axhline(args.solve_score, color="red", linestyle="--", label="solve score")
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "training_curve.png", dpi=150); plt.close()
    with (out / "run.json").open("w") as f:
        json.dump({"seed": args.seed, "episodes": len(returns_history),
                   "gamma": args.gamma, "lr": args.lr, "device": str(device),
                   "elapsed_seconds": time.perf_counter() - start}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solve-score", type=float, default=475)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())

