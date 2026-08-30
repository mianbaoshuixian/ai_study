import torch


def discount_returns(rewards, gamma=0.99):
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    return returns


def compute_gae(
    rewards,
    values,
    next_values,
    terminated,
    dones,
    gamma=0.99,
    lam=0.95,
):
    """广义优势估计 (GAE)。

    理论形式是 A_t = sum_l (gamma * lam)^l * delta_{t+l}，实现上用等价的
    倒序递推 A_t = delta_t + gamma * lam * A_{t+1}，一次扫描即可。

    terminated 只标记真正的终止(游戏结束，未来价值为 0)，用来屏蔽
    gamma * V(s_{t+1}) 这一项；dones 额外包含 truncated，用来切断递推，
    避免优势跨 episode 边界往前渗。两者必须分开，否则 CartPole 到达 500
    步被截断时会被误当成"未来毫无价值"。
    """
    advantages = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        delta = (
            rewards[t]
            + gamma * next_values[t] * (1.0 - terminated[t])
            - values[t]
        )
        running = delta + gamma * lam * (1.0 - dones[t]) * running
        advantages[t] = running
    return advantages

