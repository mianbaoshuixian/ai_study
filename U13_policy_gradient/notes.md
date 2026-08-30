# U13 | Vanilla REINFORCE：用策略梯度学会立住倒立摆

## 目标

在 `CartPole-v1` 上直接学习策略。每一步输入 `obs = [x, x_dot, theta, theta_dot]`，策略输出左推/右推的概率；一局结束后，根据每个动作之后获得的折扣回报更新网络。

## 交互循环

```text
obs -> policy -> action -> env.step(action)
 ↑                         ↓
 └──── next_obs, reward ───┘
```

- `obs` 是 observation（观测），形状 `(4,)`。
- `x`：小车位置；`x_dot`：小车速度；`theta`：杆子角度；`theta_dot`：杆子角速度。
- `episode` 是从 `reset()` 到结束的一局；`step` 是其中一次动作。

## 核心公式

策略网络给出 logits，`Categorical(logits=logits)` 隐式做 softmax，得到 `πθ(a|s)`。

从时刻 `t` 开始的 reward-to-go：

```text
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
```

目标是最大化好动作的概率，因此写成 PyTorch 要最小化的损失：

```python
loss = -(log_prob * returns).mean()
```

- `log_prob`：当时实际动作的 `log πθ(a_t|s_t)`。
- `returns`：对应动作之后的折扣回报 `G_t`。
- `backward()`：计算策略梯度；`optimizer.step()`：更新参数。

## 训练日志怎么看

- `return / length`：CartPole 每步奖励为 1，因此一局回报通常等于坚持步数。
- `entropy`：动作分布的不确定性；接近 `0.693` 表示左右都犹豫，接近 `0` 表示几乎只选一个动作。低熵且低回报可能是策略塌缩。
- `grad_norm`：这次更新整体梯度大小；过大可能震荡，接近 0 需检查计算图。
- `episode`：已经训练的局数。
- `time`：真实经过的墙钟时间，不是环境里的步数。

## Actor-Critic：从状态 baseline 到一步 TD

REINFORCE 使用完整回报 `G_t` 更新 Actor。课件第 2.4 节把状态相关
baseline 写成可学习的价值网络 `V_w(s)`，并拆成两个角色：

- Actor `pi_theta(a|s)`：根据状态选择动作；
- Critic `V_w(s)`：评价当前状态的价值。

一步 TD 目标和 TD 误差分别是：

```text
target_t = r_t + gamma * V_w(s_{t+1})
delta_t  = target_t - V_w(s_t)
```

终止状态没有未来价值，因此实际实现使用：

```text
target_t = r_t + gamma * V_w(s_{t+1}) * (1 - done)
```

Actor 使用 `delta_t` 作为优势的近似：

```text
actor_loss  = -log pi_theta(a_t | s_t) * delta_t
critic_loss = 1/2 * (target_t - V_w(s_t))^2
```

与 REINFORCE 等完整 episode 结束后再更新不同，一步 TD Actor-Critic
在每次 `env.step` 后即可同时更新 Actor 和 Critic。`delta_t` 为正时，
当前动作比 Critic 的预期更好，提升该动作概率；为负时则降低。

## 运行

在本目录执行：

```bash
python src/train.py
python src/eval.py --model artifacts/policy.pt
python src/play.py --model artifacts/policy.pt
```

`artifacts/` 中会生成权重、曲线和 `run.json`；视频在 `artifacts/videos/`。

## 常见坑

1. `log_prob` 不能 `.detach()`，否则无法回传策略梯度。
2. REINFORCE 的负号不能漏：优化器做最小化，而目标是最大化回报。
3. `terminated` 和 `truncated` 都要作为一局结束处理。
4. 训练时采样动作，评测/录制时用贪心动作，便于稳定观察效果。
