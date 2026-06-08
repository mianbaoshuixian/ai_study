# U07 笔记 | GRU 原理与公式推导

## 1. RNN 的遗留问题

梯度消失原因：反向传播要穿越多个 tanh，tanh 导数在 (0,1)，连乘后趋近于 0。

结果：序列越长，远处词的梯度越小，模型学不到长距离依赖。

---

## 2. GRU 的解决思路

加入两个「门」，让模型**自己决定**记多少、忘多少。

门 = sigmoid 输出的 (0,1) 向量，逐元素乘以信息，控制信息通过量。

⊙ = 逐元素乘法

---

## 3. 四个公式

| 编号 | 名称 | 公式 |
|------|------|------|
| ① | 重置门 | $r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$ |
| ② | 更新门 | $z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$ |
| ③ | 候选隐状态 | $\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$ |
| ④ | 新隐藏状态 | $h_t = z_t \odot h_{t-1} + (1-z_t) \odot \tilde{h}_t$ |

**计算顺序**：① → ② → ③ → ④

> 本教材采用 **PyTorch `nn.GRU` 约定**（z 高 → 保留旧记忆）。原论文 Cho 2014 / Wikipedia 是相反约定，仅命名不同，数学等价。

### 各公式作用

- **重置门 r_t**：控制计算候选状态时，旧记忆保留多少。r≈0 忘掉过去，r≈1 保留过去。
- **更新门 z_t**：控制新旧状态的混合比例。z≈1 保持旧记忆，z≈0 全部替换为新候选。
- **候选状态 h~**：如果要更新，打算更新成什么。用 r_t 过滤了旧记忆，比 RNN 更有选择性。
- **新隐状态 h_t**：线性插值合并旧记忆和候选。**z×旧 + (1-z)×新**

### 为什么能解决梯度消失

z_t ≈ 1 时，h_t ≈ h_{t-1}，梯度直通，不经过 tanh，不衰减。

---

## 4. GRU vs RNN 参数量

| 模型 | 参数量 |
|------|--------|
| RNN  | input×hidden + hidden×hidden + 2×hidden |
| GRU  | 3 × RNN 参数量（3组门各一套参数）|
| LSTM | 4 × RNN 参数量 |

---

## 5. PyTorch nn.GRU

```python
gru = nn.GRU(input_size, hidden_size, batch_first=True)
output, hn = gru(x)
# output: (batch, seq_len, hidden_size)
# hn:     (num_layers, batch, hidden_size)
```

接口与 nn.RNN 完全一致，替换只需改一行。

---

## 6. 卡壳记录

（学习过程中填写）

-
-
-
