# U05 学习笔记 · 神经网络与反向传播实战

> 核心：**回归 → 分类**；MLP 多层；DataLoader；过拟合识别。

## 1. 分类 vs 回归

| 任务 | 输出 | 损失 |
|------|------|------|
| 回归 | 连续值 | MSELoss |
| 二分类 | 2 类 | BCELoss / BCEWithLogitsLoss |
| 多分类 | N 类 | CrossEntropyLoss |

翻译 = 多分类（从词表选词）。

## 2. Softmax

把 logits 变成概率：
- 全部 exp（保证非负）
- 除以总和（归一到 [0,1]，加起来 = 1）

```python
F.softmax(logits, dim=-1)
```

## 3. CrossEntropyLoss

公式：`L = -log(p_true)`

PyTorch 用法：
```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

⚠️ **重点**：
- logits shape `(batch, num_classes)`，**不要加 softmax**
- labels shape `(batch,)`，**整数索引**，不要 one-hot

## 4. MLP 结构

```
Linear → ReLU → Linear → ReLU → Linear（最后不激活）
```

为什么最后不加 softmax？因为 `CrossEntropyLoss` 自带。

## 5. 激活函数

| 函数 | 范围 | 用途 |
|------|------|------|
| ReLU | [0, +∞) | 隐藏层默认 |
| Sigmoid | (0, 1) | 二分类输出、门控 |
| Tanh | (-1, 1) | RNN 常用 |
| Softmax | 概率 | 多分类输出（隐式在 CE Loss 里） |

## 6. Dataset + DataLoader

最小实现：
```python
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

loader = DataLoader(ds, batch_size=32, shuffle=True)
```

术语：
- 1 batch = 1 step
- 所有 batch 训完一遍 = 1 epoch

## 7. 训练模板（带 mini-batch）

```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            ...
```

注意：
- `model.train()` / `model.eval()` 切换模式（Dropout、BatchNorm 行为不同）
- 验证用 `with torch.no_grad():` 关闭梯度追踪，省内存

## 8. 过拟合 vs 欠拟合

| 状态 | train | val |
|------|-------|-----|
| 欠拟合 | 高 | 高 |
| 刚好 ✅ | 低 | 低 |
| 过拟合 | 极低 | 反弹↑ |

观察办法：同时画 train_loss 和 val_loss。

## 9. 卡壳记录
-
-
-
CrossEntropyLoss（最重要，必学）
L = -log(模型给真实类的概率)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
logits	(batch, num_classes)	float，不要加 softmax
labels	(batch,)	int，类别索引（0~num_classes-1），不要 one-hot

loss_fn = nn.BCEWithLogitsLoss()	# 内置了 sigmoid
loss = loss_fn(logits, target)

.unsqueeze(dim) — 增加一个维度
.squeeze(dim) — 去掉长度为 1 的维度

dim=k 表示沿第 k 根轴做运算，但 softmax 不像 mean 会"消除维度"，它保持形状不变，只是让那根轴上的值加起来 = 1
dim=1 就是在竖着的这个维度 所有概率加起来=1

.item() 只用于标量；向量用 .tolist() 或 .numpy()

fancy indexing 的本质：用一个索引数组去"批量取行"。
X = centers[labels] + torch.randn(5, 2)
# 每行 = 对应类的中心 + 一点随机噪声

断言：检查某个条件是不是真的。如果真，啥也不发生；如果假，直接报错停止程序。
assert 条件, "出错时显示的提示"
x = 5
assert x > 0, "x 必须是正数"   # ✅ 通过，没事
x = -3
assert x > 0, "x 必须是正数"   # ❌ AssertionError: x 必须是正数

plt.subplot 切画布 → 
plt.plot 画数据 → 
plt.xlabel/ylabel/title/legend 装饰 → 
plt.show 显示。循环记录每个 epoch 的 loss/acc，传给 plt.plot 就自动画出曲线。

model.train() 做的事：
  1. 检查模型里有没有 Dropout 层？
     有 → Dropout 开始随机丢神经元
     没有 → 什么都没发生
  
  2. 检查模型里有没有 BatchNorm 层？
     有 → BN 用当前 batch 的均值/方差
     没有 → 什么都没发生（你的情况）

model.eval() 做的事：
  1. 让 Dropout 停止工作（所有神经元都在）
  2. 让 BatchNorm 用全局统计量