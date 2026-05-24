# U04 学习笔记 · PyTorch 基础

> 核心认知：**PyTorch ≈ NumPy + 自动求导 + GPU**
> U3 学的 90% 都直接迁移过来。

## 1. Tensor vs ndarray

### 创建对照表
| NumPy | PyTorch |
|-------|---------|
| `np.array(...)` | `torch.tensor(...)` |
| `np.zeros(...)` | `torch.zeros(...)` |
| `np.ones(...)` | `torch.ones(...)` |
| `np.random.randn(...)` | `torch.randn(...)` |
| `np.arange(...)` | `torch.arange(...)` |

### 互转
- NumPy → Tensor：`torch.from_numpy(arr)`
- Tensor → NumPy：`t.numpy()`
- 注意：默认共享内存，要独立用 `.clone()`

### 形状操作（基本一样）
- 看形状：`t.shape`
- 改形状：`t.reshape(...)` 或 `t.view(...)`
- 升维：`t[:, None]` / `t.unsqueeze(dim)`
- 降维：`t.squeeze()`
- 转置：`t.T`

## 2. device / GPU

### 概念
- `device` 属性：`'cpu'` / `'cuda'` / `'cuda:0'`
- 同一运算的 tensor 必须同设备
- `.to(device)` 搬家

### 自适应代码
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
model = model.to(device)
```

## 3. autograd 自动求导（核心）

### 三步
1. `requires_grad=True` 标记追踪
2. 做运算
3. `loss.backward()`，结果在 `.grad`

### 例子
```python
x = torch.tensor(3.0, requires_grad=True)
f = x ** 2
f.backward()
print(x.grad)   # 6
```

### 两个坑
- **梯度累加**：每次 backward 会加到 .grad，要清零 `optimizer.zero_grad()` / `x.grad.zero_()`
- **更新参数要 `with torch.no_grad():`** 或者直接用 `optimizer.step()`

## 4. nn.Module

### 用现成的
- `nn.Linear(in, out)`：自动管理 W、b
- `nn.ReLU()`、`F.relu(x)`
- `nn.MSELoss()`、`nn.CrossEntropyLoss()`

### 自定义网络
```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

## 5. 优化器

- `torch.optim.SGD(model.parameters(), lr=0.01)`：朴素梯度下降
- `torch.optim.Adam(model.parameters(), lr=0.001)`：默认首选

## 6. 训练循环模板（背下来！）

```python
for epoch in range(epochs):
    optimizer.zero_grad()        # 1. 清零旧梯度
    y_pred = model(x)            # 2. 前向
    loss = loss_fn(y_pred, y)    # 3. 算 loss
    loss.backward()              # 4. 自动算梯度
    optimizer.step()             # 5. 自动更新
```

**这个模板从现在到 GRU、Transformer 都不会变。**

## 7. U3 vs U4 对照

| U3 (NumPy) | U4 (PyTorch) |
|---|---|
| 手推梯度 dw、db | `loss.backward()` |
| `w = w - lr * dw` | `optimizer.step()` |
| 自己写 forward | `nn.Linear` |
| CPU only | `.to('cuda')` 切 GPU |

## 8. 卡壳记录
-
-
-
torch.rand	[0, 1)	均匀
torch.randn	(-∞, +∞)	均值为 0、标准差为 1 的标准正态分布
N(μ, σ²) 是数学约定，括号里第二个数是方差。randn() * k 的标准差是 k，方差是 k²
torch.randint(low, high, size)	整数 [low, high)	均匀
torch.empty(...).uniform_(a, b)	[a, b)	均匀

不带下划线的方法都"返回新对象"，但新对象多半和原对象共享内存。 带下划线的方法（如 zero_、add_、reshape_）才是原地修改。

想要独立怎么办？
显式 clone / copy：
a = t.numpy().copy()           # ndarray 独立
t = torch.from_numpy(a).clone() # tensor 独立

PyTorch	(out, in)	y = x @ W.T + b
数学教材	(in, out)	y = x @ W + b
NumPy 风格	(in, out)	y = x @ W + b
nn.Linear(in, out) 的 weight.shape = (out, in)，用时要转置。

# 强制指定 dtype
torch.from_numpy(arr).float()            # 转 float32
