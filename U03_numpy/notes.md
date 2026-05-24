# U03 学习笔记 · NumPy + 张量思维

> 目标：从"一个数一个数算"升级到"整块数据一次算"。
> 核心词：**张量、形状、广播、向量化、batch**

## 1. 张量 Tensor

### 标量 / 向量 / 矩阵 / 张量 的关系
都是ndarray
- 标量：0维
- 向量：1维
- 矩阵：2维
- 张量：3维及以上

## 2. 形状 shape

### 常用操作
- 查看形状：
w.shape  # 不要写 w.shape()
- 改形状 reshape：
w.reshape(3, 4)
- 升维 [None, ...] 或 expand_dims：
w[None, ...]
w = np.expand_dims(w, axis=0)
- 转置 .T：
w.T

## 3. 广播 Broadcasting

### 规则（用自己的话）
把某个维度广播到其他维度，复制多份到其他维度
从右往左，逐维分析
每一对维度满足以下之一：
- 两个维度相等
- 其中一个维度为1
- 其中一个维度不存在（即缺失）
否则广播失败

### 常见场景
- 向量 + 标量
- 矩阵 + 向量（加偏置）
- 批处理

## 4. 向量化 Vectorization

### 为什么比 for 循环快？
- 底层是用 C 语言写的
- 不用循环，直接矩阵运算
- 内存连续，缓存命中率高

### 例子对比
- for 循环写法：
for i in range(len(x)):
    x[i] = x[i] + 1
- 向量化写法：
x = x + 1

## 5. batch 概念

### 为什么要 batch？
不能太多也不能太少
- 太大：显存爆炸，每步太慢，梯度太“平均”容易陷入局部最优
- 太小：噪声大，loss抖动，有时反而更容易跳出局部最优
- 常用：batch size = 2的幂（方便内存对齐），16/32/64/128/256/512

### 数据的第一维永远是 batch_size
- 训练时：
x.shape = (batch_size, n_features)
w.shape = (n_features, 1)
b.shape = (1,)
y_pred.shape = (batch_size, 1)
- 预测时：
x.shape = (1, n_features)
w.shape = (n_features, 1)
b.shape = (1,)
y_pred.shape = (1, 1)

## 6. 线性回归实战

### 前向公式
y_pred = x @ w + b

### 损失（MSE）
L = mean((y_pred - y_true)^2)

### 梯度（自己推一遍）
dL/dw = mean(2 * x * (y_pred - y_true))
dL/db = mean(2 * (y_pred - y_true))

### 训练循环五步
1. 初始化w, b, lr
2. 前向传播 y_pred = x @ w + b
3. 计算损失 L
4. 反向传播，计算梯度
5. 更新w, b

## 7. 卡壳记录
-
-
1 维向量默认加到"最后一维"上。想加到其他维？自己 reshape 成 (..., 1) 明说。
你的总结"(3,) 不行，加一列 (3,1) 就行" —— 完全正确

2 
np.random.uniform(-5, 5, 100)  # 均匀分布 [-5, 5)，每个数概率相等
np.random.randn(100)            # 标准正态分布，集中在 0 附近，远的越少
np.random.randint(0, 10, 100)   # 整数 [0, 10)

3 axis概念
- axis = k表示沿着第k根轴操作，那根轴消失
- shape(2, 3, 4) -> mean(axis=1) -> (2, 4)
- 想保留维度：keepdims=True

4 1D向量没有行列之分
- 1D向量.T不变，转置无效
- np.array([1,2,3]).shape = (3,) -> .T无效
- np.array([[1,2,3]]).shape = (1,3) -> .T有效
- 想区分行/列：用 reshape 或 specify (-1, 1) / (-1, ) / (1, -1) 或[None,:] [:,None]升成2D
w = np.array([1,2,3])
w.reshape(-1, 1)
w[:, None]

5 布尔索引
- b[b<0] = 0  # 负数设为0
- b<0 返回同形状的布尔数组
- 改之前记得.copy(),否则会改原数组





