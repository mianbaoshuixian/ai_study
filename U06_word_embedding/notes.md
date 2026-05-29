# U06 笔记 | 词向量与 RNN 基础

## 1. 从表格到序列：任务变了

前面学的是：**一个样本 → 一个标签**（分类/回归）
翻译要做的是：**变长序列 → 变长序列**（Seq2Seq）

---

## 2. 词向量（Word Embedding）

### One-Hot 的问题
- 词表1万 → 每词1万维向量，99.99%是0
- 词之间毫无语义关联

### Embedding 的核心思想
- 用小维度稠密向量（如64维）表示每个词
- 向量是**可学习的参数**，训练中自动调整
- "语义相近的词，向量也相近"

### nn.Embedding 用法
```python
embedding = nn.Embedding(vocab_size=10000, embed_dim=64)
word_index = torch.tensor([0, 1, 2])   # 词索引
vecs = embedding(word_index)           # (3, 64)
```

**本质**：查表。权重矩阵 shape `(vocab_size, embed_dim)`，索引 i → 取第 i 行。

---

## 3. RNN 基础

### 核心思想
每读一个词，更新一次"记忆"（隐藏状态）：

```
词1 → RNN → h1
           ↓
词2 → RNN → h2
           ↓
词3 → RNN → h3
```

### 三个变量
| 变量 | 含义 | 形状 |
|------|------|------|
| x_t | t时刻的输入（词向量）| (batch, embed_dim) |
| h_t | t时刻的隐藏状态 | (batch, hidden_dim) |
| h_{t-1} | 上一时刻的隐藏状态 | (batch, hidden_dim) |

### RNN 计算公式
```python
h_t = torch.tanh(x_t @ W_xh.T + h_{t-1} @ W_hh.T + b)
```

### PyTorch 的 nn.RNN
```python
rnn = nn.RNN(input_size=64, hidden_size=128, batch_first=True)
output, hn = rnn(x, h0)
# output: (batch, seq_len, hidden_dim) 每个时刻的h
# hn: (1, batch, hidden_dim) 最后一个时刻的h
```

---

## 4. BPTT：RNN 的反向传播

**BPTT = Backpropagation Through Time**，沿时间维度展开反向传播。

问题：序列太长时，前面词的梯度衰减（梯度消失）→ LSTM/GRU 的动机。

---

## 5. 变长序列处理

### Padding（填充）
短句补0到统一长度，简单但有冗余计算。

### Packing（打包）
只保留真实长度，不浪费计算：
```python
packed = pack_padded_sequence(padded, lengths, batch_first=True)
output, hn = rnn(packed)
```

---

## 6. 我的疑问和理解

（课堂上或练习后填写）

---
embedding.weight 本身就是一个完整的词向量表
nn.Embedding(vocab_size, embed_dim) = 创建一张 vocab_size 行 × embed_dim 列 的可学习表格

输入：词索引（0~vocab_size-1）
输出：查表，取对应行的 embed_dim 维向量

特征向量是"实数数据"，可以做运算；整数索引是"编号"，只用来查表。Embedding 的 weight 是 (vocab, dim) 的表，embedding(idx) 就是 weight[idx] ——本质上就是 fancy indexing 取行。

embedding 的 forward 接收的是一个 2D（或 1D）tensor，里面装的是整数词索引。

Embedding 不关心你传的是 batch 还是 seq_len，它只看到"一堆要查的索引"。输入 shape 是什么样，输出就在最后多一维 embed_dim。
关键直觉: 公式 $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$ 描述的是一个时间步内的事情,所以输入输出都没有 seq_len。

output.shape: (batch, seq_len, hidden_size)。

hn.shape: (num_layers * num_directions, batch, hidden_size)

模型架构层面
  ├─ MLP    全连接, 处理固定长度输入
  ├─ CNN    卷积, 擅长图像
  ├─ RNN    循环, 擅长序列 ← 你在这里
  ├─ LSTM/GRU  RNN 的改进版
  └─ Transformer  自注意力, 当前最强

LLM ⊂ NLP,LLM 用的是 Transformer 架构


为什么训练时要 Dropout，推理时不要
训练时：Dropout 随机关掉一些神经元 → 模型不能过度依赖某一个特征 → 防止过拟合（正则效果）

推理时：需要稳定、可复现的结果 → 不能让预测一会儿对一会儿错