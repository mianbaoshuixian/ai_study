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
特征向量是"实数数据"，可以做运算；整数索引是"编号"，只用来查表。Embedding 的 weight 是 (vocab, dim) 的表，embedding(idx) 就是 weight[idx] ——本质上就是 fancy indexing 取行。