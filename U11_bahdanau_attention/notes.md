# U11 学习笔记 | Bahdanau Attention

## 1. 全局图：U10 -> U11 改了哪里

用自己的话画出：

```text
src -> Encoder -> ?
Decoder hidden + ? -> Attention -> ? -> Decoder -> logits
```

---

## 2. 四个核心张量

| 张量 | shape | 我的理解 |
|---|---|---|
| encoder_outputs |  |  |
| decoder_hidden |  |  |
| attn_weights |  |  |
| context |  |  |

---

## 3. Bahdanau Attention 三步

1. 打分：
2. softmax：
3. 加权求和：

---

## 4. 为什么要 mask PAD

- 不 mask 会发生什么：
- mask 的代码是哪一行：

---

## 5. 专业分词器

| 语言 | 分词器 | 为什么不用简单 split/list |
|---|---|---|
| 中文 | jieba |  |
| 英文 | MosesTokenizer |  |

---

## 6. Attention 热力图怎么读

- 横轴：
- 纵轴：
- 颜色：
- 我观察到的一个对齐现象：

---

## 7. 我的疑问 / 自己的总结

