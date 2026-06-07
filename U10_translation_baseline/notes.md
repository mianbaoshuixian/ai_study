# U10 学习笔记 | 翻译 Baseline（无 Attention）

## 1. 整体架构（自己画图 + 标 shape）



---

## 2. 三处"PAD 屏蔽"

| 位置 | 做法 | 防止什么 |
|---|---|---|
| Embedding |  |  |
| Encoder GRU |  |  |
| Loss |  |  |

---

## 3. pack_padded_sequence 用法

- 三步走：
- 两个硬性要求：

---

## 4. 训练 vs 推理的差异

| 维度 | 训练 | 推理 |
|---|---|---|
| Decoder 输入 |  |  |
| 终止条件 |  |  |
| 是否需要 tgt |  |  |

---

## 5. Greedy Decode 流程



---

## 6. BLEU 简介

- 核心思路：
- 局限：
- 现代替代：

---

## 7. Baseline 的信息瓶颈

- 问题 1：
- 问题 2：
- 问题 3：
- U11 Attention 的核心改造：

---

## 8. 我的疑问 / 自己的总结

