# U09 学习笔记 | 中文数据处理与训练 batch

## 1. 平行语料长什么样

（自己写：源语言/目标语言、对齐方式、常见数据集）

---

## 2. 分词

- 英文怎么切：
- 中文为什么不能直接 split：
- 字符级 vs 词级（jieba）vs 子词级（BPE）的优缺点：

---

## 3. 词表 Vocab

- 4 个特殊 token 及作用：
  - `<pad>`：
  - `<sos>`：
  - `<eos>`：
  - `<unk>`：
- 为什么 `<pad>` 设为 id=0：

---

## 4. 句子 → id

- src 加什么特殊 token：
- tgt 加什么特殊 token：
- 为什么不一样：

---

## 5. Padding 与动态 padding

- 全局 padding 的问题：
- 动态 padding 怎么实现（哪个组件负责）：
- 为什么要保存 `src_len`：

---

## 6. Dataset / DataLoader / collate_fn

- Dataset 的两个必要方法：
- collate_fn 在做什么：
- 为什么要按 src 长度降序：

---

## 7. 训练 batch 错位

```
tgt     = [<sos>, w1, w2, w3, <eos>]
tgt_in  = [<sos>, w1, w2, w3]       # decoder 输入
tgt_out = [   w1, w2, w3, <eos>]    # loss 目标
```

- 这种错位的本质是：
- `CrossEntropyLoss(ignore_index=PAD)` 的作用：

---

## 8. 数据流（自己默写）

```
原始字符串 -> ... -> (src, src_len, tgt)  shape: ?
```

---

## 9. 我的疑问 / 总结
