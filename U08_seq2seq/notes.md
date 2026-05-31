# U08 学习笔记 | Seq2Seq + Teacher Forcing

## 1. 为什么需要 Seq2Seq

（用自己的话写：RNN 的局限是什么，Seq2Seq 怎么解决的）

---

## 2. Encoder

- 作用：
- 输出什么：
- context vector 的 shape：

---

## 3. Decoder

- 作用：
- 每步的输入是：
- 初始 hidden 来自：
- 何时停止：

---

## 4. Teacher Forcing

- 是什么：
- 训练时下一步输入是：
- 推理时下一步输入是：
- teacher_forcing_ratio 的含义：

---

## 5. 曝光偏差（Exposure Bias）

（用自己的话解释这个问题）

---

## 6. 数据流总结

（自己画出或写出 Seq2Seq 的完整数据流，包括 shape）

```
src (...) -> ... -> tgt (...)
```

---

## 7. 我的疑问 / 自己的总结

（自由发挥）
