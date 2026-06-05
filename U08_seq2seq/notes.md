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

encoder 输出context vector decoder输出每步的词概率分布
Teacher Forcing = Decoder 训练时，把 ground truth 作为下一步的输入，而不是用模型自己的预测

t:t+1 是 Python 切片的"左闭右开"，[t, t+1) 区间，结果只有 t 这一个元素，但保留了维度。
PyTorch（NumPy 也一样）的索引规则：用单个整数索引会"消掉"那一维，用切片范围会"保留"那一维。

归约/索引操作默认会降维，要保维就显式声明（t:t+1 / keepdim=True）。

线性插值 = 用一个 [0,1] 的系数，在两个值之间"调比例"做加权平均。GRU 把这个思想用在了"旧记忆 vs 新候选"的混合上，让模型自己学会该记还是该忘，还顺带解决了梯度消失。

__init__ 里写的是 "实例化别人写好的类，攒成自己的零件库"； forward 里写的是 "调用这些零件对象，把数据流串起来"

不管多少维，L2 范数 = 各维度平方求和、再开根号。 二维是勾股，三维是两次勾股，n 维是 n 个勾股的合体。几何意义始终是"原点到这个点的直线距离"。