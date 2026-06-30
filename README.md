# GRU 中文翻译模型 · 学习课程

## 学习目标
从零掌握基于 GRU 的 Seq2Seq + Attention 中文翻译模型，能讲清原理（面向面试/作业）。

## 课程单元

| 编号 | 单元 | 过关标准 | 状态 |
|------|------|---------|------|
| U01 | Python 强化：字典/推导式/类 | 独立写出 Vocab 类 | ✅ 已完成 |
| U02 | 数学补课：链式法则 + 矩阵乘法 | 手算 2 层网络反向传播 | ✅ 已完成 |
| U03 | NumPy + 张量思维 | NumPy 实现线性回归 | 🚀 进行中 |
| U04 | PyTorch 基础 + autograd | PyTorch 重写 U03 | ⏳ |
| U05 | MLP 实战：MNIST 分类 | 测试准确率 >95% | ⏳ |
| U06 | 手写 RNNCell + 序列思维 | RNN 做累加和任务 | ✅ 已完成 |
| U07 | GRU 原理 + 公式推导 | 默写 4 个公式并解释 | ✅ 已完成 |
| U08 | Seq2Seq + Teacher Forcing | 数字反转任务跑通 | 🚀 进行中 |
| U09 | 中文数据处理 | 构建中英训练 batch | 🚀 进行中 |
| U10 | 翻译 Baseline（无 Attention） | 能翻短句 | 🚀 进行中 |
| U11 | Bahdanau Attention | BLEU 提升 + 热力图 | 🚀 进行中 |
| U12 | 原理复述 + 面试题整理 | 10 分钟讲解录音 | 🚀 进行中 |

---

## U11 目录结构
```
U11_attention/
├── lesson.ipynb       # Bahdanau Attention 课程（公式推导、Encoder/Decoder 改造、热力图）
├── exercises.ipynb    # 手写 Attention 模块、训练 + 画热力图、默写题
└── notes.md           # 笔记模板
```

## 学习原则
1. **每个练习都手敲**，不复制粘贴
2. **每单元写笔记**（notes.md），用自己的话总结
3. **每单元结束做"复述关卡"**，向 AI 讲解原理
4. 报错先自己读 traceback，读不懂再问

## 目录结构
```
ai_study/
├── README.md              # 本文件
├── U01_python_basics/
│   ├── notes.md
│   ├── ex1_1_count.py
│   ├── ex1_2_listcomp.py
│   └── ex1_3_vocab.py
├── U07_gru/
├── U08_seq2seq/
├── U10_translation_baseline/
├── U11_attention/
│   ├── notes.md
│   ├── exercises.ipynb
│   └── lesson.ipynb
└── U12_review/ ... (后续创建)
```
