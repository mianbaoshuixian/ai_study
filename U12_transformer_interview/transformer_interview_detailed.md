# U12 · Transformer 面试题 · 15 分钟详尽口述讲稿

> 这是 `transformer_interview.md` 的「展开版」。原文件是**速查提纲**（盖住自测用）；本文件是**逐题讲透的口述脚本**，每题按「能讲满 15 分钟、零基础也能听懂」的标准写。
>
> **怎么用**：对着每一节从头念，遇到 `🗣️` 是可以照着说的口语；遇到 `📐` 是要写在白板上的推导；配图以内联 SVG 形式直接嵌在正文里，用支持渲染的 Markdown 预览器即可看到彩色图。
>
> **讲解节奏建议**：直觉比喻（3 分钟）→ 严谨推导/数值例子（7 分钟）→ 形状与代码落地（3 分钟）→ 追问与陷阱（2 分钟）。

---

# Q1. 手推 Self-Attention 的完整计算过程

## 第一步：先用一个比喻把「注意力」讲活（直觉，约 3 分钟）

🗣️ 「我先不写公式，先讲个场景。假设你走进一个图书馆，脑子里有个问题：『我想找讲猫的书』。这个问题，就是 **Query**。

书架上每本书的书脊上都有个标签，比如『猫』『鱼』『吃』，这些标签就是 **Key**。你会拿你的问题（Query）去和每本书的标签（Key）比一比，看哪个最匹配。比对出来的匹配程度，就是**注意力分数**。

匹配完之后，你不会把书脊标签搬走，你要的是书里的**正文内容**——那就是 **Value**。最后你按匹配程度，从每本书里『按比例』取内容：和『猫』最相关就多取一点，不相关的少取一点。把这些加权取来的内容拼成一份新的笔记，这份笔记就是这个位置的 **注意力输出**。」

🗣️ 「所以一句话总结三者：**Query 是我要找什么，Key 是每个位置能被找到的标识，Value 是每个位置真正的内容。分数只决定『拿多少』，Value 决定『拿什么』。**」

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430" viewBox="0 0 760 430" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="ar" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#555"/>
    </marker>
  </defs>
  <rect x="1" y="1" width="758" height="428" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="32" text-anchor="middle" font-size="16" fill="#333">图书馆比喻：Q / K / V 在做什么</text>
  <rect x="40" y="70" width="170" height="80" fill="#fff3cd" stroke="#caa000" rx="8"/>
  <text x="125" y="100" text-anchor="middle" font-size="14" fill="#333">Query（我想找什么）</text>
  <text x="125" y="124" text-anchor="middle" font-size="12" fill="#777">"我在找讲猫的书"</text>
  <text x="125" y="142" text-anchor="middle" font-size="11" fill="#999">= 当前 token 的提问</text>
  <rect x="300" y="60" width="150" height="44" fill="#d6e4ff" stroke="#3366cc" rx="6"/>
  <text x="375" y="80" text-anchor="middle" font-size="12">Key₁：标签"猫"</text>
  <text x="375" y="97" text-anchor="middle" font-size="10" fill="#888">Value₁：关于猫的内容</text>
  <rect x="300" y="120" width="150" height="44" fill="#d6e4ff" stroke="#3366cc" rx="6"/>
  <text x="375" y="140" text-anchor="middle" font-size="12">Key₂：标签"鱼"</text>
  <text x="375" y="157" text-anchor="middle" font-size="10" fill="#888">Value₂：关于鱼的内容</text>
  <rect x="300" y="180" width="150" height="44" fill="#d6e4ff" stroke="#3366cc" rx="6"/>
  <text x="375" y="200" text-anchor="middle" font-size="12">Key₃：标签"吃"</text>
  <text x="375" y="217" text-anchor="middle" font-size="10" fill="#888">Value₃：关于吃的内容</text>
  <line x1="210" y1="110" x2="300" y2="82" stroke="#cc4444" stroke-width="2.5" marker-end="url(#ar)"/>
  <text x="250" y="92" font-size="11" fill="#cc4444">匹配度高 0.8</text>
  <line x1="210" y1="115" x2="300" y2="142" stroke="#999" stroke-dasharray="3,3" marker-end="url(#ar)"/>
  <text x="250" y="140" font-size="11" fill="#999">0.05</text>
  <line x1="210" y1="120" x2="300" y2="202" stroke="#999" stroke-dasharray="3,3" marker-end="url(#ar)"/>
  <text x="250" y="195" font-size="11" fill="#999">0.15</text>
  <rect x="520" y="60" width="200" height="164" fill="#e2f0d9" stroke="#5a9e3a" rx="8"/>
  <text x="620" y="84" text-anchor="middle" font-size="13" fill="#333">softmax 后的权重</text>
  <text x="620" y="110" text-anchor="middle" font-size="12" fill="#555">猫 0.80</text>
  <text x="620" y="132" text-anchor="middle" font-size="12" fill="#555">吃 0.15</text>
  <text x="620" y="154" text-anchor="middle" font-size="12" fill="#555">鱼 0.05</text>
  <line x1="616" y1="166" x2="616" y2="180" stroke="#5a9e3a"/>
  <text x="620" y="200" text-anchor="middle" font-size="12" fill="#2f8f3e">输出 = 0.8·V₁ + 0.15·V₃ + 0.05·V₂</text>
  <line x1="450" y1="82" x2="520" y2="100" stroke="#555" marker-end="url(#ar)"/>
  <rect x="40" y="270" width="680" height="120" fill="#f7f7f7" stroke="#ccc" rx="8"/>
  <text x="60" y="298" font-size="13" fill="#333">· Query：当前词「拿着问题去检索」——我现在需要什么信息</text>
  <text x="60" y="324" font-size="13" fill="#333">· Key  ：每本书的「书脊标签」——用来和 Query 比对，算匹配分数</text>
  <text x="60" y="350" font-size="13" fill="#333">· Value：每本书的「正文内容」——按匹配分数加权，真正被取走的信息</text>
  <text x="60" y="376" font-size="13" fill="#2f8f3e">分数(Q·K) 只决定「拿多少」，Value 才是「拿什么」。这就是注意力。</text>
</svg>

## 第二步：为什么需要这套机制（动机，约 2 分钟）

🗣️ 「为什么不直接用原始词向量，非要搞 Q/K/V？因为同一个词在不同句子里的含义依赖上下文。比如『苹果』在『我吃苹果』和『苹果发布手机』里完全不同。Self-Attention 做的事，就是让每个词主动去『看』句子里其他所有词，按相关性把它们的信息融合进来，得到一个**带上下文的新表示**。Q/K/V 就是实现这种『看』的工具：用 Q 发问，用 K 应答，用 V 提供素材。」

## 第三步：逐步推导计算过程（核心，约 6 分钟）

📐 设输入序列有 n 个 token，每个 token 已经是一个 `d_model` 维向量，整体拼成矩阵 `X`，形状 `(n, d_model)`。

**① 生成 Q、K、V（线性投影）**

```
Q = X · Wq      # (n, d_model)·(d_model, dk) = (n, dk)
K = X · Wk      # (n, dk)
V = X · Wv      # (n, dv)，通常 dv = dk
```

🗣️ 「`Wq、Wk、Wv` 是三个可学习的权重矩阵，是训练出来的。它们把同一个输入 X，投影到三个不同的空间：一个专门用来『发问』，一个专门用来『被检索』，一个专门用来『提供内容』。注意：三者的输入都是同一个 X（这正是『自』注意力的含义——自己看自己）。」

**② 算原始注意力分数**

```
scores = Q · Kᵀ     # (n, dk)·(dk, n) = (n, n)
```

🗣️ 「`scores[i][j]` 表示第 i 个词对第 j 个词的关注强度。为什么用点积衡量相关性？因为两个向量点积越大，说明它们方向越接近、越『像』，也就越相关。这一步算出来是个 n×n 的方阵——每个词对每个词都有一个分数。」

**③ 缩放**

```
scores = scores / √dk
```

🗣️ 「这一步先记住『要除以 √dk』，原因留到 Q2 专门推导。简单说是为了防止分数太大导致 softmax 出问题。」

**④（可选）Mask**

🗣️ 「如果是 Decoder，要把『未来位置』和『PAD 位置』的分数改成负无穷，这样后面 softmax 之后这些位置权重≈0。Encoder 一般只 mask PAD。」

**⑤ softmax 归一化**

```
weights = softmax(scores, dim=每一行)   # (n, n)，每行加起来=1
```

🗣️ 「对**每一行**做 softmax。第 i 行表示『第 i 个词，把它的注意力如何分配给所有词』，所以这一行加起来必须等于 1，是一个概率分布。」

**⑥ 加权求和**

```
output = weights · V    # (n, n)·(n, dv) = (n, dv)
```

🗣️ 「最后用权重去加权 Value。第 i 个词的新表示 = 它对各个词的权重 × 各个词的 Value，再求和。这样每个词就融合了全句信息。」

📐 完整公式：

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √dk ) · V
```

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470" viewBox="0 0 760 470" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="arq1" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#555"/>
    </marker>
  </defs>
  <rect x="1" y="1" width="758" height="468" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="30" text-anchor="middle" font-size="16" fill="#333">Self-Attention 六步流程（带形状）</text>
  <rect x="40" y="55" width="200" height="56" fill="#fff3cd" stroke="#caa000" rx="6"/>
  <text x="140" y="80" text-anchor="middle" font-size="13">①输入 X</text>
  <text x="140" y="99" text-anchor="middle" font-size="11" fill="#777">(n, d_model)</text>
  <rect x="290" y="40" width="190" height="40" fill="#d6e4ff" stroke="#3366cc" rx="6"/>
  <text x="385" y="65" text-anchor="middle" font-size="12">②Q=XWq (n,dk)</text>
  <rect x="290" y="90" width="190" height="40" fill="#d6e4ff" stroke="#3366cc" rx="6"/>
  <text x="385" y="115" text-anchor="middle" font-size="12">K=XWk (n,dk)</text>
  <rect x="290" y="140" width="190" height="40" fill="#d6e4ff" stroke="#3366cc" rx="6"/>
  <text x="385" y="165" text-anchor="middle" font-size="12">V=XWv (n,dv)</text>
  <line x1="240" y1="83" x2="290" y2="60" stroke="#555" marker-end="url(#arq1)"/>
  <line x1="240" y1="85" x2="290" y2="110" stroke="#555" marker-end="url(#arq1)"/>
  <line x1="240" y1="90" x2="290" y2="160" stroke="#555" marker-end="url(#arq1)"/>
  <rect x="540" y="65" width="180" height="50" fill="#ffe0e0" stroke="#cc4444" rx="6"/>
  <text x="630" y="86" text-anchor="middle" font-size="12">③ QKᵀ</text>
  <text x="630" y="104" text-anchor="middle" font-size="11" fill="#777">scores (n, n)</text>
  <line x1="480" y1="60" x2="540" y2="82" stroke="#555" marker-end="url(#arq1)"/>
  <line x1="480" y1="110" x2="540" y2="95" stroke="#555" marker-end="url(#arq1)"/>
  <rect x="540" y="140" width="180" height="46" fill="#ffe0e0" stroke="#cc4444" rx="6"/>
  <text x="630" y="168" text-anchor="middle" font-size="12">④ ÷√dk（缩放）</text>
  <line x1="630" y1="115" x2="630" y2="140" stroke="#555" marker-end="url(#arq1)"/>
  <rect x="540" y="210" width="180" height="46" fill="#f0e6ff" stroke="#7a4fcf" rx="6"/>
  <text x="630" y="232" text-anchor="middle" font-size="12">⑤ (可选)Mask</text>
  <text x="630" y="249" text-anchor="middle" font-size="10" fill="#888">未来/PAD 置 -∞</text>
  <line x1="630" y1="186" x2="630" y2="210" stroke="#555" marker-end="url(#arq1)"/>
  <rect x="540" y="280" width="180" height="46" fill="#e2f0d9" stroke="#5a9e3a" rx="6"/>
  <text x="630" y="302" text-anchor="middle" font-size="12">⑥ softmax(每行)</text>
  <text x="630" y="319" text-anchor="middle" font-size="10" fill="#888">权重 (n,n) 行和=1</text>
  <line x1="630" y1="256" x2="630" y2="280" stroke="#555" marker-end="url(#arq1)"/>
  <rect x="290" y="280" width="190" height="46" fill="#e2f0d9" stroke="#5a9e3a" rx="6"/>
  <text x="385" y="302" text-anchor="middle" font-size="12">⑦ 权重·V</text>
  <text x="385" y="319" text-anchor="middle" font-size="11" fill="#777">输出 (n, dv)</text>
  <line x1="540" y1="303" x2="480" y2="303" stroke="#555" marker-end="url(#arq1)"/>
  <path d="M385 180 C 385 230, 385 250, 385 280" fill="none" stroke="#5a9e3a" stroke-dasharray="4,3" marker-end="url(#arq1)"/>
  <text x="395" y="245" font-size="10" fill="#5a9e3a">V 参与加权</text>
  <rect x="40" y="360" width="680" height="44" fill="#f7f7f7" stroke="#ccc" rx="6"/>
  <text x="380" y="388" text-anchor="middle" font-size="15" fill="#333">Attention(Q,K,V) = softmax( QKᵀ / √dk ) · V</text>
  <text x="380" y="432" text-anchor="middle" font-size="12" fill="#777">形状变化主线：(n,d) → 投影 → (n,dk) → 打分 (n,n) → 加权 → (n,dv)</text>
  <text x="380" y="454" text-anchor="middle" font-size="12" fill="#2f8f3e">口诀：投影 → 打分 → 缩放 → 掩码 → 归一 → 加权</text>
</svg>

形状主线是：`(n, d) → (n, dk) → (n, n) → (n, dv)`。

## 第四步：一个超小数值例子（落地，约 2 分钟）

📐 假设只有 2 个词，dk=2，简化到能心算：

```
Q = [[1, 0],      K = [[1, 0],      V = [[10, 0],
     [0, 1]]           [0, 1]]           [0, 20]]

QKᵀ = [[1, 0],          # 词1 和 词1 点积=1，和 词2 点积=0
       [0, 1]]

÷√2 ≈ [[0.71, 0],
        [0, 0.71]]

softmax 每行 ≈ [[0.67, 0.33],
                [0.33, 0.67]]

output 行1 = 0.67·[10,0] + 0.33·[0,20] = [6.7, 6.6]
output 行2 = 0.33·[10,0] + 0.67·[0,20] = [3.3, 13.4]
```

🗣️ 「看行1：它主要关注自己（0.67），所以输出更偏向 V₁=[10,0]，但也吸收了一点 V₂。这就是『融合上下文』的具体数字体现。」

## 第五步：高频追问与陷阱（约 2 分钟）

- **追问「自注意力的复杂度？」**：scores 是 n×n，所以是 O(n²·d)，序列越长开销平方增长（详见 Q12）。
- **陷阱「softmax 对行还是对列？」**：对**行**。说反会被认为没真正算过。
- **陷阱「输出形状」**：是 `(n, dv)`，和输入 token 数 n 一致——输入多少词，输出多少个表示。

🗣️ 收尾：「所以整个自注意力就是六个字：**投影、打分、加权**。投影出 Q/K/V，用 Q·K 打分并缩放归一，再用分数加权 V。」

---

# Q2. 为什么 softmax 前要除以 √dk？不缩放会怎样？

## 第一步：先讲清楚要解决的麻烦（直觉，约 3 分钟）

🗣️ 「softmax 有个脾气：如果输入里有个值特别大，它会把几乎全部概率都分给那一个，其它全压到接近 0，输出变成接近 one-hot 的『赢家通吃』。

问题来了：一旦 softmax 输出接近 one-hot，它就进入了**饱和区**。在饱和区，输入再怎么变，输出几乎不动——也就是**梯度接近 0**。梯度接近 0 意味着反向传播时这一层学不到东西，模型训不动。

那什么时候 softmax 的输入会出现特别大的值？答案是：**当 dk（向量维度）很大时，Q·K 点积天然会变得很大、很分散**。所以我们要把它『压回来』。」

## 第二步：严格推导点积的方差（核心，约 6 分钟）

📐 假设 q 和 k 的每一维都是独立随机变量，均值 0、方差 1（这是常见的初始化假设）。

点积是：

```
q · k = q₁k₁ + q₂k₂ + ... + q_dk·k_dk     （共 dk 项）
```

**先看单独一项 `qᵢ·kᵢ`：**

```
E[qᵢkᵢ] = E[qᵢ]·E[kᵢ] = 0·0 = 0           （独立，期望可拆）
Var(qᵢkᵢ) = E[(qᵢkᵢ)²] - (E[qᵢkᵢ])²
          = E[qᵢ²]·E[kᵢ²] - 0
          = Var(qᵢ)·Var(kᵢ)  = 1·1 = 1     （均值0时 E[x²]=Var）
```

**再看 dk 项相加**（独立变量相加，方差直接相加）：

```
E[q·k]   = 0 + 0 + ... = 0
Var(q·k) = 1 + 1 + ... + 1 = dk
```

📐 **结论：点积的均值是 0，方差是 dk，标准差是 √dk。**

🗣️ 「这就是核心。dk=64 时标准差是 8，dk=512 时标准差约 22.6。维度越大，点积的波动范围越大，越容易冒出绝对值很大的分数，把 softmax 推进饱和区。

那怎么把方差拉回 1？很简单，除以标准差 √dk 就行：」

```
Var( (q·k) / √dk ) = Var(q·k) / (√dk)² = dk / dk = 1
```

🗣️ 「除以 √dk 之后，分数的方差稳定回 1，不管 dk 多大都一样平稳。softmax 就回到了那个『梯度健康、能正常学习』的区间。」

对照下图：左边 dk 小、分布窄、softmax 平滑；右边 dk 大、分布宽、出现极大值、softmax 塌成 one-hot、梯度消失。

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect x="1" y="1" width="758" height="418" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="30" text-anchor="middle" font-size="16" fill="#333">为什么除以 √dk：点积方差随维度增长</text>
  <g transform="translate(40,60)">
    <text x="150" y="0" text-anchor="middle" font-size="14" fill="#3366cc">dk 小（=4）：点积分散程度小</text>
    <path d="M30 170 Q150 30 270 170" fill="none" stroke="#3366cc" stroke-width="2.5"/>
    <line x1="30" y1="170" x2="270" y2="170" stroke="#999"/>
    <text x="150" y="190" text-anchor="middle" font-size="11" fill="#777">点积 q·k 取值</text>
    <text x="150" y="212" text-anchor="middle" font-size="12" fill="#3366cc">方差 ≈ dk = 4，分布较集中</text>
    <rect x="40" y="240" width="220" height="60" fill="#e2f0d9" stroke="#5a9e3a" rx="6"/>
    <text x="150" y="266" text-anchor="middle" font-size="12">softmax 输出较平滑</text>
    <text x="150" y="287" text-anchor="middle" font-size="11" fill="#5a9e3a">梯度健康，能学习</text>
  </g>
  <g transform="translate(420,60)">
    <text x="150" y="0" text-anchor="middle" font-size="14" fill="#cc4444">dk 大（=512）：点积非常分散</text>
    <path d="M10 170 Q150 150 145 60 Q150 150 290 170" fill="none" stroke="#cc4444" stroke-width="2.5"/>
    <line x1="10" y1="170" x2="290" y2="170" stroke="#999"/>
    <text x="150" y="190" text-anchor="middle" font-size="11" fill="#777">点积 q·k 取值</text>
    <text x="150" y="212" text-anchor="middle" font-size="12" fill="#cc4444">方差 ≈ dk = 512，出现极大值</text>
    <rect x="40" y="240" width="220" height="60" fill="#ffe0e0" stroke="#cc4444" rx="6"/>
    <text x="150" y="262" text-anchor="middle" font-size="12">softmax 趋近 one-hot</text>
    <text x="150" y="283" text-anchor="middle" font-size="11" fill="#cc4444">梯度≈0 → 训不动</text>
  </g>
  <rect x="40" y="350" width="680" height="56" fill="#f7f7f7" stroke="#ccc" rx="6"/>
  <text x="60" y="375" font-size="13" fill="#333">推导：q,k 各维独立、均值0方差1 → 每项 qᵢkᵢ 方差1 → dk 项相加 → Var=dk，标准差=√dk</text>
  <text x="60" y="396" font-size="13" fill="#2f8f3e">除以 √dk 把方差拉回 ≈1，softmax 回到平滑区，梯度不消失。</text>
</svg>

## 第三步：补一个直觉性数值例子（落地，约 3 分钟）

📐 比较两组 softmax 输入：

```
不缩放(分数大)： softmax([10, 2, 1]) ≈ [0.9997, 0.0003, 0.0001]   ← 几乎 one-hot
缩放后(分数小)： softmax([2.0, 0.4, 0.2]) ≈ [0.68, 0.14, 0.18]    ← 平滑
```

🗣️ 「上面那组里，第一个值吃掉了几乎所有概率，其它两个梯度基本为 0；下面那组分配合理，每一项都有梯度可学。这就是缩放前后的真实差别。」

## 第四步：高频追问与陷阱（约 3 分钟）

- **追问「为什么是 √dk 而不是 dk？」**：因为我们要让**标准差**回到 1，标准差是 √方差 = √dk，所以除 √dk。除以 dk 会过度缩小，方差变成 1/dk，又太平了。
- **追问「不缩放真的训不出来吗？」**：dk 小时影响不大；dk 大时（Transformer 默认每头 64）不缩放会明显变差甚至发散，论文做过对比实验。
- **陷阱**：很多人只会背「防止梯度消失」，但答不出方差推导。能写出 `Var(q·k)=dk` 这一步，是算法岗的硬分水岭。

🗣️ 收尾：「一句话：**点积的方差等于 dk，会随维度膨胀；除以 √dk 把方差拉回 1，让 softmax 不饱和、梯度不消失。**」

---

# Q3. 为什么 Q 和 K 要用不同的权重矩阵？能否共享、或拿 K 和自己点乘？

## 第一步：先把问题翻译成人话（直觉，约 3 分钟）

🗣️ 「面试官其实在问：既然 Q 和 K 都来自同一个输入 X，那我用一套权重不就行了？为什么非要 `Wq` 和 `Wk` 两套？

打个比方：相亲。Q 是『我希望对方是什么样』（我的择偶需求），K 是『我自己对外展示的标签』（我的人设）。这两件事显然不是一回事——你想找的，和你能提供的，是两个不同的角色。如果强行用同一套描述，就等于说『我想找的人 = 我自己』，这显然限制太死了。」

## 第二步：核心论证——共享会让注意力矩阵变对称（推导，约 5 分钟）

📐 假设共享权重，即 `Q = K = X·W`，那么注意力分数矩阵：

```
scores = Q·Kᵀ = (XW)·(XW)ᵀ
```

📐 这个矩阵有一个致命性质——**它是对称矩阵**：`scores[i][j] == scores[j][i]`。

🗣️ 「为什么对称是坏事？因为语言里的关注关系**本来就是不对称的**。

举个例子：『那只动物没过马路，因为**它**太累了』。这里『它』要强烈关注『动物』（指代消解），但反过来，『动物』这个词并不需要同等强度地关注『它』。也就是说 `关注(它→动物) ≠ 关注(动物→它)`。

如果矩阵被强制对称，模型就无法表达这种单向的、方向敏感的依赖。表达能力直接被砍掉一半。」

## 第三步：第二层原因——同空间投影丢失表达力（约 4 分钟）

🗣️ 「除了对称，还有一个更深的原因：**子空间**。

`Wq` 把输入投到『查询子空间』，`Wk` 投到『键子空间』，这是两个不同的方向。在两个不同空间里做匹配，模型能学到更丰富、更灵活的相关性模式。

如果共享权重，Q 和 K 被投到**同一个空间**，相当于让一个向量和它自己（或同空间的兄弟）做点积。同空间自点积有个倾向：一个向量和自己最像，于是对角线（自己关注自己）的分数会异常大，注意力容易『塌缩到自己身上』，泛化很差。」

📐 小结对比：

```
共享 Wq=Wk：scores 对称、投影同空间 → 表达力受限、易自关注塌缩
独立 Wq/Wk：scores 非对称、投影双空间 → 方向敏感、表达力强
```

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect x="1" y="1" width="758" height="358" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="30" text-anchor="middle" font-size="16" fill="#333">共享 vs 独立 权重：注意力矩阵的对称性</text>
  <!-- shared -->
  <g transform="translate(70,60)">
    <text x="130" y="0" text-anchor="middle" font-size="14" fill="#cc4444">共享 Wq=Wk → 对称矩阵（坏）</text>
    <g font-size="13" text-anchor="middle">
      <rect x="40" y="20" width="56" height="40" fill="#ffe0e0" stroke="#cc4444"/><text x="68" y="46">.9</text>
      <rect x="96" y="20" width="56" height="40" fill="#fff" stroke="#cc4444"/><text x="124" y="46">.5</text>
      <rect x="152" y="20" width="56" height="40" fill="#fff" stroke="#cc4444"/><text x="180" y="46">.2</text>
      <rect x="40" y="60" width="56" height="40" fill="#fff" stroke="#cc4444"/><text x="68" y="86">.5</text>
      <rect x="96" y="60" width="56" height="40" fill="#ffe0e0" stroke="#cc4444"/><text x="124" y="86">.9</text>
      <rect x="152" y="60" width="56" height="40" fill="#fff" stroke="#cc4444"/><text x="180" y="86">.4</text>
      <rect x="40" y="100" width="56" height="40" fill="#fff" stroke="#cc4444"/><text x="68" y="126">.2</text>
      <rect x="96" y="100" width="56" height="40" fill="#fff" stroke="#cc4444"/><text x="124" y="126">.4</text>
      <rect x="152" y="100" width="56" height="40" fill="#ffe0e0" stroke="#cc4444"/><text x="180" y="126">.9</text>
    </g>
    <text x="124" y="165" text-anchor="middle" font-size="11" fill="#cc4444">[i][j]=[j][i]，无法表达单向关注</text>
  </g>
  <!-- independent -->
  <g transform="translate(420,60)">
    <text x="130" y="0" text-anchor="middle" font-size="14" fill="#5a9e3a">独立 Wq/Wk → 非对称矩阵（好）</text>
    <g font-size="13" text-anchor="middle">
      <rect x="40" y="20" width="56" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="68" y="46">.3</text>
      <rect x="96" y="20" width="56" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="124" y="46">.8</text>
      <rect x="152" y="20" width="56" height="40" fill="#fff" stroke="#5a9e3a"/><text x="180" y="46">.1</text>
      <rect x="40" y="60" width="56" height="40" fill="#fff" stroke="#5a9e3a"/><text x="68" y="86">.2</text>
      <rect x="96" y="60" width="56" height="40" fill="#fff" stroke="#5a9e3a"/><text x="124" y="86">.3</text>
      <rect x="152" y="60" width="56" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="180" y="86">.7</text>
      <rect x="40" y="100" width="56" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="68" y="126">.6</text>
      <rect x="96" y="100" width="56" height="40" fill="#fff" stroke="#5a9e3a"/><text x="124" y="126">.1</text>
      <rect x="152" y="100" width="56" height="40" fill="#fff" stroke="#5a9e3a"/><text x="180" y="126">.2</text>
    </g>
    <text x="124" y="165" text-anchor="middle" font-size="11" fill="#5a9e3a">[i][j]≠[j][i]，"它→动物"可单向强关注</text>
  </g>
  <rect x="40" y="270" width="680" height="70" fill="#f7f7f7" stroke="#ccc" rx="6"/>
  <text x="60" y="298" font-size="13" fill="#333">· 对称性：共享→Q·Kᵀ=(XW)(XW)ᵀ 必对称；语言关注本身不对称（指代、因果）</text>
  <text x="60" y="324" font-size="13" fill="#2f8f3e">· 子空间：独立权重把"想找什么/能被找到什么"投到两个空间，表达力更强</text>
</svg>

## 第四步：高频追问与陷阱（约 3 分钟）

- **追问「那 V 为什么也要单独一套 Wv？」**：因为「用什么匹配」和「匹配后取走什么内容」是两件事。K 负责被检索，V 负责提供信息，职责不同，所以三套权重各司其职。
- **陷阱「会不会说成 Q/K 不同是为了维度对齐？」**：在 self-attention 里 Q/K 维度其实相同，核心理由是**对称性 + 子空间表达力**，不是维度。这是常见误答。

🗣️ 收尾：「核心两句话：**共享权重会让注意力矩阵对称，丢掉语言里单向的依赖关系；并且把 Q、K 投到同一空间，表达力和泛化都变差。独立的 Wq/Wk 让『我要找什么』和『我能被找到什么』解耦。**」

---

# Q4. 多头注意力的设计目的是什么？为什么每个头要降维？

## 第一步：用比喻讲清「为什么要多个头」（直觉，约 3 分钟）

🗣️ 「想象你在分析一句话：『那只动物没过马路，因为它太累了』。这句话里同时藏着好几种关系：

- 指代关系：『它』指『动物』；
- 因果关系：『因为』连接前后两个分句；
- 动宾关系：『过马路』是个动宾短语。

如果只有一个注意力头，它就像只有一双眼睛、一个视角，很难同时盯住这么多种关系。多头注意力就是**给模型配多双眼睛**，每双眼睛（每个头）专门盯一种关系：一个头管指代，一个头管因果，一个头管句法……最后把各双眼睛看到的拼起来综合判断。」

## 第二步：计算流程逐步拆解（推导，约 5 分钟）

📐 设 `d_model=512`，头数 `h=8`，则每个头的维度 `d_k = d_model/h = 64`。

```
① 拆分/投影：对每个头 i，用各自的 Wqᵢ/Wkᵢ/Wvᵢ
   把 X 投影成 (n, 64) 的小 Q/K/V
② 各头独立算注意力：headᵢ = softmax(Qᵢ·Kᵢᵀ/√64)·Vᵢ   形状 (n, 64)
③ 拼接：Concat(head₁,...,head₈)                      形状 (n, 512)
④ 融合：乘以 Wo                                       形状 (n, 512)
```

📐 公式：

```
MultiHead(Q,K,V) = Concat(head₁,…,head_h)·Wo
head_i = Attention(X·Wqᵢ, X·Wkᵢ, X·Wvᵢ)
```

## 第三步：为什么每个头要降维（核心，约 4 分钟）

🗣️ 「关键问题：8 个头，是不是计算量就变成 8 倍？答案是**几乎不变**，原因就是降维。

每个头不是用满 512 维，而是只用 `512/8 = 64` 维。8 个头各算各的 64 维，拼起来恰好又是 `8×64 = 512`。所以：

- 总参数量：8 套 64 维投影 ≈ 1 套 512 维投影，基本相等；
- 总计算量：和单个 512 维全头注意力差不多。

也就是说，多头是用**几乎相同的成本**，把『一个大空间的注意力』拆成『8 个小空间各自的注意力』，换来了多视角能力。这是一笔非常划算的买卖。」

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="arq4" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#555"/>
    </marker>
  </defs>
  <rect x="1" y="1" width="758" height="358" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="30" text-anchor="middle" font-size="16" fill="#333">多头注意力：512 维拆成 8×64，并行后拼回</text>
  <rect x="30" y="150" width="90" height="56" fill="#fff3cd" stroke="#caa000" rx="6"/>
  <text x="75" y="175" text-anchor="middle" font-size="13">输入 X</text>
  <text x="75" y="194" text-anchor="middle" font-size="11" fill="#777">(n,512)</text>
  <g font-size="12" text-anchor="middle">
    <rect x="170" y="44" width="150" height="40" fill="#d6e4ff" stroke="#3366cc" rx="6"/><text x="245" y="69">head1 (n,64)·关注句法</text>
    <rect x="170" y="100" width="150" height="40" fill="#d6e4ff" stroke="#3366cc" rx="6"/><text x="245" y="125">head2 (n,64)·关注指代</text>
    <rect x="170" y="190" width="150" height="40" fill="#d6e4ff" stroke="#3366cc" rx="6"/><text x="245" y="215">... (n,64)</text>
    <rect x="170" y="246" width="150" height="40" fill="#d6e4ff" stroke="#3366cc" rx="6"/><text x="245" y="271">head8 (n,64)·关注因果</text>
  </g>
  <line x1="120" y1="175" x2="170" y2="64" stroke="#555" marker-end="url(#arq4)"/>
  <line x1="120" y1="178" x2="170" y2="120" stroke="#555" marker-end="url(#arq4)"/>
  <line x1="120" y1="183" x2="170" y2="210" stroke="#555" marker-end="url(#arq4)"/>
  <line x1="120" y1="186" x2="170" y2="266" stroke="#555" marker-end="url(#arq4)"/>
  <rect x="400" y="135" width="120" height="70" fill="#ffe0e0" stroke="#cc4444" rx="6"/>
  <text x="460" y="166" text-anchor="middle" font-size="13">Concat</text>
  <text x="460" y="186" text-anchor="middle" font-size="11" fill="#777">(n,512)</text>
  <line x1="320" y1="64" x2="400" y2="150" stroke="#555" marker-end="url(#arq4)"/>
  <line x1="320" y1="120" x2="400" y2="162" stroke="#555" marker-end="url(#arq4)"/>
  <line x1="320" y1="210" x2="400" y2="178" stroke="#555" marker-end="url(#arq4)"/>
  <line x1="320" y1="266" x2="400" y2="190" stroke="#555" marker-end="url(#arq4)"/>
  <rect x="580" y="135" width="120" height="70" fill="#e2f0d9" stroke="#5a9e3a" rx="6"/>
  <text x="640" y="166" text-anchor="middle" font-size="13">·Wo</text>
  <text x="640" y="186" text-anchor="middle" font-size="11" fill="#777">输出 (n,512)</text>
  <line x1="520" y1="170" x2="580" y2="170" stroke="#555" marker-end="url(#arq4)"/>
  <text x="380" y="330" text-anchor="middle" font-size="13" fill="#2f8f3e">8×64=512：总算量≈单头；多视角靠拆分子空间，融合靠 Wo</text>
</svg>

## 第四步：高频追问与陷阱（约 3 分钟）

- **追问「Wo 的作用是什么？容易被漏」**：拼接只是把 8 个头摆在一起，它们之间还没『交流』。`Wo` 做一次线性融合，让不同头的信息混合，得到最终输出。漏讲 Wo 是常见扣分点。
- **追问「头数越多越好吗？」**：不是。头太多每头维度太小（如 512/64=8 维），单头表达力不足；论文里 8 头是经验平衡点。
- **陷阱**：别把多头说成「为了增加计算量/参数量」——恰恰相反，降维让它几乎不增加成本。

🗣️ 收尾：「一句话：**多头 = 把一个大注意力空间拆成 h 个低维子空间并行计算，每头关注一种关系，拼接后用 Wo 融合；降维保证总成本≈单头，却换来多视角表达力。**」

---

# Q5. 为什么用点积注意力而不是加性注意力？

## 第一步：先摆出两个选手（直觉，约 2 分钟）

🗣️ 「计算 Q 和 K 相关性，历史上有两种主流打分方式：

- **加性注意力**（Bahdanau，就是我们 U11 学的）：`score = vᵀ·tanh(Wq·q + Wk·k)`，先把 q 和 k 投影相加，过 tanh，再用一个向量 v 压成分数。它带了额外参数和非线性。
- **点积注意力**（Transformer 用的）：`score = q·k / √dk`，直接点积，没有额外参数。

面试官想听你对比这两者的取舍。」

## 第二步：核心理由——点积能写成矩阵乘法，GPU 友好（推导，约 5 分钟）

🗣️ 「两者理论复杂度其实相近，关键差别在**工程实现效率**。

点积注意力的全部分数，可以一次性写成一个大矩阵乘法：

```
scores = Q · Kᵀ      # (n,dk)·(dk,n) = (n,n)，一次矩阵乘法搞定
```

矩阵乘法是 GPU 上**优化到极致**的操作（有专门的 Tensor Core、cuBLAS 内核），并行度极高、显存访问高效。

而加性注意力里有 `tanh` 和逐元素相加，没法压成一个干净的大矩阵乘法，要么逐对计算，要么写成不规则操作，GPU 利用率低、慢、还费显存。」

📐 复杂度直觉对比（n 为序列长，d 为维度）：

```
点积注意力：主要是 QKᵀ 的矩阵乘 → O(n²·d)，且全是矩阵乘法，GPU 极快
加性注意力：每对 (i,j) 都要过一次小 MLP → 同量级但常数大、并行差
```

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="330" viewBox="0 0 760 330" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="arq5" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#555"/>
    </marker>
  </defs>
  <rect x="1" y="1" width="758" height="328" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="28" text-anchor="middle" font-size="16" fill="#333">点积 vs 加性：能否压成一个大矩阵乘法</text>
  <!-- additive -->
  <g transform="translate(40,55)">
    <text x="150" y="0" text-anchor="middle" font-size="14" fill="#cc4444">加性注意力（Bahdanau）</text>
    <rect x="40" y="20" width="90" height="36" fill="#fff3cd" stroke="#caa000" rx="5"/><text x="85" y="43" text-anchor="middle" font-size="12">Wq·q</text>
    <rect x="170" y="20" width="90" height="36" fill="#fff3cd" stroke="#caa000" rx="5"/><text x="215" y="43" text-anchor="middle" font-size="12">Wk·k</text>
    <rect x="90" y="80" width="120" height="36" fill="#f0e6ff" stroke="#7a4fcf" rx="5"/><text x="150" y="103" text-anchor="middle" font-size="12">相加 + tanh</text>
    <rect x="90" y="140" width="120" height="36" fill="#ffe0e0" stroke="#cc4444" rx="5"/><text x="150" y="163" text-anchor="middle" font-size="12">vᵀ 压成分数</text>
    <line x1="85" y1="56" x2="130" y2="80" stroke="#555" marker-end="url(#arq5)"/>
    <line x1="215" y1="56" x2="170" y2="80" stroke="#555" marker-end="url(#arq5)"/>
    <line x1="150" y1="116" x2="150" y2="140" stroke="#555" marker-end="url(#arq5)"/>
    <text x="150" y="205" text-anchor="middle" font-size="11" fill="#cc4444">有 tanh 非线性，每对(i,j)单独算</text>
    <text x="150" y="225" text-anchor="middle" font-size="11" fill="#cc4444">→ 无法压成一个大矩阵乘 → GPU 慢</text>
  </g>
  <!-- dot product -->
  <g transform="translate(420,55)">
    <text x="150" y="0" text-anchor="middle" font-size="14" fill="#5a9e3a">点积注意力（Transformer）</text>
    <rect x="30" y="60" width="80" height="60" fill="#d6e4ff" stroke="#3366cc" rx="5"/><text x="70" y="95" text-anchor="middle" font-size="13">Q</text>
    <rect x="130" y="60" width="80" height="60" fill="#d6e4ff" stroke="#3366cc" rx="5"/><text x="170" y="95" text-anchor="middle" font-size="13">Kᵀ</text>
    <rect x="230" y="60" width="80" height="60" fill="#e2f0d9" stroke="#5a9e3a" rx="5"/><text x="270" y="88" text-anchor="middle" font-size="12">scores</text><text x="270" y="106" text-anchor="middle" font-size="10" fill="#777">(n,n)</text>
    <text x="120" y="97" font-size="16" fill="#333">·</text>
    <line x1="210" y1="90" x2="230" y2="90" stroke="#555" marker-end="url(#arq5)"/>
    <text x="150" y="160" text-anchor="middle" font-size="11" fill="#5a9e3a">一次矩阵乘法搞定全部分数</text>
    <text x="150" y="180" text-anchor="middle" font-size="11" fill="#5a9e3a">→ Tensor Core/cuBLAS 高度优化 → 极快</text>
  </g>
  <text x="380" y="315" text-anchor="middle" font-size="12" fill="#2f8f3e">理论复杂度相近，胜负在工程效率：点积=纯矩阵乘，对 GPU 友好</text>
</svg>

## 第三步：补充——大维度下为什么还要缩放（约 3 分钟）

🗣️ 「论文里还提到一个细节：当 dk 较小时，点积和加性效果差不多；但 dk 较大时，**不缩放的点积**会因为方差过大（见 Q2）而效果变差，甚至不如加性。所以 Transformer 的选择是『点积 + √dk 缩放』，既要点积的速度，又用缩放补上它在高维下的稳定性短板。」

📐 换个角度理解两者的『参数』差异：

```
加性注意力：打分函数里带 Wq、Wk、v 三组可学习参数 + tanh 非线性
           → 表达力强，但每个 (i,j) 对都要过这套小网络，算不动大 batch
点积注意力：打分函数本身 0 参数（Q/K 的投影权重在注意力之外）
           → 打分阶段就是纯 Q·Kᵀ，天然可批量化
```

🗣️ 「所以可以这么记：加性注意力把『复杂度』放在了打分函数里（带非线性和参数），换来表达力；点积注意力把打分函数做到极简（纯点积），把复杂度让给了『可被 GPU 加速的矩阵乘法』，换来速度。Transformer 选后者，因为它要堆很深、要吃大规模数据，速度是第一优先级。」

## 第四步：高频追问与陷阱（约 3 分钟）

- **追问「那加性注意力一无是处吗？」**：不是。它表达力（非线性）理论上更强，在维度大、不缩放时反而可能更稳。只是综合速度/效果，点积+缩放更优。
- **联系 U11**：可以主动说「我之前实现过 Bahdanau 加性注意力 `vᵀtanh(Wq·q+Wk·k)`，Transformer 换成点积主要就是为了能批量矩阵乘、吃满 GPU」。这种前后串联会让面试官眼前一亮。

🗣️ 收尾：「一句话：**点积能整体写成矩阵乘法，GPU 高度优化、又快又省显存；代价是高维下方差大，于是配一个 √dk 缩放补回稳定性。加性注意力表达力强但并行差，综合下来点积胜出。**」

---

# Q6. Decoder 里两种 Mask（Padding Mask、Look-Ahead Mask）的区别与作用？

## 第一步：先讲清楚两个完全不同的动机（直觉，约 3 分钟）

🗣️ 「Decoder 里有两种 mask，很多人会搞混。它们解决的是**两个完全不同的问题**：

- **Padding Mask（填充掩码）**：解决『batch 里句子不一样长，要用 <pad> 补齐』的问题。我们不希望模型把注意力浪费在这些无意义的填充符上。
- **Look-Ahead Mask（前瞻/因果掩码）**：解决『训练时并行，但不能偷看未来词』的问题。生成任务是逐词往后写的，预测第 3 个词时不能让模型看到第 4、5 个词的答案。

一个是『别看垃圾』，一个是『别看答案』。」

## 第二步：Padding Mask 怎么做（约 3 分钟）

🗣️ 「假设一个 batch 里：

```
句子A: [我, 爱, 你]              → 补成 [我, 爱, 你, <pad>, <pad>]
句子B: [今天, 天气, 真, 好, 啊]  → [今天, 天气, 真, 好, 啊]
```

句子A 后面两个 `<pad>` 是凑数的。Padding Mask 就是把这些 `<pad>` 位置对应的注意力分数置为 `-∞`，softmax 后它们的权重≈0，模型就不会关注它们。形状是 `(batch, seq_len)`，标记哪些位置是 pad。Encoder 和 Decoder 只要有 pad 都需要。」

## 第三步：Look-Ahead Mask 怎么做（核心，约 5 分钟）

🗣️ 「这是 Decoder 的灵魂。先讲清楚矛盾：

训练时为了效率，我们把整句目标序列**一次性**喂进去并行预测每个位置。但这样一来，预测第 3 个词时，模型在自注意力里会同时看到第 4、5 个词——而这些正是它要预测的答案。等于考试时把答案抄进去了，训练出来的模型一推理就废（推理时根本没有未来词）。

解决办法：在注意力分数矩阵上盖一个**下三角可见**的掩码。第 i 行（第 i 个词）只允许看第 1~i 个词，第 i+1 及以后全置 `-∞`。」

📐 一个 5×5 的因果掩码（✓可见，-∞屏蔽）：

```
        看k1  看k2  看k3  看k4  看k5
q1 →     ✓    -∞   -∞   -∞   -∞
q2 →     ✓    ✓    -∞   -∞   -∞
q3 →     ✓    ✓    ✓    -∞   -∞
q4 →     ✓    ✓    ✓    ✓    -∞
q5 →     ✓    ✓    ✓    ✓    ✓
```

🗣️ 「softmax 之后，那些 `-∞` 位置权重变成 0，加权求和时未来词的信息几乎不参与。这样即使在并行训练，每个位置也只能『看见它该看见的』，和逐词推理时的因果约束完全一致。」

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="330" viewBox="0 0 760 330" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect x="1" y="1" width="758" height="328" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="30" text-anchor="middle" font-size="16" fill="#333">Look-Ahead Mask：下三角可见，上三角置 -∞</text>
  <g transform="translate(250,55)" font-size="13" text-anchor="middle">
    <text x="28" y="-8" font-size="11" fill="#777">k1</text><text x="76" y="-8" font-size="11" fill="#777">k2</text><text x="124" y="-8" font-size="11" fill="#777">k3</text><text x="172" y="-8" font-size="11" fill="#777">k4</text><text x="220" y="-8" font-size="11" fill="#777">k5</text>
    <rect x="6" y="6" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="28" y="32">✓</text>
    <rect x="54" y="6" width="44" height="40" fill="#eeeeee" stroke="#bbb"/><text x="76" y="32" fill="#999">-∞</text>
    <rect x="102" y="6" width="44" height="40" fill="#eeeeee" stroke="#bbb"/><text x="124" y="32" fill="#999">-∞</text>
    <rect x="150" y="6" width="44" height="40" fill="#eeeeee" stroke="#bbb"/><text x="172" y="32" fill="#999">-∞</text>
    <rect x="198" y="6" width="44" height="40" fill="#eeeeee" stroke="#bbb"/><text x="220" y="32" fill="#999">-∞</text>
    <rect x="6" y="52" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="28" y="78">✓</text>
    <rect x="54" y="52" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="76" y="78">✓</text>
    <rect x="102" y="52" width="44" height="40" fill="#eeeeee" stroke="#bbb"/><text x="124" y="78" fill="#999">-∞</text>
    <rect x="150" y="52" width="44" height="40" fill="#eeeeee" stroke="#bbb"/><text x="172" y="78" fill="#999">-∞</text>
    <rect x="198" y="52" width="44" height="40" fill="#eeeeee" stroke="#bbb"/><text x="220" y="78" fill="#999">-∞</text>
    <rect x="6" y="98" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="28" y="124">✓</text>
    <rect x="54" y="98" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="76" y="124">✓</text>
    <rect x="102" y="98" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="124" y="124">✓</text>
    <rect x="150" y="98" width="44" height="40" fill="#eeeeee" stroke="#bbb"/><text x="172" y="124" fill="#999">-∞</text>
    <rect x="198" y="98" width="44" height="40" fill="#eeeeee" stroke="#bbb"/><text x="220" y="124" fill="#999">-∞</text>
    <rect x="6" y="144" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="28" y="170">✓</text>
    <rect x="54" y="144" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="76" y="170">✓</text>
    <rect x="102" y="144" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="124" y="170">✓</text>
    <rect x="150" y="144" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="172" y="170">✓</text>
    <rect x="198" y="144" width="44" height="40" fill="#eeeeee" stroke="#bbb"/><text x="220" y="170" fill="#999">-∞</text>
    <rect x="6" y="190" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="28" y="216">✓</text>
    <rect x="54" y="190" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="76" y="216">✓</text>
    <rect x="102" y="190" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="124" y="216">✓</text>
    <rect x="150" y="190" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="172" y="216">✓</text>
    <rect x="198" y="190" width="44" height="40" fill="#e2f0d9" stroke="#5a9e3a"/><text x="220" y="216">✓</text>
  </g>
  <g font-size="11" fill="#777" text-anchor="end">
    <text x="244" y="92">q1</text><text x="244" y="138">q2</text><text x="244" y="184">q3</text><text x="244" y="230">q4</text><text x="244" y="276">q5</text>
  </g>
  <text x="120" y="150" font-size="12" fill="#5a9e3a">✓ 可见(当前及历史)</text>
  <text x="120" y="175" font-size="12" fill="#999">-∞ 屏蔽未来</text>
  <text x="120" y="205" font-size="11" fill="#777">softmax 后</text>
  <text x="120" y="223" font-size="11" fill="#777">未来权重→0</text>
</svg>

## 第四步：高频追问与陷阱（约 2 分钟）

- **区别一句话**：Padding Mask 屏蔽『无意义的 pad 位置』，跟内容有关、形状 `(batch, seq_len)`；Look-Ahead Mask 屏蔽『未来位置』，跟顺序有关、形状 `(seq_len, seq_len)` 上三角。两者在 Decoder self-attention 里**叠加使用**。
- **追问「Encoder 要 Look-Ahead Mask 吗？」**：不要。Encoder 是理解整句、双向看，没有『不能看未来』的约束，只需要 Padding Mask。
- **追问「Cross-Attention 用哪个？」**：用 memory 的 Padding Mask（屏蔽源句 pad），不用 Look-Ahead（解码当前位置本就该看完整源句）。

🗣️ 收尾：「一句话：**Padding Mask 是『别看垃圾 pad』，靠内容判断；Look-Ahead Mask 是『别看未来答案』，靠下三角结构，让并行训练也满足逐词生成的因果约束。两者都用置 -∞ + softmax 趋 0 实现。**」

---

# Q7. Encoder 和 Decoder 之间是如何交互的？

## 第一步：用「翻译」场景讲清交互（直觉，约 3 分钟）

🗣️ 「想象一个翻译员把『我爱你』翻成英文。

- **Encoder** 先把整句中文读懂，输出每个中文词的『理解笔记』，这叫 `memory`。Encoder 是双向的，读『爱』时能同时看『我』和『你』，所以理解很充分。
- **Decoder** 一个词一个词地生成英文。生成每个英文词时，它需要回头看中文笔记：『我现在要写第二个词，中文里哪个词最相关？』这个『回头看中文笔记』的动作，就发生在 **Cross-Attention（编码器-解码器注意力）** 子层里。

这其实就是我们 U10/U11 学的 Seq2Seq Attention，只不过搬进了 Transformer 框架。」

## 第二步：核心——Cross-Attention 的 Q/K/V 来源（推导，约 5 分钟）

📐 这是整道题的得分点，必须背准：

```
Cross-Attention:
  Q ← 来自 Decoder（当前生成状态，经过 Masked Self-Attn 之后的表示）
  K ← 来自 Encoder 的 memory（整个源句的理解）
  V ← 来自 Encoder 的 memory
```

🗣️ 「直觉解释：

- **Q 来自解码端**，因为『发问的是当前要生成的词』——我现在想写什么，我去问。
- **K、V 来自编码端**，因为『被查询的内容是源句』——中文每个词能提供什么信息。

所以 Cross-Attention 做的事是：用『当前解码状态』这个 Query，去和『源句每个位置』的 Key 算相关性，得到权重，再对源句的 Value 加权求和，得到一份『针对当前生成步、从源句提取的上下文』。」

📐 对比一下 Decoder 里的两种注意力：

```
① Masked Self-Attn:  Q,K,V 全来自 Decoder 自己（看已生成的目标词，带因果mask）
② Cross-Attn:        Q 来自 Decoder，K/V 来自 Encoder memory（看源句）
```

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="340" viewBox="0 0 760 340" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="arq7" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#555"/>
    </marker>
  </defs>
  <rect x="1" y="1" width="758" height="338" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="30" text-anchor="middle" font-size="16" fill="#333">Cross-Attention：Q 来自解码端，K/V 来自编码端</text>
  <!-- encoder -->
  <rect x="40" y="70" width="220" height="200" fill="#f4f9ff" stroke="#3366cc" stroke-dasharray="5,4" rx="8"/>
  <text x="150" y="94" text-anchor="middle" font-size="14" fill="#3366cc">Encoder（理解源句）</text>
  <rect x="70" y="120" width="160" height="40" fill="#d6e4ff" stroke="#3366cc" rx="6"/>
  <text x="150" y="145" text-anchor="middle" font-size="12">双向 Self-Attn ×N</text>
  <rect x="70" y="185" width="160" height="50" fill="#e2f0d9" stroke="#5a9e3a" rx="6"/>
  <text x="150" y="208" text-anchor="middle" font-size="12">memory</text>
  <text x="150" y="225" text-anchor="middle" font-size="10" fill="#777">(b, src_len, d_model)</text>
  <!-- decoder -->
  <rect x="500" y="70" width="220" height="200" fill="#fff6f6" stroke="#cc4444" stroke-dasharray="5,4" rx="8"/>
  <text x="610" y="94" text-anchor="middle" font-size="14" fill="#cc4444">Decoder（逐词生成）</text>
  <rect x="530" y="120" width="160" height="40" fill="#ffe0e0" stroke="#cc4444" rx="6"/>
  <text x="610" y="145" text-anchor="middle" font-size="12">Masked Self-Attn</text>
  <rect x="530" y="185" width="160" height="50" fill="#d6e4ff" stroke="#3366cc" rx="6"/>
  <text x="610" y="208" text-anchor="middle" font-size="12">Cross-Attn</text>
  <text x="610" y="225" text-anchor="middle" font-size="10" fill="#777">Q←本层, K/V←memory</text>
  <line x1="610" y1="160" x2="610" y2="185" stroke="#555" marker-end="url(#arq7)"/>
  <!-- memory -> cross K/V -->
  <path d="M260 210 C 380 210, 380 210, 530 210" fill="none" stroke="#3366cc" stroke-width="2.5" marker-end="url(#arq7)"/>
  <text x="395" y="200" text-anchor="middle" font-size="12" fill="#3366cc">K, V（源句信息）</text>
  <text x="610" y="300" text-anchor="middle" font-size="12" fill="#2f8f3e">= U10/U11 的 Seq2Seq Attention 的 Transformer 版本</text>
</svg>

## 第三步：放进整层流程看（约 4 分钟）

🗣️ 「Decoder 每一层的完整顺序是：

1. **Masked Self-Attn**：当前已生成的目标词之间互相看（带因果 mask，不能看未来）；
2. **Cross-Attn**：拿上一步的输出当 Q，去 Encoder memory 里捞源句信息（K/V）；
3. **FFN**：逐位置非线性变换。

每个子层都配残差 + LayerNorm。也就是说，Decoder 先『理顺自己已经写了什么』，再『回头看原文』，最后『加工输出』。」

## 第四步：高频追问与陷阱（约 3 分钟）

- **背诵口诀**：「**Cross-Attn 的 Q 来自解码端，K/V 来自编码端**」。说反就错。
- **追问「memory 算几次？」**：Encoder 只跑一次，得到 memory；之后 Decoder 每一层、每一步都复用同一份 memory 当 K/V，不重算。
- **联系 Seq2Seq**：主动点明这就是 Bahdanau Attention 的本质——用解码状态查询编码输出。体现知识连贯。

🗣️ 收尾：「一句话：**Encoder 把源句编成 memory；Decoder 通过 Cross-Attention 交互——Q 来自当前解码状态，K/V 来自 Encoder memory，等价于用『当前要生成的词』去源句里捞最相关的信息。这就是 Seq2Seq Attention 的 Transformer 实现。**」

---

# Q8. 为什么用 LayerNorm 而不是 BatchNorm？它在 Transformer 的哪个位置？

## 第一步：先讲清两者「归一化方向」的本质区别（直觉，约 4 分钟）

🗣️ 「归一化就是把一堆数减均值、除标准差，拉成均值 0、方差 1 的分布，让训练更稳。关键问题是：**对哪一堆数算均值方差？**

- **BatchNorm**：竖着算。对**同一个特征维度**，跨整个 batch 的所有样本算均值方差。比如『第 3 维特征』，把这个 batch 里所有样本的第 3 维收集起来一起归一化。
- **LayerNorm**：横着算。对**同一个样本**，把它自己所有特征维度收集起来算均值方差。跟 batch 里别的样本无关。

一个跨样本（竖），一个跨特征（横）。」

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="320" viewBox="0 0 760 320" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect x="1" y="1" width="758" height="318" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="30" text-anchor="middle" font-size="16" fill="#333">LayerNorm（横/跨特征） vs BatchNorm（竖/跨样本）</text>
  <g transform="translate(70,60)">
    <text x="114" y="-10" text-anchor="middle" font-size="14" fill="#cc4444">BatchNorm：竖着归一（坏在NLP）</text>
    <rect x="48" y="0" width="44" height="180" fill="#ffe0e0"/>
    <g stroke="#bbb" fill="none">
      <rect x="4" y="0" width="44" height="45"/><rect x="48" y="0" width="44" height="45"/><rect x="92" y="0" width="44" height="45"/><rect x="136" y="0" width="44" height="45"/><rect x="180" y="0" width="44" height="45"/>
      <rect x="4" y="45" width="44" height="45"/><rect x="48" y="45" width="44" height="45"/><rect x="92" y="45" width="44" height="45"/><rect x="136" y="45" width="44" height="45"/><rect x="180" y="45" width="44" height="45"/>
      <rect x="4" y="90" width="44" height="45"/><rect x="48" y="90" width="44" height="45"/><rect x="92" y="90" width="44" height="45"/><rect x="136" y="90" width="44" height="45"/><rect x="180" y="90" width="44" height="45"/>
      <rect x="4" y="135" width="44" height="45"/><rect x="48" y="135" width="44" height="45"/><rect x="92" y="135" width="44" height="45"/><rect x="136" y="135" width="44" height="45"/><rect x="180" y="135" width="44" height="45"/>
    </g>
    <text x="-4" y="100" text-anchor="end" font-size="11" fill="#777" transform="rotate(-90 -4 100)">样本(batch)↓</text>
    <text x="114" y="200" text-anchor="middle" font-size="11" fill="#777">特征维度 →</text>
  </g>
  <g transform="translate(440,60)">
    <text x="114" y="-10" text-anchor="middle" font-size="14" fill="#5a9e3a">LayerNorm：横着归一（NLP用）</text>
    <rect x="4" y="45" width="220" height="45" fill="#e2f0d9"/>
    <g stroke="#bbb" fill="none">
      <rect x="4" y="0" width="44" height="45"/><rect x="48" y="0" width="44" height="45"/><rect x="92" y="0" width="44" height="45"/><rect x="136" y="0" width="44" height="45"/><rect x="180" y="0" width="44" height="45"/>
      <rect x="4" y="45" width="44" height="45"/><rect x="48" y="45" width="44" height="45"/><rect x="92" y="45" width="44" height="45"/><rect x="136" y="45" width="44" height="45"/><rect x="180" y="45" width="44" height="45"/>
      <rect x="4" y="90" width="44" height="45"/><rect x="48" y="90" width="44" height="45"/><rect x="92" y="90" width="44" height="45"/><rect x="136" y="90" width="44" height="45"/><rect x="180" y="90" width="44" height="45"/>
      <rect x="4" y="135" width="44" height="45"/><rect x="48" y="135" width="44" height="45"/><rect x="92" y="135" width="44" height="45"/><rect x="136" y="135" width="44" height="45"/><rect x="180" y="135" width="44" height="45"/>
    </g>
    <text x="-4" y="100" text-anchor="end" font-size="11" fill="#777" transform="rotate(-90 -4 100)">样本(batch)↓</text>
    <text x="114" y="200" text-anchor="middle" font-size="11" fill="#777">特征维度 →</text>
  </g>
  <text x="380" y="295" text-anchor="middle" font-size="12" fill="#2f8f3e">LN 只看单个 token 自己的特征，与 batch 大小、序列长度都无关</text>
</svg>

## 第二步：为什么 NLP 必须选 LayerNorm（核心，4 个理由，约 6 分钟）

🗣️ 「**理由①：变长序列，BN 在位置维上不合理。** NLP 句子长短不一，同一 batch 里要补 pad。BatchNorm 要跨样本统计，但句子第 1 个词和第 5 个词在语义上根本不是『同一类东西』，把它们硬凑到一个分布里算统计量，理论上就站不住脚。LayerNorm 只管单个 token 自己，没这个问题。

**理由②：BN 对 batch 大小敏感。** BatchNorm 要靠足够大的 batch 才能估出稳定的均值方差。NLP 序列长、显存吃紧，batch 往往很小，小 batch 下 BN 的统计量抖动很大，训练不稳。LayerNorm 完全不依赖 batch 大小。

**理由③：推理一致性。** BatchNorm 推理时用的是训练阶段累积的『全局均值/方差』。但自回归生成时，模型每步只处理一个 token，这个全局统计量未必准。LayerNorm 的统计量只来自当前样本，训练和推理行为完全一致。

**理由④：天然适配并行。** LayerNorm 对每个位置独立计算，不受 batch 和序列长度影响，正好契合 Transformer 的并行结构。」

## 第三步：它在 Transformer 的什么位置（约 3 分钟）

📐 原始 Transformer 是 **Post-Norm**（先子层、再加残差、最后归一化）：

```
output = LayerNorm( x + SubLayer(x) )
```

🗣️ 「位置是：每个子层（Self-Attn / Cross-Attn / FFN）之后，配合残差连接做归一化。

加分项：现代大模型大多改用 **Pre-Norm**：`output = x + SubLayer(LayerNorm(x))`，先归一化再进子层。Pre-Norm 让残差路径更干净、梯度直接回传，深层训练更稳、不太需要 warmup。再进一步，很多模型用 **RMSNorm**——只除以均方根、不减均值，更省算力、效果几乎不变。」

## 第四步：高频追问与陷阱（约 2 分钟）

- **追问「BN 在 CV 里为什么好用？」**：图像 batch 里同一通道的统计是有意义的（同类特征），且 batch 通常较大，所以 BN 在 CV 表现好。NLP 不满足这些前提。
- **陷阱**：只答「LN 对每个样本归一化」是不够的，要能说出『变长 / 小 batch / 推理一致 / 并行』至少 2~3 条，才显得真懂。

🗣️ 收尾：「一句话：**BatchNorm 跨样本竖着归一，依赖 batch 且对变长序列不合理、推理不一致；LayerNorm 跨特征横着归一，只看单个 token 自身，与 batch/长度无关，训练推理一致。位置在每个子层之后配残差（原始 Post-Norm，现代多用 Pre-Norm/RMSNorm）。**」

---

# Q9. 位置编码为什么必要？正弦余弦编码有什么优点？还知道哪些位置编码？

## 第一步：先证明「不加位置编码会出大事」（直觉，约 4 分钟）

🗣️ 「先记住一个关键性质：**Self-Attention 是排列等变的（permutation equivariant）**。

通俗说：如果你把输入词的顺序打乱，Self-Attention 的输出只是跟着相应地重排，每个词算出来的表示**完全不变**。因为注意力只看『哪些词之间有关系』（靠 Q·K 内容算），根本不看『谁在前谁在后』。

后果很严重：『猫吃鱼』和『鱼吃猫』，词完全一样，只是顺序不同。在没有位置信息的 Self-Attention 眼里，这两句话里『猫』算出的表示是一模一样的。但这俩意思完全相反！所以必须想办法把『位置』这个信息硬塞进去。」

## 第二步：位置编码怎么塞进去 + 为什么用正余弦（核心，约 6 分钟）

🗣️ 「做法很简单：给每个位置生成一个和词向量同维度的『位置向量』，**加到**词向量上，作为输入。这样模型拿到的每个词，既有词义信息，又有位置信息。

那位置向量怎么造？先看几个失败的朴素方案：

- **方案A：直接用 0,1,2,3...** 越靠后数值越大，加到词向量上会造成数值倾斜，淹没词义。
- **方案B：归一化到 [0,1]，即 pos/句长。** 问题是同一个位置在不同长度句子里编码不一致——位置 5 在长度 10 的句子是 0.5，在长度 1000 的句子是 0.005，模型学不到稳定的位置感。

理想要求：每个位置有唯一、稳定、与句长无关的编码。Transformer 的答案是**正弦/余弦函数**：」

📐 公式：

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )      # 偶数维用 sin
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )      # 奇数维用 cos
```

🗣️ 「直觉：每个维度是一个不同频率的正弦波。低维频率高、变化快（区分相邻位置），高维频率低、变化慢（区分远距离位置）。多个不同频率的波组合起来，就像二进制编码一样，能给每个位置一个独一无二的『指纹』。」

📐 正余弦编码的优点：

```
① 数值稳定：所有值都在 [-1, 1]，加到词向量上不会数值爆炸
② 无需训练：固定公式可预计算，不增加参数
③ 句长无关：同一位置在任何句子里编码都一致
④ 含相对关系：PE(pos+k) 可由 PE(pos) 线性表示，便于模型感知相对距离
⑤ 可外推：理论上能算出训练时没见过的更长位置的编码
```

## 第三步：还知道哪些位置编码（拓展，约 3 分钟）

🗣️ 「面试官追问『还知道哪些』，是想看你的知识边界。按演进顺序：

- **可学习位置编码**（BERT）：把位置向量当可训练参数，随机初始化后让模型自己学。灵活贴合数据，但**无法外推**到训练时没见过的长度。
- **相对位置编码**（T5、Transformer-XL、DeBERTa）：不编码绝对位置，而是编码 token 之间的『相对距离』。对翻译等位置敏感任务更友好，但实现复杂、开销略大。
- **RoPE 旋转位置编码**（LLaMA、ChatGLM、GPT 系）：当前主流。通过对 Q/K 向量做『旋转』把相对位置信息融进注意力点积里，外推能力比正余弦更好，是现在大模型的标配。延伸还有 YaRN、NTK-aware 等扩展 RoPE 有效长度的方法。」

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="300" viewBox="0 0 760 300" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect x="1" y="1" width="758" height="298" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="28" text-anchor="middle" font-size="16" fill="#333">不同维度 = 不同频率的正弦波（给每个位置唯一指纹）</text>
  <!-- axes -->
  <line x1="50" y1="160" x2="710" y2="160" stroke="#999"/>
  <text x="715" y="164" font-size="11" fill="#777">pos</text>
  <!-- high freq wave (low dim) -->
  <path d="M50 160 Q80 90 110 160 Q140 230 170 160 Q200 90 230 160 Q260 230 290 160 Q320 90 350 160 Q380 230 410 160 Q440 90 470 160 Q500 230 530 160 Q560 90 590 160 Q620 230 650 160 Q680 90 710 160" fill="none" stroke="#cc4444" stroke-width="2"/>
  <text x="120" y="80" font-size="11" fill="#cc4444">低维：高频、变化快（区分相邻位置）</text>
  <!-- low freq wave (high dim) -->
  <path d="M50 160 Q200 70 350 160 Q500 250 710 160" fill="none" stroke="#3366cc" stroke-width="2"/>
  <text x="430" y="255" font-size="11" fill="#3366cc">高维：低频、变化慢（区分远距离位置）</text>
  <text x="380" y="285" text-anchor="middle" font-size="12" fill="#2f8f3e">偶数维 sin、奇数维 cos，多频率组合 → 每个 pos 一个唯一编码</text>
</svg>

## 第四步：高频追问与陷阱（约 2 分钟）

- **第一句必须是**：「Self-Attention 排列等变，没有位置信息，分不清『猫吃鱼/鱼吃猫』」。这是采分句。
- **追问「为什么用 sin 和 cos 两个？」**：成对的 sin/cos 使得 `PE(pos+k)` 能写成 `PE(pos)` 的线性变换（旋转），这正是它能表达相对位置的数学基础。
- **加分**：主动提 RoPE，说明你跟进了 LLaMA 等现代模型。

🗣️ 收尾：「一句话：**Self-Attention 排列等变、天生无序，必须注入位置信息。正余弦编码用不同频率的 sin/cos 给每个位置唯一指纹，优点是稳定、无需训练、句长无关、含相对关系、可外推。现代模型多用 RoPE，外推更强。**」

---

# Q10. FFN 的结构是什么？为什么是「两层全连接 + ReLU」？

## 第一步：先点明 FFN 要补的窟窿（直觉，约 3 分钟）

🗣️ 「先想一个问题：注意力子层本质在干嘛？它是对 Value 做**加权求和**——而加权求和是个**线性操作**。你把一堆向量按权重加起来，无论怎么加，都跳不出线性变换的范畴。

光有线性变换，模型表达力很弱（多个线性层叠起来还是等价于一个线性层）。所以每个注意力子层后面，都要跟一个 **FFN（前馈网络）**，专门负责引入**非线性**，给每个位置的表示做一次『深加工』。」

## 第二步：结构逐步拆解（推导，约 5 分钟）

📐 FFN 的结构是两个线性层夹一个激活：

```
FFN(x) = Linear2( ReLU( Linear1(x) ) )
       = W2 · ReLU(W1·x + b1) + b2
```

📐 维度变化（关键）：

```
输入 x:        (..., d_model)        例如 512
Linear1 升维:  d_model → d_ff        512 → 2048   (d_ff 通常是 4×d_model)
ReLU:          非线性，形状不变       2048
Linear2 降维:  d_ff → d_model        2048 → 512
输出:          (..., d_model)        512
```

🗣️ 「所以是一个『先胖后瘦』的结构：先把 512 维升到 2048 维，在这个更宽的空间里做非线性变换（ReLU），再压回 512 维。

为什么要先升维？因为更高维的空间能容纳更复杂的特征组合，相当于给模型一个更大的『工作台』来加工信息，加工完再收回原尺寸，方便堆叠下一层。」

## 第三步：一个容易被忽略的点——逐位置（约 4 分钟）

🗣️ 「FFN 是 **position-wise（逐位置）** 的，这个词面试很爱考。

意思是：序列里每个 token 各自独立地过同一套 FFN 权重，token 之间**不发生任何信息交换**。第 1 个词的 FFN 和第 5 个词的 FFN 用的是完全相同的 W1、W2，但各算各的。

为什么这样设计？因为『跨位置混合信息』这件事已经交给注意力层做了。FFN 的分工就是：注意力负责『横向』把全句信息融合进每个位置，FFN 负责『纵向』把每个位置的表示加工得更有表达力。两者分工明确。」

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="280" viewBox="0 0 760 280" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="arq10" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#555"/>
    </marker>
  </defs>
  <rect x="1" y="1" width="758" height="278" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="30" text-anchor="middle" font-size="16" fill="#333">FFN：先升维 → ReLU 非线性 → 再降维（逐位置）</text>
  <rect x="40" y="105" width="110" height="70" fill="#fff3cd" stroke="#caa000" rx="6"/>
  <text x="95" y="135" text-anchor="middle" font-size="13">输入 x</text>
  <text x="95" y="155" text-anchor="middle" font-size="11" fill="#777">512</text>
  <rect x="210" y="95" width="130" height="90" fill="#d6e4ff" stroke="#3366cc" rx="6"/>
  <text x="275" y="130" text-anchor="middle" font-size="13">Linear1 升维</text>
  <text x="275" y="152" text-anchor="middle" font-size="11" fill="#777">512 → 2048</text>
  <rect x="400" y="105" width="110" height="70" fill="#f0e6ff" stroke="#7a4fcf" rx="6"/>
  <text x="455" y="135" text-anchor="middle" font-size="13">ReLU</text>
  <text x="455" y="155" text-anchor="middle" font-size="11" fill="#777">非线性 2048</text>
  <rect x="570" y="95" width="150" height="90" fill="#e2f0d9" stroke="#5a9e3a" rx="6"/>
  <text x="645" y="130" text-anchor="middle" font-size="13">Linear2 降维</text>
  <text x="645" y="152" text-anchor="middle" font-size="11" fill="#777">2048 → 512</text>
  <line x1="150" y1="140" x2="210" y2="140" stroke="#555" marker-end="url(#arq10)"/>
  <line x1="340" y1="140" x2="400" y2="140" stroke="#555" marker-end="url(#arq10)"/>
  <line x1="510" y1="140" x2="570" y2="140" stroke="#555" marker-end="url(#arq10)"/>
  <text x="380" y="235" text-anchor="middle" font-size="13" fill="#333">FFN(x) = W2 · ReLU(W1·x + b1) + b2</text>
  <text x="380" y="262" text-anchor="middle" font-size="12" fill="#2f8f3e">注意力管"跨位置融合"，FFN 管"逐位置非线性加工"，分工互补</text>
</svg>

## 第四步：高频追问与陷阱（约 3 分钟）

- **追问「为什么 d_ff 是 4 倍？」**：经验值。更宽的中间层提供更强的非线性容量，4× 是性能与开销的平衡点，原论文如此，后续沿用。
- **追问「ReLU 有什么缺点？现代用什么？」**：ReLU 输出非零中心，负区间梯度为 0 可能导致『神经元死亡』。现代模型常换成 GeLU、SwiGLU 等更平滑的激活。
- **陷阱**：别忘了强调 position-wise（逐位置、权重共享、不跨位置混合），这是高频采分点。

🗣️ 收尾：「一句话：**注意力是线性加权，缺非线性；FFN 用『升维 → ReLU → 降维』补上非线性，且逐位置独立加工（跨位置融合交给注意力）。d_ff 通常 4×d_model，现代常用 GeLU/SwiGLU 替代 ReLU。**」

---

# Q11. 为什么取词向量后要乘以 √d_model？残差连接的意义是什么？

> 这题是「魔鬼细节」合并考，能答出 embedding 乘 √d_model 的人极少，是顶尖候选人的区分点。

## 第一步：embedding 乘 √d_model——直觉与动机（约 4 分钟）

🗣️ 「先看背景。输入端做的事是：`输入 = 词向量(embedding) + 位置编码(PE)`，两者相加。

这里有个量级匹配问题。位置编码用 sin/cos，值域固定在 `[-1, 1]`，量级是 1 左右。而词向量是用类似 Xavier 的方式初始化的，它的方差大约是 `1/d_model`，也就是说每个分量都很小（d_model=512 时，标准差才约 0.044）。

如果直接相加，词向量太小、位置编码相对太大，会出现**位置编码淹没词义**的情况——模型一上来更关注『你在第几个位置』，反而忽略了『你是哪个词』。这显然不对。」

## 第二步：为什么是 √d_model（推导，约 4 分钟）

📐 推导：

```
设 embedding 每个分量方差 ≈ 1/d_model
乘以一个常数 c 后，方差变为 c² · (1/d_model)
要让方差 ≈ 1，需 c² / d_model = 1  →  c = √d_model
```

🗣️ 「所以乘 `√d_model`，恰好把词向量的方差从 `1/d_model` 拉回到约 1，量级变成 1 左右，正好和位置编码（量级 1）匹配。相加时两者势均力敌，词义和位置都能被模型感知。同时方差为 1 的 embedding 也更利于后续层的收敛。

注意是『**乘**以 √d_model』（放大词向量），不是除——很多人会记反。」

## 第三步：残差连接的意义（核心，约 5 分钟）

📐 残差结构：

```
y = x + SubLayer(x)
```

🗣️ 「就是把子层的输入 `x` 直接加到子层输出上，形成一条『跳过子层』的捷径。它有三层意义：

**① 缓解梯度消失（最重要）。** 反向传播时，梯度对 `y = x + SubLayer(x)` 求导，会得到一个『1 + 子层导数』。这个 **+1** 保证了哪怕子层的梯度很小，梯度也总有一条『恒等高速路』能直接流回前面的层，不会逐层衰减到 0。这是 Transformer 能堆到 6 层、几十层甚至上百层的根本前提。

**② 让子层只学增量。** 子层不用学完整的目标映射，只需学『相对输入要改动什么』(SubLayer(x) 是增量)，优化更容易。

**③ 信息保底。** 即使某个子层学砸了（输出接近噪声），残差也能让原始输入 x 至少原样传下去，不至于把信息全毁掉。」

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="260" viewBox="0 0 760 260" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="arq11" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#555"/>
    </marker>
  </defs>
  <rect x="1" y="1" width="758" height="258" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="30" text-anchor="middle" font-size="16" fill="#333">残差连接：给梯度一条 "+1" 的恒等高速路</text>
  <rect x="40" y="110" width="80" height="50" fill="#fff3cd" stroke="#caa000" rx="6"/>
  <text x="80" y="140" text-anchor="middle" font-size="13">输入 x</text>
  <rect x="230" y="110" width="160" height="50" fill="#d6e4ff" stroke="#3366cc" rx="6"/>
  <text x="310" y="140" text-anchor="middle" font-size="13">SubLayer(x)</text>
  <circle cx="500" cy="135" r="22" fill="#ffe0e0" stroke="#cc4444"/>
  <text x="500" y="142" text-anchor="middle" font-size="18" fill="#cc4444">+</text>
  <rect x="600" y="110" width="120" height="50" fill="#e2f0d9" stroke="#5a9e3a" rx="6"/>
  <text x="660" y="140" text-anchor="middle" font-size="13">y = x+SubL(x)</text>
  <line x1="120" y1="135" x2="230" y2="135" stroke="#555" marker-end="url(#arq11)"/>
  <line x1="390" y1="135" x2="478" y2="135" stroke="#555" marker-end="url(#arq11)"/>
  <line x1="522" y1="135" x2="600" y2="135" stroke="#555" marker-end="url(#arq11)"/>
  <!-- skip path -->
  <path d="M80 110 C 80 50, 500 50, 500 113" fill="none" stroke="#cc4444" stroke-width="2.5" stroke-dasharray="5,4" marker-end="url(#arq11)"/>
  <text x="290" y="62" text-anchor="middle" font-size="12" fill="#cc4444">捷径：x 直接跳过子层（梯度的 +1 通路）</text>
  <text x="380" y="225" text-anchor="middle" font-size="12" fill="#2f8f3e">∂y/∂x = 1 + ∂SubLayer/∂x，那个 1 保证梯度不消失 → 可堆深层</text>
</svg>

## 第四步：高频追问与陷阱（约 2 分钟）

- **embedding 缩放陷阱**：是『乘』√d_model 不是除；目的是让词向量方差≈1、与位置编码量级匹配。
- **残差追问「为什么能缓解梯度消失？」**：因为求导出现 `1 + ...`，恒等项保证梯度有一条不衰减的回传路径。说不出这个 `+1` 就是没真懂。
- **联系 LayerNorm**：残差和 LayerNorm 总是搭配出现：`LayerNorm(x + SubLayer(x))`，一个保梯度通路、一个稳分布。

🗣️ 收尾：「一句话：**乘 √d_model 是把词向量方差从 1/d_model 拉回约 1，让它和值域 [-1,1] 的位置编码量级匹配、不被淹没；残差 `y=x+SubLayer(x)` 求导出现 +1，给梯度一条恒等高速路，缓解梯度消失，是堆叠深层的关键。**」

---

# Q12. Transformer 的复杂度是多少？带来哪些工程局限？如何缓解？

## 第一步：先把复杂度的来源讲清楚（直觉 + 推导，约 4 分钟）

🗣️ 「Transformer 最被诟病的就是它的**平方复杂度**。来源很直接：

自注意力要算 `scores = Q·Kᵀ`，这是『每个位置 和 每个位置』两两算相关性。n 个位置两两组合，就是 n×n 个分数。」

📐 复杂度：

```
scores = Q·Kᵀ : (n,d)·(d,n) → (n,n)，计算量 O(n²·d)
时间和显存都是 O(n²·d)，其中 n=序列长度，d=维度
```

🗣️ 「关键在那个 **n²**：序列长度翻倍，注意力的计算量和显存就变成 **4 倍**。这就是一切长上下文问题的根源。」

## 第二步：由此引出的 4 个工程局限（核心，约 6 分钟）

🗣️ 「**局限①：上下文窗口有硬上限。** 因为是 n²，序列不能无限长。想支持更长上下文，显存和算力是平方级涨的，成本扛不住，所以每个模型都有一个最大 token 数。

**局限②：位置编码外推问题。** 模型训练时见过的位置范围有限。如果训练时最长见过 4096，那它对第 8000 个位置的位置编码是『没学过』的，硬上会导致长文本后半段质量下降。

**局限③：中间迷失（Lost in the Middle）。** 实验发现 Transformer 对输入**中间部分**的关注度，明显低于开头和结尾。开头的信息因为靠前，对后续所有 token 都有影响；结尾的信息离生成位置近，也容易被关注；中间的两头不靠，容易被忽略。

**局限④：生成是串行的。** Decoder 自回归，第 N 个 token 必须等前 N-1 个生成完才能生成，没法并行，生成速度有天花板。」

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="300" viewBox="0 0 760 300" font-family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect x="1" y="1" width="758" height="298" fill="#ffffff" stroke="#2f8f3e" stroke-width="2" rx="6"/>
  <text x="380" y="28" text-anchor="middle" font-size="16" fill="#333">O(n²) 注意力：序列翻倍，开销变 4 倍</text>
  <!-- n=4 grid -->
  <g transform="translate(70,55)">
    <text x="80" y="-8" text-anchor="middle" font-size="12" fill="#3366cc">n=4 → 16 个分数</text>
    <g stroke="#3366cc" fill="#d6e4ff">
      <rect x="0" y="0" width="35" height="35"/><rect x="35" y="0" width="35" height="35"/><rect x="70" y="0" width="35" height="35"/><rect x="105" y="0" width="35" height="35"/>
      <rect x="0" y="35" width="35" height="35"/><rect x="35" y="35" width="35" height="35"/><rect x="70" y="35" width="35" height="35"/><rect x="105" y="35" width="35" height="35"/>
      <rect x="0" y="70" width="35" height="35"/><rect x="35" y="70" width="35" height="35"/><rect x="70" y="70" width="35" height="35"/><rect x="105" y="70" width="35" height="35"/>
      <rect x="0" y="105" width="35" height="35"/><rect x="35" y="105" width="35" height="35"/><rect x="70" y="105" width="35" height="35"/><rect x="105" y="105" width="35" height="35"/>
    </g>
  </g>
  <text x="300" y="125" font-size="26" fill="#cc4444">→</text>
  <!-- n=8 grid -->
  <g transform="translate(360,55)">
    <text x="100" y="-8" text-anchor="middle" font-size="12" fill="#cc4444">n=8 → 64 个分数（4 倍）</text>
    <g stroke="#cc4444" fill="#ffe0e0">
      <!-- 8x8 -->
      <g>
        <!-- generate via rows -->
        <rect x="0" y="0" width="24" height="24"/><rect x="24" y="0" width="24" height="24"/><rect x="48" y="0" width="24" height="24"/><rect x="72" y="0" width="24" height="24"/><rect x="96" y="0" width="24" height="24"/><rect x="120" y="0" width="24" height="24"/><rect x="144" y="0" width="24" height="24"/><rect x="168" y="0" width="24" height="24"/>
        <rect x="0" y="24" width="24" height="24"/><rect x="24" y="24" width="24" height="24"/><rect x="48" y="24" width="24" height="24"/><rect x="72" y="24" width="24" height="24"/><rect x="96" y="24" width="24" height="24"/><rect x="120" y="24" width="24" height="24"/><rect x="144" y="24" width="24" height="24"/><rect x="168" y="24" width="24" height="24"/>
        <rect x="0" y="48" width="24" height="24"/><rect x="24" y="48" width="24" height="24"/><rect x="48" y="48" width="24" height="24"/><rect x="72" y="48" width="24" height="24"/><rect x="96" y="48" width="24" height="24"/><rect x="120" y="48" width="24" height="24"/><rect x="144" y="48" width="24" height="24"/><rect x="168" y="48" width="24" height="24"/>
        <rect x="0" y="72" width="24" height="24"/><rect x="24" y="72" width="24" height="24"/><rect x="48" y="72" width="24" height="24"/><rect x="72" y="72" width="24" height="24"/><rect x="96" y="72" width="24" height="24"/><rect x="120" y="72" width="24" height="24"/><rect x="144" y="72" width="24" height="24"/><rect x="168" y="72" width="24" height="24"/>
        <rect x="0" y="96" width="24" height="24"/><rect x="24" y="96" width="24" height="24"/><rect x="48" y="96" width="24" height="24"/><rect x="72" y="96" width="24" height="24"/><rect x="96" y="96" width="24" height="24"/><rect x="120" y="96" width="24" height="24"/><rect x="144" y="96" width="24" height="24"/><rect x="168" y="96" width="24" height="24"/>
        <rect x="0" y="120" width="24" height="24"/><rect x="24" y="120" width="24" height="24"/><rect x="48" y="120" width="24" height="24"/><rect x="72" y="120" width="24" height="24"/><rect x="96" y="120" width="24" height="24"/><rect x="120" y="120" width="24" height="24"/><rect x="144" y="120" width="24" height="24"/><rect x="168" y="120" width="24" height="24"/>
        <rect x="0" y="144" width="24" height="24"/><rect x="24" y="144" width="24" height="24"/><rect x="48" y="144" width="24" height="24"/><rect x="72" y="144" width="24" height="24"/><rect x="96" y="144" width="24" height="24"/><rect x="120" y="144" width="24" height="24"/><rect x="144" y="144" width="24" height="24"/><rect x="168" y="144" width="24" height="24"/>
        <rect x="0" y="168" width="24" height="24"/><rect x="24" y="168" width="24" height="24"/><rect x="48" y="168" width="24" height="24"/><rect x="72" y="168" width="24" height="24"/><rect x="96" y="168" width="24" height="24"/><rect x="120" y="168" width="24" height="24"/><rect x="144" y="168" width="24" height="24"/><rect x="168" y="168" width="24" height="24"/>
      </g>
    </g>
  </g>
  <text x="380" y="285" text-anchor="middle" font-size="12" fill="#2f8f3e">窗口上限 / 外推差 / 中间迷失 / 串行生成，都源于这个 n²</text>
</svg>

## 第三步：怎么缓解（分模型层 / 应用层，约 4 分钟）

🗣️ 「缓解手段分两个层面讲，显得有体系：

**模型/系统层：**
- **稀疏注意力 / 滑动窗口注意力**：不算全局 n²，只在局部窗口里算，牺牲一点全局信息换更长长度；
- **FlashAttention**：不改变数学结果，但通过分块计算大幅省显存、提速（工程优化）；
- **RoPE / YaRN / NTK-aware**：扩展位置编码的有效外推范围，缓解局限②；
- **投机解码（Speculative Decoding）**：小模型先快速猜几个 token，大模型并行验证，缓解局限④的串行慢。

**应用层（做应用开发要会答）：**
- 关键信息放 Prompt 的**开头或结尾**，避开『中间迷失』；
- 做**上下文管理**，控制喂进去的 token 数，别无脑塞满；
- 输出长内容用**流式返回**，改善体验。」

## 第四步：高频追问与陷阱（约 2 分钟）

- **锚点先抛**：先说『O(n²·d)』，再顺着 n² 推出 4 个后果，逻辑最清晰。
- **追问「FlashAttention 降低复杂度了吗？」**：没有，它仍是 O(n²) 计算量，但通过 IO 优化（减少 HBM 读写）大幅提速省显存，数学结果不变。这是常见误区，能澄清是加分。
- **结构化收尾**：用『原理 → 后果 → 对策』三段式，最能打动面试官。

🗣️ 收尾：「一句话：**自注意力要两两算相关性，复杂度 O(n²·d)，序列翻倍开销变 4 倍。由此带来上下文窗口上限、位置外推差、中间迷失、生成串行四个局限。模型层用稀疏/滑窗注意力、FlashAttention、RoPE 外推、投机解码缓解；应用层靠关键信息前置、上下文管理、流式返回应对。**」

---

## 结语：怎么用这份讲稿

- **第一遍**：每题从头念一遍，把 🗣️ 部分说出声、📐 部分写在纸上，对着内嵌 SVG 走流程。
- **第二遍**：盖住答案，只看每题标题，自己复述四步（直觉 → 推导 → 落地 → 追问）。
- **第三遍**：只看每题末尾的「🗣️ 收尾一句话」，确认能瞬间展开成 15 分钟讲解。
- 配合 `transformer_interview.md`（速查提纲版）做最终冲刺自测。
