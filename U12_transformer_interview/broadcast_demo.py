"""
广播(Broadcasting)对实际数值影响的演示

场景对应 Transformer 位置编码:
    x        : [batch_size, seq_len, dim_model]  (3维)
    part_pe  : [seq_len, dim_model]              (2维)
    x + part_pe  ->  part_pe 在 batch 维上被"复制"后逐元素相加
"""

import numpy as np

np.set_printoptions(precision=1, suppress=True)


def line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------------------
# 例 1: 最简单的情况 —— 2维矩阵 + 1维向量
# ----------------------------------------------------------------------
line("例1: [3, 4] 的矩阵  +  [4] 的向量")

A = np.array([
    [0,  0,  0,  0],
    [10, 10, 10, 10],
    [20, 20, 20, 20],
])  # shape (3, 4)

b = np.array([1, 2, 3, 4])  # shape (4,)

print("A.shape =", A.shape, "   b.shape =", b.shape)
print("A =\n", A)
print("b =", b)
print("\nA + b =\n", A + b)
print("""
解读: b 被"看作"沿着每一行复制了3份:
    [1,2,3,4]
    [1,2,3,4]
    [1,2,3,4]
然后与 A 逐元素相加。每一行加的都是同一个 b。
""")


# ----------------------------------------------------------------------
# 例 2: 你截图里的真实场景 —— 3维 + 2维 (位置编码)
# ----------------------------------------------------------------------
line("例2: x[batch=2, seq=3, dim=4]  +  part_pe[seq=3, dim=4]")

batch_size, seq_len, dim_model = 2, 3, 4

# x: 假设两个样本, 数值用 100/200 区分 batch, 方便观察
x = np.array([
    [[100, 100, 100, 100],   # batch0, 位置0
     [101, 101, 101, 101],   # batch0, 位置1
     [102, 102, 102, 102]],  # batch0, 位置2

    [[200, 200, 200, 200],   # batch1, 位置0
     [201, 201, 201, 201],   # batch1, 位置1
     [202, 202, 202, 202]],  # batch1, 位置2
], dtype=float)  # shape (2, 3, 4)

# part_pe: 每个位置一个编码向量, 与 batch 无关
part_pe = np.array([
    [0.0, 0.1, 0.2, 0.3],   # 位置0 的编码
    [1.0, 1.1, 1.2, 1.3],   # 位置1 的编码
    [2.0, 2.1, 2.2, 2.3],   # 位置2 的编码
])  # shape (3, 4)

print("x.shape       =", x.shape)
print("part_pe.shape =", part_pe.shape)

result = x + part_pe  # 广播发生在 batch 维

print("\n结果 result.shape =", result.shape)
print("\nresult[batch0] =\n", result[0])
print("\nresult[batch1] =\n", result[1])

print("""
解读:
  - part_pe 只有 [seq, dim], 缺一个 batch 维。
  - 广播时它被当作 [1, seq, dim], 再沿 batch 维复制 2 份。
  - 关键: 两个 batch 加的是【完全相同】的位置编码。
    batch0 的位置0 加 [0,0.1,0.2,0.3];
    batch1 的位置0 也加 [0,0.1,0.2,0.3]。
  - 所以同一位置、不同样本得到的位置偏移一致, 这正是位置编码想要的效果。
""")


# ----------------------------------------------------------------------
# 例 3: 广播规则验证 —— 什么时候能广播, 什么时候报错
# ----------------------------------------------------------------------
line("例3: 广播规则 (从右往左对齐, 维度相等或其中一个为1)")

cases = [
    ((2, 3, 4), (3, 4)),   # OK: 4==4, 3==3, 左边补1
    ((2, 3, 4), (4,)),     # OK: 4==4, 中间补1
    ((2, 3, 4), (1, 4)),   # OK: 4==4, 3 vs 1 -> 广播
    ((2, 3, 4), (3,)),     # 错误: 3 vs 4 不匹配
]

for s1, s2 in cases:
    s2_str = str(s2)
    try:
        r = np.broadcast_shapes(s1, s2)
        print(f"  {str(s1):<12} +  {s2_str:<10} ->  {str(r):<12} OK")
    except ValueError:
        print(f"  {str(s1):<12} +  {s2_str:<10} ->  报错 (维度不匹配)")

print("""
规则: 把两个 shape 右对齐, 逐维检查:
  - 维度相等        -> 通过
  - 其中一个是 1    -> 通过 (会被复制扩展)
  - 缺失的维度      -> 当作 1
  - 否则            -> 报错
""")
