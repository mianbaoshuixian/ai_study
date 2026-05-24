"""
练习 1.2 · 列表推导式 3 连

每一题只能用一行列表推导式解决（不能用 for 循环多行写）
"""


# 题 1：把字符串列表拆成字符的二维列表
# 输入: ["你好世界", "人工智能"]
# 输出: [["你","好","世","界"], ["人","工","智","能"]]
def split_to_chars(sentences):
    # TODO: 一行列表推导式
    # return [sentences[:len(sentences) - 1][:len(sentences[:len(sentences) - 1]) - 1]]
    # return [[c for c in s] for s in sentences]
    # return [[s[:-1]] for s in sentences] 
    return [list(s) for s in sentences]
    

    
# 题 2：从 1~20 中筛出所有 3 的倍数，并返回它们的平方
# 输出: [9, 36, 81, 144, 225, 324, 441]
def multiples_of_3_squared():
    # TODO: 一行列表推导式
    return [x*x for x in range(1,21) if x % 3 == 0]


# 题 3：提取 pairs 中所有的中文部分
# 输入: [("hi","你好"), ("bye","再见"), ("ok","好的")]
# 输出: ["你好", "再见", "好的"]
def extract_chinese(pairs):
    # TODO: 一行列表推导式
    return [pair[1] for pair in pairs]


# ========== 测试 ==========
if __name__ == "__main__":
    print("题 1:", split_to_chars(["你好世界", "人工智能"]))
    # 期望: [['你', '好', '世', '界'], ['人', '工', '智', '能']]

    print("题 2:", multiples_of_3_squared())
    # 期望: [9, 36, 81, 144, 225, 324, 441]

    print("题 3:", extract_chinese([("hi", "你好"), ("bye", "再见"), ("ok", "好的")]))
    # 期望: ['你好', '再见', '好的']
