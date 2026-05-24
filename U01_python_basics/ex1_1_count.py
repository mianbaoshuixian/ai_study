"""
练习 1.1 · 字频统计

任务：实现 count_chars(text, k) 函数
- 输入：中文字符串 text，整数 k
- 输出：出现次数 Top k 的字，格式 [(字, 次数), ...]

要求：
- 必须用 dict.get() 方法统计
- 必须用 sorted + lambda 排序
- 不能用 collections.Counter（这次要手写练习）
"""


def count_chars(text, k=5):
    # TODO: 在这里实现
    pass

def count_chars(text, k):
    
    # dict = {}
    # for i in range(len(text)):
    #     dict[text[i]] = dict.get(text[i],0) + 1
    # dict = sorted(dict.items(),key = lambda x : -x[1])
    # return dict[:k] 

    counter = {}
    for ch in text:
        counter[ch] = counter.get(ch,0) + 1
    return counter

# ========== 测试（别改动下面） ==========
if __name__ == "__main__":
    # 测试 1
    result = count_chars("我爱中国，中国爱我，我爱我爱", k=5)
    print("测试 1:", result)
    # 期望类似: [('我', 4), ('爱', 4), ('中', 2), ('国', 2), ('，', 2)]

    # 测试 2
    result = count_chars("aaabbc", k=2)
    print("测试 2:", result)
    # 期望: [('a', 3), ('b', 2)]

    # 测试 3
    result = count_chars("", k=3)
    print("测试 3:", result)
    # 期望: []
