"""
练习 1.3 · 核心挑战 🌟 · 手写 Vocab 类

这是翻译模型必备组件，以后会在 U09-U11 反复使用！
请认真实现。

词表 Vocab 的作用：把"词/字" <-> "整数 id" 互相转换，
因为神经网络只能处理数字，不能直接处理文字。
"""


class Vocab:
    def __init__(self, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]):
        """
        初始化词表
        - 默认 special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
          分别分配 id: 0, 1, 2, 3
        - <pad>: 填充，用于把不等长的句子对齐
        - <unk>: 未知词
        - <bos>: 句子开始
        - <eos>: 句子结束

        提示：需要两个字典
          self.w2i: word -> id
          self.i2w: id -> word
        """
        self.w2i ={}
        self.i2w = {}
        for idx, token in enumerate(special_tokens):
            self.w2i[token] = idx
            self.i2w[idx] = token

        # TODO
        

    def add_word(self, word):
        """添加一个词，如果已存在则跳过"""
        # TODO
        if word not in self.w2i: # 这里是怎么判断word是否已经存在的？
            self.i2w[len(self.i2w)] = word
            self.w2i[word] = len(self.w2i)

    def build_from_texts(self, texts, min_freq=1):
        """
        从文本构建词表
        - texts: List[List[str]]，如 [["你","好"], ["世","界","你"]]
        - min_freq: 只收录出现次数 >= min_freq 的字

        步骤提示：
        1. 先统计所有 token 的出现次数（类似练习 1.1）
        2. 过滤掉出现次数 < min_freq 的
        3. 依次 add_word
        """
        # TODO
        dict = {}
        for t in texts:
            for w in t:
                dict[w] = dict.get(w,0) + 1
        for k,v in dict.items():
            if v >= min_freq:
                self.add_word(word=k)

    def word2id(self, word):
        """不存在则返回 <unk> 的 id"""
        # TODO
        if word not in self.w2i:
            return self.w2i["<unk>"]
        return self.w2i.get(word)



    def id2word(self, idx):
        """id 转 word"""
        # TODO
        if idx in self.i2w:
            return self.i2w.get(idx)

    def __len__(self):
        """返回词表大小（让 len(vocab) 能用）"""
        # TODO
        return len(self.i2w)

    def encode(self, tokens):
        """
        输入 ["你","好","哈"]
        输出 [id_你, id_好, id_<unk>]
        """
        # TODO
        return [self.word2id(token) for token in tokens]

    def decode(self, ids):
        """
        输入 id 列表
        输出字符列表，遇到 <eos> 就截断（不包含 <eos>）

        例: [4, 5, 3, 6] 且 3=<eos>  ->  ['你', '好']
        """
        # TODO
        # return [self.id2word(id) for id in ids if id != self.word2id("<eos>")] # 应该有个类似于break的功能？
        list = []
        for id in ids :
            if id == self.word2id("<eos>"):
                break
            list.append(self.id2word(id))
        return list


# ========== 测试 ==========
if __name__ == "__main__":
    v = Vocab()

    # 测试 1: 初始状态，只有 4 个特殊 token
    print("测试 1 词表大小:", len(v))  # 期望 4

    # 测试 2: 构建词表
    v.build_from_texts(
        [["你", "好"], ["世", "界", "你", "好"]],
        min_freq=1
    )
    print("测试 2 词表大小:", len(v))  # 期望 8 (4 特殊 + 你好世界)

    # 测试 3: encode 未知字
    ids = v.encode(["你", "好", "哈"])  # 哈没见过
    print("测试 3 encode:", ids)
    print("  其中'哈'的 id 应该等于 <unk> 的 id:", v.word2id("<unk>"))

    # 测试 4: decode 遇到 <eos> 截断
    eos_id = v.word2id("<eos>")  # 应该是 3
    test_ids = [v.word2id("你"), v.word2id("好"), eos_id, v.word2id("世")]
    print("测试 4 decode:", v.decode(test_ids))  # 期望 ['你', '好']

    # 测试 5: min_freq 过滤
    v2 = Vocab()
    v2.build_from_texts(
        [["a", "b"], ["a", "c"], ["a", "b"]],
        min_freq=2
    )
    # a 出现 3 次, b 出现 2 次, c 出现 1 次（被过滤）
    print("测试 5 词表大小:", len(v2))  # 期望 6 (4 特殊 + a,b)
    print("  c 应返回 <unk> id:", v2.word2id("c") == v2.word2id("<unk>"))
