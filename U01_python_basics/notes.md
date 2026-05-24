# U01 学习笔记 · Python 强化

> 用自己的话写，不要抄。写得烂没关系，重要的是自己组织语言。

## 1. 字典 dict

### dict.get(key, default) 的作用
（你来写：为什么要用 get 而不是 `d[key]`？）

字典和列表都能用[],区别在于存在性检查
d[key]:字典里如果key不存在 -> 抛 keyerror
d.get(key, default):字典里如果key不存在 -> 返回 default
统计词频时用get最方便 counter[w] = counter.get(w, 0) + 1


### 常用操作
- 增：dict[key] = value
- 删：dict.pop(key) 或 del dict[key]
- 改：dict[key] = value
- 查：dict.get(key, default) 或 dict[key]
- 判断存在：key in dict

## 2. 列表推导式

### 基本模板
`[表达式 for 变量 in 可迭代 if 条件]`

### 我自己的例子
（写 3 个你自己造的例子）
[i ** 2 for i in range(10) if i % 2 == 0]
[3 * x for x in (3,80) if x < 50]
[x for x in range(10) if x % 2 == 0]
## 3. lambda

### 什么时候用？
需要一个一次性的简短函数，不值得用 def 单独定义时

### sorted + lambda 的搭配
sorted(iterable, key=lambda x: -x[1]) -> 根据可迭代对象的第二个元素倒序排序

## 4. 类 class

### self 是什么？
self 就是"当前这个实例本身"
调用 v.add_word("你") 时，Python 自动把 v 传给 self
所以 self.w2i 实际上就是 v.w2i
  → 这样 v1 和 v2 的数据互相隔离

### __init__ 什么时候被调用？
创建类实例时自动调用，用来初始化这个实例的属性
例：v = Vocab()  → 自动跑 __init__，初始化 self.w2i 和 self.i2w

## 5. 遇到的坑

- [1.2-1] 误把 s[:-1] 当"长度"，实际 s[:-1] 是切片（去掉最后一个字符）
  正确：字符串直接 list(s) 或 [c for c in s] 拆成字符列表
- [1.2-2] Python 里是 % 和 ==，不是 mod 和 =
- [1.3] decode 需要"遇到 eos 截断"不是"过滤 eos"
  列表推导式做不了 break，要用 for + break
- [其他] dict 是内置类型名，不要用作变量名 —— 会把内置给盖掉
