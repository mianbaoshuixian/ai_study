import nbformat as nbf
from pathlib import Path


BASE = Path(__file__).parent


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


lesson_cells = [
    md(
        """
# U11 | Bahdanau Attention：让 Decoder 每一步回头看源句

本单元目标：在 U10 翻译 baseline 上加入 **Bahdanau Attention**，让 Decoder 不再只依赖一个固定大小的 context vector，而是在生成每个英文词时，动态查看中文源句的不同位置。

过关标准：

1. 能说清楚 `encoder_outputs`、`decoder_hidden`、`attention weights`、`context` 四个张量各自是什么。
2. 能跑通 Hugging Face 小批量中英数据，使用专业分词器构建 batch。
3. 能画出一张 attention 热力图，并解释“每一行”和“每一列”代表什么。
4. 能独立补全练习里的 `BahdanauAttention` 和 `AttnDecoder`。
"""
    ),
    md(
        """
## 1. 全局图：U11 在整体项目里的位置

U10 的问题是 Encoder 把整句中文压成一个 hidden，Decoder 从头到尾都靠这个 hidden 翻译：

```text
U10 baseline
中文 src -> Encoder -> 最后 hidden/context -> Decoder -> 英文 tgt
                         ↑
                  所有源句信息都挤在这里
```

U11 的改造是：Encoder 不只返回最后 hidden，还返回每个源位置的输出 `encoder_outputs`。Decoder 每生成一个词，就用当前 hidden 去给源句每个位置打分，然后加权求和得到这一步专用的 context。

```text
U11 attention
中文 src -> Encoder -> encoder_outputs: [每个中文 token 的表示]
                         ↓    ↓    ↓
Decoder 第 t 步 hidden -> Attention 打分 -> attention weights -> context_t -> 预测英文第 t 个词
```

一句话：**U10 是“先读完再凭记忆翻译”，U11 是“每写一个词都回头看原文”。**
"""
    ),
    md(
        """
## 2. 为什么必须从固定 context 改成 Attention

固定 context 的设计选择很自然：Encoder 最后一步 hidden 看起来像是“整句摘要”。短句时它能工作，但句子稍长就会出现三个问题。

| 问题 | 发生在哪里 | 后果 |
|---|---|---|
| 信息瓶颈 | 整句被压进一个向量 | 前面的词、细节、顺序容易丢 |
| 对齐缺失 | Decoder 不知道该看哪个源词 | 生成时容易漏译、重复、乱套 |
| 可解释性弱 | 没有每一步看哪里的信息 | 很难判断模型为什么翻错 |

Attention 的设计动机不是“多加一层显得高级”，而是解决一个具体需求：**Decoder 每一步需要一个和当前生成位置相关的源句信息**。
"""
    ),
    md(
        """
## 3. 本节核心公式：Bahdanau Attention

设：

- `encoder_outputs`：源句每个位置的表示，形状 `(B, S, 2H_enc)`。
- `decoder_hidden`：Decoder 当前 hidden，形状 `(1, B, H_dec)`。
- `S`：源句长度。

Bahdanau Attention 做三步。

第一步，对每个源位置打分：

$$
e_{t,i} = \\mathbf{v}_a^\\top \\tanh\\left(\\mathbf{W}_e \\mathbf{h}^{enc}_i + \\mathbf{W}_d \\mathbf{s}_{t-1}\\right)
$$

第二步，对分数做 softmax，得到权重：

$$
\\alpha_{t,i} = \\frac{\\exp(e_{t,i})}{\\sum_{j=1}^{S}\\exp(e_{t,j})}
$$

第三步，用权重加权求和源句表示，得到本步 context：

$$
\\mathbf{c}_t = \\sum_{i=1}^{S} \\alpha_{t,i}\\mathbf{h}^{enc}_i
$$

这里的 $\\alpha_{t,i}$ 可以理解成：**生成第 $t$ 个英文词时，模型有多关注第 $i$ 个中文 token**。
"""
    ),
    md(
        """
## 4. 代码框架总览：这节课新增哪些模块

| 模块 | 作用 | 关键输入 shape | 关键输出 shape |
|---|---|---|---|
| `Encoder` | 双向 GRU 编码中文 | `src: (B,S)`, `src_len: (B,)` | `encoder_outputs: (B,S,2H_enc)`, `hidden: (1,B,H_dec)` |
| `BahdanauAttention` | 给源句每个位置打分 | `decoder_hidden: (1,B,H_dec)`, `encoder_outputs: (B,S,2H_enc)` | `context: (B,2H_enc)`, `attn_weights: (B,S)` |
| `AttnDecoder` | 带 attention 的一步解码 | `x: (B,1)` | `logits: (B,V)`, `hidden`, `attn_weights` |
| `Seq2SeqAttention` | 组装训练循环 | `src, src_len, tgt` | `outputs: (B,T,V)`, `attentions: (B,T,S)` |

注意这里 Encoder 用双向 GRU，所以 `encoder_outputs` 最后一维是 `2H_enc`。Decoder hidden 维度是 `H_dec`，两者维度不同，所以需要投影层对齐。
"""
    ),
    md(
        """
## 5. 环境与依赖

本节默认使用：

- Hugging Face `datasets`：加载小批量中英翻译数据。
- `jieba`：中文专业分词器，默认精确模式。
- `sacremoses`：英文 Moses 分词器。
- PyTorch：搭建 GRU + attention 模型。

安装命令：

```bash
python -m pip install datasets sacremoses
```

说明：`datasets` 是 Python 包，不是 npm 包。npm 装的是 Node.js 生态，不能让 Python Notebook 里的 `from datasets import load_dataset` 生效。
"""
    ),
    code(
        """
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import jieba
from sacremoses import MosesTokenizer

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except Exception as e:
    HF_DATASETS_AVAILABLE = False
    print('datasets 导入失败，将使用 fallback 数据:', repr(e))

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    MATPLOTLIB_AVAILABLE = False
    print('matplotlib 导入失败，热力图小节会跳过:', repr(e))

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ['<pad>', '<sos>', '<eos>', '<unk>']

EMBED_DIM = 64
ENC_HIDDEN_DIM = 64
DEC_HIDDEN_DIM = 128
BATCH_SIZE = 16
MAX_PAIRS = 128
MIN_FREQ = 1
LR = 1e-3
EPOCHS = 20
USE_HF_DATASETS = False  # 改成 True 后会尝试联网下载 Hugging Face 小批量数据

random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device =', device)
"""
    ),
    md(
        """
## 6. 数据：优先 Hugging Face，小批量即可

我们优先尝试加载 Hugging Face 上的中英翻译数据。为了让课程运行时间可控，只取前面一小段。

本节代码写得比较保守：不同翻译数据集的字段名可能不同，有的叫 `translation`，有的叫 `zh`/`en`，有的叫 `source`/`target`。所以我们写一个抽取函数，把数据统一成：

```python
raw_pairs = [(中文句子, 英文句子), ...]
```

如果下载失败或字段不匹配，就使用 fallback 数据继续学习 attention。
"""
    ),
    code(
        """
def normalize_text(text):
    return str(text).replace('\\u3000', ' ').strip()


def extract_zh_en(example):
    \"\"\"把不同数据集格式统一成 (zh, en)。\"\"\"
    if 'translation' in example and isinstance(example['translation'], dict):
        trans = example['translation']
        zh = trans.get('zh') or trans.get('zh-cn') or trans.get('cmn') or trans.get('chinese')
        en = trans.get('en') or trans.get('eng') or trans.get('english')
        if zh and en:
            return normalize_text(zh), normalize_text(en)

    zh_keys = ['zh', 'zh-cn', 'chinese', 'src_zh', 'source_zh']
    en_keys = ['en', 'eng', 'english', 'tgt_en', 'target_en']
    for zk in zh_keys:
        for ek in en_keys:
            if zk in example and ek in example:
                return normalize_text(example[zk]), normalize_text(example[ek])

    if 'source' in example and 'target' in example:
        return normalize_text(example['source']), normalize_text(example['target'])

    return None


def load_translation_pairs_from_hf(max_pairs=MAX_PAIRS):
    if not USE_HF_DATASETS:
        raise RuntimeError('USE_HF_DATASETS=False，当前使用离线 fallback 数据')
    if not HF_DATASETS_AVAILABLE:
        raise RuntimeError('未安装或无法导入 datasets')

    candidates = [
        ('Helsinki-NLP/opus-100', 'en-zh', f'train[:{max_pairs * 4}]'),
        ('wmt/wmt19', 'zh-en', f'train[:{max_pairs * 4}]'),
    ]

    last_error = None
    for name, config, split in candidates:
        try:
            ds = load_dataset(name, config, split=split)
            pairs = []
            for ex in ds:
                item = extract_zh_en(ex)
                if item is None:
                    continue
                zh, en = item
                if 0 < len(zh) <= 80 and 0 < len(en) <= 120:
                    pairs.append((zh, en.lower()))
                if len(pairs) >= max_pairs:
                    print(f'使用 Hugging Face 数据集: {name} / {config}, pairs={len(pairs)}')
                    return pairs
            if pairs:
                print(f'使用 Hugging Face 数据集: {name} / {config}, pairs={len(pairs)}')
                return pairs
        except Exception as e:
            last_error = e
            print(f'加载 {name} / {config} 失败:', repr(e))

    raise RuntimeError(f'Hugging Face 数据加载失败: {last_error!r}')


fallback_pairs = [
    ('我 爱 你', 'i love you'),
    ('谢谢 你', 'thank you'),
    ('猫 在 睡觉', 'the cat is sleeping'),
    ('狗 在 跑步', 'the dog is running'),
    ('老师 读书', 'the teacher reads a book'),
    ('学生 写字', 'the student writes words'),
    ('今天天气 很 好', 'the weather is nice today'),
    ('明天 见', 'see you tomorrow'),
    ('我 喜欢 猫', 'i like cats'),
    ('他 在 看书', 'he is reading a book'),
    ('她 喜欢 音乐', 'she likes music'),
    ('我们 学习 翻译', 'we study translation'),
]

try:
    raw_pairs = load_translation_pairs_from_hf(MAX_PAIRS)
except Exception as e:
    print('使用 fallback_pairs。原因:', repr(e))
    raw_pairs = fallback_pairs

print('样本数:', len(raw_pairs))
for zh, en in raw_pairs[:5]:
    print(zh, ' -> ', en)
"""
    ),
    md(
        """
## 7. 专业分词：中文 jieba，英文 Moses

U09/U10 为了降低难度，中文直接 `list(sentence)` 按字切分。这能帮你练 shape，但真实翻译里不够合理。

例子：

```text
我喜欢机器翻译
按字：我 / 喜 / 欢 / 机 / 器 / 翻 / 译
分词：我 / 喜欢 / 机器翻译
```

按字切分会把“机器翻译”拆碎，模型更难学词义。U11 开始用专业分词器：

- 中文：`jieba.lcut(text)`，默认精确模式。
- 英文：`MosesTokenizer.tokenize(text)`，能更规范地处理英文标点和缩写。

为了兼容 fallback 数据里已经用空格分好的中文短句，我们写一个规则：如果中文句子里已经有空格，就按空格切；否则用 `jieba.lcut`。
"""
    ),
    code(
        """
moses_en = MosesTokenizer(lang='en')


def tokenize_zh(text):
    text = normalize_text(text)
    if ' ' in text:
        return [tok for tok in text.split() if tok]
    return [tok for tok in jieba.lcut(text) if tok.strip()]


def tokenize_en(text):
    text = normalize_text(text).lower()
    return moses_en.tokenize(text, return_str=False)


for zh, en in raw_pairs[:3]:
    print('ZH:', zh, '=>', tokenize_zh(zh))
    print('EN:', en, '=>', tokenize_en(en))
"""
    ),
    md(
        """
## 8. Vocab、Dataset、collate_fn

这部分沿用 U09/U10 的习惯，但要注意：现在 token 是“词”，不是“字”。

数据流仍然是：

```text
原始句子 -> 专业分词 -> Vocab 转 id -> 加 SOS/EOS -> Dataset -> collate_fn 动态 padding
```

Attention 里仍然需要 `src_len`，因为 Encoder 要 `pack_padded_sequence`，并且 attention 要用 `src.ne(PAD)` 做 mask，避免把权重分给 padding 位置。
"""
    ),
    code(
        """
class Vocab:
    def __init__(self, token_lists, min_freq=1):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        self.itos = list(SPECIALS)
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.itos:
                self.itos.append(token)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(token, UNK) for token in tokens]

    def decode(self, ids):
        return [self.itos[int(i)] for i in ids]


def sentence_to_ids(sentence, vocab, tokenizer, add_sos=False, add_eos=True):
    ids = vocab.encode(tokenizer(sentence))
    if add_sos:
        ids = [SOS] + ids
    if add_eos:
        ids = ids + [EOS]
    return ids


def pad_sequence(ids_list, pad_id=PAD):
    max_len = max(len(ids) for ids in ids_list)
    padded = [ids + [pad_id] * (max_len - len(ids)) for ids in ids_list]
    return torch.tensor(padded, dtype=torch.long)


src_vocab = Vocab([tokenize_zh(zh) for zh, en in raw_pairs], min_freq=MIN_FREQ)
tgt_vocab = Vocab([tokenize_en(en) for zh, en in raw_pairs], min_freq=MIN_FREQ)

print('src vocab size =', len(src_vocab))
print('tgt vocab size =', len(tgt_vocab))
print('src vocab 前 20 个:', src_vocab.itos[:20])
print('tgt vocab 前 20 个:', tgt_vocab.itos[:20])
"""
    ),
    code(
        """
class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        zh, en = self.pairs[idx]
        src_ids = sentence_to_ids(zh, src_vocab, tokenize_zh, add_sos=False, add_eos=True)
        tgt_ids = sentence_to_ids(en, tgt_vocab, tokenize_en, add_sos=True, add_eos=True)
        return src_ids, tgt_ids


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    src_list, tgt_list = zip(*batch)

    src_len = torch.tensor([len(src_ids) for src_ids in src_list], dtype=torch.long)
    src = pad_sequence(list(src_list))
    tgt = pad_sequence(list(tgt_list))
    return src, src_len, tgt


dataset = TranslationDataset(raw_pairs)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

src, src_len, tgt = next(iter(loader))
print('src shape:', src.shape)
print('src_len:', src_len[:8].tolist())
print('tgt shape:', tgt.shape)
"""
    ),
    md(
        """
## 9. Encoder：双向 GRU + bridge

U10 的 Encoder 只需要返回最后 hidden。U11 的 Encoder 必须多返回一个东西：`encoder_outputs`。

为什么用双向 GRU？

- 单向 GRU 的第 i 个输出只看过第 i 个位置之前的内容。
- 双向 GRU 的第 i 个输出同时包含左边和右边的上下文。
- Attention 要在每个源位置上做选择，因此每个位置的表示越完整越好。

但双向 Encoder 的 hidden 形状和 Decoder hidden 不一样，所以需要一个 `bridge`：

```text
h_forward, h_backward -> concat -> Linear -> tanh -> decoder initial hidden
```
"""
    ),
    code(
        """
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.gru = nn.GRU(
            embed_dim,
            enc_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.bridge = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)

    def forward(self, src, src_len):
        embedded = self.embedding(src)
        packed = pack_padded_sequence(
            embedded,
            src_len.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        packed_outputs, hidden = self.gru(packed)
        encoder_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src.size(1),
        )

        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        decoder_init = torch.tanh(self.bridge(hidden_cat)).unsqueeze(0)
        return encoder_outputs, decoder_init


encoder_test = Encoder(len(src_vocab), EMBED_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM)
enc_out, enc_hidden = encoder_test(src, src_len)
print('encoder_outputs:', enc_out.shape)
print('decoder_init_hidden:', enc_hidden.shape)
"""
    ),
    md(
        """
## 10. BahdanauAttention：打分、mask、softmax、加权求和

这是本节最重要的类。

先看 shape：

```text
encoder_outputs: (B, S, 2H_enc)
decoder_hidden:  (1, B, H_dec)
query:           (B, 1, H_dec)
scores:          (B, S)
attn_weights:    (B, S)
context:         (B, 2H_enc)
```

为什么要 mask？

因为 batch 里有 padding。Attention 的 softmax 会把权重分给每个位置，如果不 mask，模型可能关注 `<pad>`。所以我们把 pad 位置的 score 改成一个很小的数，让 softmax 后几乎为 0。
"""
    ),
    code(
        """
class BahdanauAttention(nn.Module):
    def __init__(self, enc_output_dim, dec_hidden_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_output_dim, dec_hidden_dim, bias=False)
        self.W_dec = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        query = decoder_hidden[-1].unsqueeze(1)
        energy = torch.tanh(self.W_enc(encoder_outputs) + self.W_dec(query))
        scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(~src_mask, -1e9)

        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


attn_test = BahdanauAttention(ENC_HIDDEN_DIM * 2, DEC_HIDDEN_DIM)
src_mask = src.ne(PAD)
context, attn_weights = attn_test(enc_hidden, enc_out, src_mask)
print('context:', context.shape)
print('attn_weights:', attn_weights.shape)
print('每条样本 attention 权重和:', attn_weights.sum(dim=1)[:5])
"""
    ),
    md(
        """
## 11. AttnDecoder：每一步先看源句，再预测词

U10 Decoder 的输入只有当前 token embedding。

U11 Decoder 的输入变成：

```text
当前 token embedding + 本步 attention context
```

然后 GRU 输出后，再把三类信息拼起来预测词：

```text
GRU output + context + embedding -> Linear -> vocab logits
```

为什么把 `context` 也送进 `fc`？因为预测当前词时，模型不仅需要“Decoder 当前状态”，也需要“这一刻从源句拿到的信息”。
"""
    ),
    code(
        """
class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_output_dim, dec_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.attention = BahdanauAttention(enc_output_dim, dec_hidden_dim)
        self.gru = nn.GRU(embed_dim + enc_output_dim, dec_hidden_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim + enc_output_dim + dec_hidden_dim, vocab_size)

    def forward(self, x, hidden, encoder_outputs, src_mask):
        embedded = self.embedding(x)
        context, attn_weights = self.attention(hidden, encoder_outputs, src_mask)

        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.gru(rnn_input, hidden)

        features = torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=-1)
        logits = self.fc(features)
        return logits, hidden, attn_weights


decoder_test = AttnDecoder(len(tgt_vocab), EMBED_DIM, ENC_HIDDEN_DIM * 2, DEC_HIDDEN_DIM)
input_tok = tgt[:, 0:1]
logits, next_hidden, attn_weights = decoder_test(input_tok, enc_hidden, enc_out, src_mask)
print('logits:', logits.shape)
print('next_hidden:', next_hidden.shape)
print('attn_weights:', attn_weights.shape)
"""
    ),
    md(
        """
## 12. Seq2SeqAttention：组装训练流程

训练时仍然使用 teacher forcing。变化只有一个：Decoder 每一步除了接收 `input_token` 和 `hidden`，还要接收 `encoder_outputs` 和 `src_mask`。

返回值有两个：

- `outputs`: `(B, T, V)`，用于算 loss。
- `attentions`: `(B, T, S)`，用于可视化和调试。

注意：第 0 个时间步是 `<sos>`，不需要预测，所以 loss 仍然从 `[:, 1:, :]` 开始算。
"""
    ),
    code(
        """
class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        src_steps = src.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
        attentions = torch.zeros(batch_size, tgt_len, src_steps, device=src.device)

        encoder_outputs, hidden = self.encoder(src, src_len)
        src_mask = src.ne(PAD)
        input_token = tgt[:, 0:1]

        for t in range(1, tgt_len):
            logits, hidden, attn_weights = self.decoder(
                input_token,
                hidden,
                encoder_outputs,
                src_mask,
            )
            outputs[:, t, :] = logits
            attentions[:, t, :] = attn_weights

            if random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t:t+1]
            else:
                input_token = logits.argmax(dim=-1, keepdim=True)

        return outputs, attentions
"""
    ),
    md(
        """
## 13. 训练

注意几个细节：

1. `src`、`tgt` 放到 `device`，但 `src_len` 可以保持 CPU，`pack_padded_sequence` 用 CPU 长度最稳。
2. `CrossEntropyLoss(ignore_index=PAD)` 屏蔽 target 里的 padding。
3. `clip_grad_norm_` 防止 GRU 梯度爆炸。
4. HF 数据比 fallback 稍复杂，20 轮只是演示；想看更明显效果，可以把 `EPOCHS` 调大。
"""
    ),
    code(
        """
encoder = Encoder(len(src_vocab), EMBED_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM)
decoder = AttnDecoder(len(tgt_vocab), EMBED_DIM, ENC_HIDDEN_DIM * 2, DEC_HIDDEN_DIM)
model = Seq2SeqAttention(encoder, decoder).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
V_tgt = len(tgt_vocab)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n_steps = 0.0, 0

    for src, src_len, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        outputs, attentions = model(src, src_len, tgt, teacher_forcing_ratio=0.5)
        loss = loss_fn(
            outputs[:, 1:, :].reshape(-1, V_tgt),
            tgt[:, 1:].reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

    if epoch == 1 or epoch % 5 == 0:
        print(f'Epoch {epoch:3d} | loss={total_loss / max(n_steps, 1):.4f}')
"""
    ),
    md(
        """
## 14. 推理：翻译时也保存 attention

推理和 U10 一样是 greedy decode：

```text
输入 SOS -> 预测第 1 个词 -> 把预测词喂回去 -> 继续预测 -> 遇到 EOS 停止
```

U11 多保存一个 `attn_rows`。每生成一个英文词，就保存这一行 attention 权重。最后得到一个矩阵：

```text
行：生成出来的英文 token
列：中文源句 token
值：这一英文 token 对每个中文 token 的关注程度
```
"""
    ),
    code(
        """
def translate_with_attention(model, sentence, max_len=30):
    model.eval()
    with torch.no_grad():
        src_tokens = tokenize_zh(sentence)
        src_ids = sentence_to_ids(sentence, src_vocab, tokenize_zh, add_sos=False, add_eos=True)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_len = torch.tensor([len(src_ids)], dtype=torch.long)

        encoder_outputs, hidden = model.encoder(src, src_len)
        src_mask = src.ne(PAD)
        input_token = torch.tensor([[SOS]], dtype=torch.long, device=device)

        result_ids = []
        attn_rows = []

        for _ in range(max_len):
            logits, hidden, attn_weights = model.decoder(
                input_token,
                hidden,
                encoder_outputs,
                src_mask,
            )
            next_id = logits.argmax(dim=-1).item()
            if next_id == EOS:
                break
            result_ids.append(next_id)
            attn_rows.append(attn_weights.squeeze(0).cpu())
            input_token = torch.tensor([[next_id]], dtype=torch.long, device=device)

        pred_tokens = tgt_vocab.decode(result_ids)
        pred_text = ' '.join(pred_tokens)
        if attn_rows:
            attn_matrix = torch.stack(attn_rows)
        else:
            attn_matrix = torch.empty(0, len(src_ids))

        return pred_text, attn_matrix, src_tokens + ['<eos>'], pred_tokens


for zh, en in raw_pairs[:8]:
    pred, attn_matrix, src_tokens, pred_tokens = translate_with_attention(model, zh)
    print(f'{zh} -> {pred}   (gold: {en})')
"""
    ),
    md(
        """
## 15. Attention 热力图

热力图的读法：

- 横轴：源句中文 token。
- 纵轴：模型生成的英文 token。
- 颜色越深：生成这个英文词时越关注对应中文 token。

注意：attention 不是严格的人类词典对齐。它是模型内部为了降低 loss 学出的权重，但通常能给我们提供很有价值的调试线索。
"""
    ),
    code(
        """
def plot_attention(attn_matrix, src_tokens, pred_tokens):
    if not MATPLOTLIB_AVAILABLE:
        print('matplotlib 不可用，跳过热力图。')
        return
    if attn_matrix.numel() == 0 or not pred_tokens:
        print('没有生成 token，无法画热力图。')
        return

    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.8), max(3, len(pred_tokens) * 0.5)))
    im = ax.imshow(attn_matrix.numpy(), aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(src_tokens)), src_tokens, rotation=45, ha='right')
    ax.set_yticks(range(len(pred_tokens)), pred_tokens)
    ax.set_xlabel('source tokens')
    ax.set_ylabel('generated tokens')
    ax.set_title('Bahdanau attention weights')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()


sample_zh, sample_en = raw_pairs[0]
pred, attn_matrix, src_tokens, pred_tokens = translate_with_attention(model, sample_zh)
print(sample_zh, '->', pred, '(gold:', sample_en, ')')
plot_attention(attn_matrix, src_tokens, pred_tokens)
"""
    ),
    md(
        """
## 16. 一个关键追问：为什么 attention 可以缓解长句问题？

不要把 attention 理解成“魔法查词典”。它缓解长句问题的原因更具体：

U10 中，Decoder 第 1 步、第 5 步、第 10 步拿到的源句信息几乎都是同一个最后 hidden。句子越长，这个 hidden 越难同时保留所有细节。

U11 中，Decoder 每一步都会重新计算：

```text
当前我准备生成什么？
源句每个位置和我当前状态有多相关？
我应该从哪些源位置拿信息？
```

所以它把“整句一次性压缩”的压力，改成了“每一步动态检索”。这就是 attention 的核心价值。
"""
    ),
    md(
        """
## 17. 本单元小结

你需要能默写这条数据流：

```text
src -> Embedding -> BiGRU -> encoder_outputs + decoder_init_hidden
                           ↓
decoder_hidden + encoder_outputs -> BahdanauAttention -> context + attn_weights
                           ↓
[tgt token embedding, context] -> Decoder GRU -> logits
```

本节最重要的四个 shape：

| 张量 | shape | 含义 |
|---|---|---|
| `encoder_outputs` | `(B,S,2H_enc)` | 源句每个位置的表示 |
| `decoder_hidden` | `(1,B,H_dec)` | Decoder 当前状态 |
| `attn_weights` | `(B,S)` | 当前步对源句每个位置的关注权重 |
| `context` | `(B,2H_enc)` | 当前步从源句加权拿到的信息 |

进入练习前，请先自己回答：

1. 为什么 attention 要 mask 掉 PAD？
2. `torch.bmm(attn.unsqueeze(1), encoder_outputs)` 的三个维度分别是什么？
3. 为什么 U11 的 Encoder 要返回 `encoder_outputs`，而 U10 不需要？
"""
    ),
]


exercise_cells = [
    md(
        """
# U11 练习题 | Bahdanau Attention

过关标准：

1. 跑通 Hugging Face 小批量数据或 fallback 数据。
2. 补全 `BahdanauAttention`，确认 attention 权重每行求和接近 1。
3. 补全 `AttnDecoder` 和 `Seq2SeqAttention`。
4. 训练至少 5 轮，并能画出一张 attention 热力图。

建议：先独立写，卡住后再回 lesson 对照。
"""
    ),
    md(
        """
## 练习 11.1：数据加载 + 专业分词器

目标：

- 使用 Hugging Face `datasets` 加载小批量中英数据。
- 中文用 `jieba`，英文用 Moses tokenizer。
- 如果 HF 下载失败，使用 fallback 数据。

填空重点：`tokenize_zh`、`tokenize_en`、`extract_zh_en`。
"""
    ),
    code(
        """
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import jieba
from sacremoses import MosesTokenizer
from datasets import load_dataset

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ['<pad>', '<sos>', '<eos>', '<unk>']

EMBED_DIM = 64
ENC_HIDDEN_DIM = 64
DEC_HIDDEN_DIM = 128
BATCH_SIZE = 16
MAX_PAIRS = 128
LR = 1e-3
EPOCHS = 5

random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device =', device)

moses_en = MosesTokenizer(lang='en')


def normalize_text(text):
    return str(text).replace('\\u3000', ' ').strip()


def tokenize_zh(text):
    # TODO 1: 如果文本里已经有空格，按空格切；否则用 jieba.lcut
    pass


def tokenize_en(text):
    # TODO 2: 小写后用 MosesTokenizer 分词，返回 list[str]
    pass


def extract_zh_en(example):
    # TODO 3: 优先处理 example['translation'] = {'zh': ..., 'en': ...} 的情况
    # 返回 (zh, en)，失败返回 None
    pass


def load_translation_pairs_from_hf(max_pairs=MAX_PAIRS):
    ds = load_dataset('Helsinki-NLP/opus-100', 'en-zh', split=f'train[:{max_pairs * 4}]')
    pairs = []
    for ex in ds:
        item = extract_zh_en(ex)
        if item is None:
            continue
        zh, en = item
        if zh and en:
            pairs.append((zh, en.lower()))
        if len(pairs) >= max_pairs:
            break
    return pairs


fallback_pairs = [
    ('我 爱 你', 'i love you'),
    ('谢谢 你', 'thank you'),
    ('猫 在 睡觉', 'the cat is sleeping'),
    ('狗 在 跑步', 'the dog is running'),
]

try:
    raw_pairs = load_translation_pairs_from_hf(MAX_PAIRS)
except Exception as e:
    print('HF 数据加载失败，使用 fallback:', repr(e))
    raw_pairs = fallback_pairs

print('pairs:', len(raw_pairs))
print(raw_pairs[:2])
print(tokenize_zh(raw_pairs[0][0]))
print(tokenize_en(raw_pairs[0][1]))
"""
    ),
    md(
        """
## 练习 11.2：Vocab、Dataset、collate_fn

目标：复用 U09/U10 的数据管道，但这次 token 来自专业分词器。
"""
    ),
    code(
        """
class Vocab:
    def __init__(self, token_lists, min_freq=1):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        self.itos = list(SPECIALS)
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.itos:
                self.itos.append(token)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(token, UNK) for token in tokens]

    def decode(self, ids):
        return [self.itos[int(i)] for i in ids]


def sentence_to_ids(sentence, vocab, tokenizer, add_sos=False, add_eos=True):
    # TODO 1: 分词 -> encode -> 按需加 SOS/EOS
    pass


def pad_sequence(ids_list, pad_id=PAD):
    # TODO 2: padding 到 batch 内最长，返回 LongTensor
    pass


src_vocab = Vocab([tokenize_zh(zh) for zh, en in raw_pairs])
tgt_vocab = Vocab([tokenize_en(en) for zh, en in raw_pairs])


class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # TODO 3: 返回 src_ids, tgt_ids
        pass


def collate_fn(batch):
    # TODO 4: 按 src 长度降序，返回 src, src_len, tgt
    pass


loader = DataLoader(TranslationDataset(raw_pairs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
src, src_len, tgt = next(iter(loader))
print(src.shape, src_len.shape, tgt.shape)
"""
    ),
    md(
        """
## 练习 11.3：Encoder 输出每个源位置

目标：双向 GRU 返回：

- `encoder_outputs: (B,S,2H_enc)`
- `decoder_init: (1,B,H_dec)`
"""
    ),
    code(
        """
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        # TODO 1: embedding
        # TODO 2: bidirectional GRU
        # TODO 3: bridge Linear，把 2H_enc -> H_dec
        pass

    def forward(self, src, src_len):
        # TODO 4: embedding -> pack -> gru -> unpack
        # TODO 5: 拼接最后一层 forward/backward hidden，过 bridge 得到 decoder_init
        pass


enc = Encoder(len(src_vocab), EMBED_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM)
enc_out, enc_hidden = enc(src, src_len)
print(enc_out.shape)
print(enc_hidden.shape)
"""
    ),
    md(
        """
## 练习 11.4：补全 BahdanauAttention

目标：实现：

$$
e_{t,i} = \\mathbf{v}_a^\\top \\tanh\\left(\\mathbf{W}_e \\mathbf{h}^{enc}_i + \\mathbf{W}_d \\mathbf{s}_{t-1}\\right)
$$

$$
\\alpha_{t,i} = \\frac{\\exp(e_{t,i})}{\\sum_{j=1}^{S}\\exp(e_{t,j})}
$$

$$
\\mathbf{c}_t = \\sum_{i=1}^{S} \\alpha_{t,i}\\mathbf{h}^{enc}_i
$$

代码实现时对应四步：`score -> mask PAD -> softmax -> bmm`。
"""
    ),
    code(
        """
class BahdanauAttention(nn.Module):
    def __init__(self, enc_output_dim, dec_hidden_dim):
        super().__init__()
        # TODO 1: W_enc, W_dec, v 三个 Linear
        pass

    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        # decoder_hidden: (1,B,H_dec)
        # encoder_outputs: (B,S,2H_enc)
        # src_mask: (B,S), True 表示真实 token
        # TODO 2: 取 decoder_hidden[-1] 并 unsqueeze 成 query
        # TODO 3: 计算 scores: (B,S)
        # TODO 4: mask PAD 位置
        # TODO 5: softmax 得 attn_weights
        # TODO 6: bmm 得 context
        pass


attn = BahdanauAttention(ENC_HIDDEN_DIM * 2, DEC_HIDDEN_DIM)
src_mask = src.ne(PAD)
context, attn_weights = attn(enc_hidden, enc_out, src_mask)
print(context.shape)
print(attn_weights.shape)
print(attn_weights.sum(dim=1)[:5])
"""
    ),
    md(
        """
## 练习 11.5：补全 AttnDecoder + Seq2SeqAttention

目标：把 attention 接入 decoder，并返回 `outputs` 和 `attentions`。
"""
    ),
    code(
        """
class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_output_dim, dec_hidden_dim):
        super().__init__()
        # TODO 1: embedding, attention, gru, fc
        pass

    def forward(self, x, hidden, encoder_outputs, src_mask):
        # TODO 2: embedding
        # TODO 3: attention 得 context
        # TODO 4: concat embedding + context 送 GRU
        # TODO 5: concat output + context + embedding 送 fc
        pass


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        # TODO 6: 参考 lesson，循环解码并保存 outputs/attentions
        pass
"""
    ),
    md(
        """
## 练习 11.6：训练 + 推理 + 热力图

目标：跑通 5 轮训练，并实现 `translate_with_attention`。
"""
    ),
    code(
        """
encoder = Encoder(len(src_vocab), EMBED_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM)
decoder = AttnDecoder(len(tgt_vocab), EMBED_DIM, ENC_HIDDEN_DIM * 2, DEC_HIDDEN_DIM)
model = Seq2SeqAttention(encoder, decoder).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
V_tgt = len(tgt_vocab)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n_steps = 0.0, 0
    for src, src_len, tgt in loader:
        # TODO 1: to(device), zero_grad, forward, loss, backward, clip_grad, step
        pass
    print(f'Epoch {epoch:3d} | loss={total_loss / max(n_steps, 1):.4f}')


def translate_with_attention(model, sentence, max_len=30):
    # TODO 2: greedy decode，同时保存每一步 attn_weights
    pass


for zh, en in raw_pairs[:5]:
    pred, attn_matrix, src_tokens, pred_tokens = translate_with_attention(model, zh)
    print(f'{zh} -> {pred}   (gold: {en})')
"""
    ),
    md(
        """
## 练习 11.7：口头/笔头回答

1. U10 的 Decoder 为什么只需要 `hidden`，U11 为什么还需要 `encoder_outputs`？
2. `attn_weights` 的 shape 是 `(B,S)`，为什么不是 `(B,T,S)`？
3. `Seq2SeqAttention` 返回的 `attentions` 为什么是 `(B,T,S)`？
4. 为什么 attention 里必须 mask PAD？
5. `torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)` 中，两个矩阵乘法输入的 shape 分别是什么？
6. 专业分词器相比中文按字切分，解决了什么问题？
"""
    ),
]


notes = """# U11 学习笔记 | Bahdanau Attention

## 1. 全局图：U10 -> U11 改了哪里

用自己的话画出：

```text
src -> Encoder -> ?
Decoder hidden + ? -> Attention -> ? -> Decoder -> logits
```

---

## 2. 四个核心张量

| 张量 | shape | 我的理解 |
|---|---|---|
| encoder_outputs |  |  |
| decoder_hidden |  |  |
| attn_weights |  |  |
| context |  |  |

---

## 3. Bahdanau Attention 三步

1. 打分：
2. softmax：
3. 加权求和：

---

## 4. 为什么要 mask PAD

- 不 mask 会发生什么：
- mask 的代码是哪一行：

---

## 5. 专业分词器

| 语言 | 分词器 | 为什么不用简单 split/list |
|---|---|---|
| 中文 | jieba |  |
| 英文 | MosesTokenizer |  |

---

## 6. Attention 热力图怎么读

- 横轴：
- 纵轴：
- 颜色：
- 我观察到的一个对齐现象：

---

## 7. 我的疑问 / 自己的总结

"""


def write_nb(path, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nbf.write(nb, path)


write_nb(BASE / "lesson.ipynb", lesson_cells)
write_nb(BASE / "exercises.ipynb", exercise_cells)
(BASE / "notes.md").write_text(notes, encoding="utf-8")
print("generated U11 lesson/exercises/notes")
