# 图解 Transformer 架构：大模型底层原理

> **Python 版** | 基于 PyTorch 实现，适合新手理解大模型底层原理
> 前置知识：[Embedding 模型](../01-第一阶段-Agent基础入门/13_Embedding模型.md)

---

## Transformer 是什么？

Transformer 是 2017 年 Google 论文《Attention Is All You Need》提出的架构，是目前所有大模型（GPT、Claude、Llama、Qwen 等）的基础。

### 核心创新

| 创新点 | 说明 | 优势 |
|--------|------|------|
| **完全基于自注意力机制** | 抛弃了传统的 RNN/LSTM | 解决长距离依赖问题 |
| **高度并行化训练** | 所有位置同时计算 | 训练速度大幅提升 |
| **多头注意力** | 多个注意力头并行工作 | 捕捉不同维度的语义关系 |

### 为什么 Transformer 取代了 RNN？

| 对比维度 | RNN/LSTM | Transformer |
|----------|-----------|-------------|
| **长距离依赖** | 差（信息随距离衰减） | 好（直接计算任意位置关系） |
| **并行训练** | 差（必须按顺序计算） | 好（所有位置同时计算） |
| **训练速度** | 慢 | 快（GPU 充分利用） |
| **模型规模** | 难以扩展到数十亿参数 | 轻松扩展到千亿/万亿参数 |

---

## 整体架构图

```
                         ┌─────────────────────┐
                         │     输出 (Output)    │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   线性层 + Softmax    │
                         │  (预测下一个词)        │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  解码器 N× (Decoder)  │
                         │  ┌─────────────────┐  │
                         │  │ 掩码多头注意力    │  │
                         │  ├─────────────────┤  │
                         │  │ 交叉注意力       │  │←── 关注编码器输出
                         │  ├─────────────────┤  │
                         │  │ 前馈网络         │  │
                         │  └─────────────────┘  │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
    ┌─────────▼─────────┐ ┌───────▼────────┐ ┌──────────▼──────────┐
    │  编码器 N× (Encoder)│ │  编码器输出      │ │  输出嵌入 (Output)   │
    │  ┌───────────────┐ │ │  (Encoder Out)  │ │  + 位置编码           │
    │  │ 多头注意力     │ │ │                 │ └─────────────────────┘
    │  ├───────────────┤ │ └─────────────────┘
    │  │ 前馈网络       │ │
    │  └───────────────┘ │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │  输入嵌入 (Input)   │
    │  + 位置编码          │
    └─────────────────────┘
```

### 架构类型

| 类型 | 说明 | 代表模型 | 适用场景 |
|------|------|----------|----------|
| **Encoder-only** | 只用编码器 | BERT、RoBERTa | 文本理解、分类、命名实体识别 |
| **Decoder-only** | 只用解码器 | GPT、Claude、Llama、Qwen | 文本生成、对话、代码生成 |
| **Encoder-Decoder** | 编码器+解码器 | T5、BART、翻译模型 | 机器翻译、摘要生成 |

**目前主流大模型都是 Decoder-only 架构。**

---

## 核心组件详解

### 1. 位置编码（Positional Encoding）

Self-Attention 本身不考虑顺序（词袋模型），需要位置编码告诉模型词的位置。

```python
"""
positional_encoding.py - 位置编码实现
使用正弦/余弦函数生成位置编码
"""
import numpy as np
import torch


def get_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    生成位置编码矩阵

    Args:
        max_len: 最大序列长度
        d_model: 模型维度（嵌入维度）

    Returns:
        torch.Tensor: 位置编码矩阵，形状 (max_len, d_model)
    """
    # 初始化位置编码矩阵
    pe = np.zeros((max_len, d_model))

    # 位置索引：[0, 1, 2, ..., max_len-1]
    position = np.arange(0, max_len)[:, np.newaxis]

    # 计算分频项：不同维度使用不同的频率
    # 偶数维度用 sin，奇数维度用 cos
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )

    # 偶数维度：sin
    pe[:, 0::2] = np.sin(position * div_term)

    # 奇数维度：cos
    pe[:, 1::2] = np.cos(position * div_term)

    return torch.tensor(pe, dtype=torch.float32)


# 示例：512维，最大长度1000
pe = get_positional_encoding(max_len=1000, d_model=512)
print(f"位置编码形状: {pe.shape}")  # torch.Size([1000, 512])
print(f"第0个位置的前10维: {pe[0, :10]}")
print(f"第1个位置的前10维: {pe[1, :10]}")
```

**位置编码的作用**：
- 让模型知道每个词在句子中的位置
- 不同维度使用不同频率，捕捉不同粒度的位置信息
- 可以外推到训练时没见过的更长序列

### 2. 自注意力机制（Self-Attention）

核心公式：`Attention(Q, K, V) = softmax(QK^T / √d_k) V`

| 符号 | 名称 | 含义 | 类比 |
|------|------|------|------|
| **Q** | Query（查询） | "我在找什么" | 搜索关键词 |
| **K** | Key（键） | "我包含什么" | 网页标题 |
| **V** | Value（值） | "我的实际内容" | 网页内容 |

```python
"""
self_attention.py - 自注意力机制实现
"""
import torch
import torch.nn.functional as F


def self_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple:
    """
    自注意力计算

    Args:
        Q: 查询矩阵，形状 (batch, seq_len, d_k)
        K: 键矩阵，形状 (batch, seq_len, d_k)
        V: 值矩阵，形状 (batch, seq_len, d_v)
        mask: 掩码矩阵，形状 (seq_len, seq_len)，可选

    Returns:
        tuple: (输出, 注意力权重)
    """
    d_k = Q.size(-1)

    # 步骤1：计算 Q·K^T / √d_k
    # (batch, seq_len, d_k) × (batch, d_k, seq_len) = (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )

    # 步骤2：应用掩码（可选）
    # 因果掩码：让每个位置只能看到前面的位置（用于解码器）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 步骤3：Softmax 归一化，得到注意力权重
    attn_weights = F.softmax(scores, dim=-1)

    # 步骤4：注意力权重 × V，得到输出
    # (batch, seq_len, seq_len) × (batch, seq_len, d_v) = (batch, seq_len, d_v)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


# ========== 示例 ==========
if __name__ == "__main__":
    # batch=2, seq_len=10, d_model=512
    Q = torch.randn(2, 10, 512)
    K = torch.randn(2, 10, 512)
    V = torch.randn(2, 10, 512)

    output, weights = self_attention(Q, K, V)
    print(f"输出形状: {output.shape}")    # torch.Size([2, 10, 512])
    print(f"注意力权重形状: {weights.shape}")  # torch.Size([2, 10, 10])

    # 验证注意力权重每行和为1
    print(f"第0个样本第0行权重和: {weights[0, 0].sum().item():.4f}")  # 接近1.0

    # 因果掩码示例（解码器用）
    causal_mask = torch.tril(torch.ones(10, 10))  # 下三角矩阵
    output_masked, weights_masked = self_attention(Q, K, V, mask=causal_mask)
    print(f"\n因果掩码后，第0个位置只能看到自己:")
    print(f"  第0行权重: {weights_masked[0, 0]}")  # 只有第0列非零
```

### 3. 多头注意力（Multi-Head Attention）

把 Q/K/V 分成多个头，每个头学习不同的注意力模式，最后拼接。

```
输入 (d_model=512)
    │
    ├── Q (512) ──→ 分成8个头 ──→ 头1(64) 头2(64) ... 头8(64)
    ├── K (512) ──→ 分成8个头 ──→ 头1(64) 头2(64) ... 头8(64)
    └── V (512) ──→ 分成8个头 ──→ 头1(64) 头2(64) ... 头8(64)
                                    │
                              每个头独立计算自注意力
                                    │
                              拼接 (8×64=512)
                                    │
                              线性变换输出 (512)
```

```python
"""
multi_head_attention.py - 多头注意力实现
"""
import torch
import torch.nn as nn
from self_attention import self_attention


class MultiHeadAttention(nn.Module):
    """多头注意力层"""

    def __init__(self, d_model: int = 512, n_heads: int = 8):
        """
        Args:
            d_model: 模型维度（必须能被 n_heads 整除）
            n_heads: 注意力头数量
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 每个头的维度

        # Q/K/V 的线性变换
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 输出线性变换
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入，形状 (batch, seq_len, d_model)
            mask: 掩码，可选

        Returns:
            torch.Tensor: 输出，形状 (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # 步骤1：线性变换 + 分头
        # (batch, seq_len, d_model) → (batch, seq_len, n_heads, d_head) → (batch, n_heads, seq_len, d_head)
        Q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 步骤2：每个头独立计算自注意力
        attn_out, _ = self_attention(Q, K, V, mask)

        # 步骤3：拼接所有头的输出
        # (batch, n_heads, seq_len, d_head) → (batch, seq_len, n_heads, d_head) → (batch, seq_len, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # 步骤4：输出线性变换
        return self.W_O(attn_out)


# 示例
if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=512, n_heads=8)
    x = torch.randn(2, 10, 512)
    output = mha(x)
    print(f"输入形状: {x.shape}")       # torch.Size([2, 10, 512])
    print(f"输出形状: {output.shape}")  # torch.Size([2, 10, 512])
    print(f"参数数量: {sum(p.numel() for p in mha.parameters())}")
```

### 4. 前馈网络（Feed Forward Network）

每个注意力层后接一个两层的全连接网络，增加非线性表达能力。

```python
"""
feed_forward.py - 前馈网络实现
"""
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """前馈网络层（Position-wise Feed-Forward Networks）"""

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度（通常是 d_model 的4倍）
            dropout: Dropout 比率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：Linear → ReLU → Dropout → Linear

        公式：FFN(x) = max(0, xW1 + b1)W2 + b2
        """
        x = self.linear1(x)   # (batch, seq_len, d_ff)
        x = self.relu(x)       # 激活函数
        x = self.dropout(x)    # 正则化
        x = self.linear2(x)    # (batch, seq_len, d_model)
        return x


# 示例
if __name__ == "__main__":
    ff = FeedForward(d_model=512, d_ff=2048)
    x = torch.randn(2, 10, 512)
    output = ff(x)
    print(f"输入形状: {x.shape}")       # torch.Size([2, 10, 512])
    print(f"输出形状: {output.shape}")  # torch.Size([2, 10, 512])
```

### 5. 残差连接 + LayerNorm

每个子层都有残差连接和层归一化，解决深层网络训练问题（梯度消失/爆炸）。

```python
"""
encoder_layer.py - 编码器层完整实现
包含：多头注意力 + 残差 + LayerNorm + 前馈网络 + 残差 + LayerNorm
"""
import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """Transformer 编码器层"""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        # 子层1：多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 子层2：前馈网络
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        结构：
        x → 自注意力 → Dropout → 残差 + LayerNorm → 前馈 → Dropout → 残差 + LayerNorm → 输出
        """
        # 子层1：自注意力 + 残差 + 归一化
        # 公式：x = LayerNorm(x + Dropout(SelfAttention(x)))
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 子层2：前馈 + 残差 + 归一化
        # 公式：x = LayerNorm(x + Dropout(FFN(x)))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


# 完整编码器
class Encoder(nn.Module):
    """Transformer 编码器（N 层编码器层堆叠）"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_layers: int = 6,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        from positional_encoding import get_positional_encoding
        self.register_buffer("pos_encoding", get_positional_encoding(max_len, d_model))

        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 token IDs，形状 (batch, seq_len)
            mask: 掩码，可选

        Returns:
            torch.Tensor: 编码输出，形状 (batch, seq_len, d_model)
        """
        seq_len = x.size(1)

        # 词嵌入 + 位置编码
        x = self.embedding(x) + self.pos_encoding[:seq_len, :]
        x = self.dropout(x)

        # 通过 N 层编码器
        for layer in self.layers:
            x = layer(x, mask)

        return x


# 示例
if __name__ == "__main__":
    # 编码器层
    encoder_layer = EncoderLayer(d_model=512, n_heads=8, d_ff=2048)
    x = torch.randn(2, 10, 512)
    output = encoder_layer(x)
    print(f"编码器层输入: {x.shape}")
    print(f"编码器层输出: {output.shape}")

    # 完整编码器
    encoder = Encoder(vocab_size=10000, d_model=512, n_heads=8, d_ff=2048, n_layers=6)
    tokens = torch.randint(0, 10000, (2, 10))
    encoded = encoder(tokens)
    print(f"\n完整编码器输入: {tokens.shape}")
    print(f"完整编码器输出: {encoded.shape}")
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters()):,}")
```

---

## 编码器 vs 解码器

| 组件 | 编码器 (Encoder) | 解码器 (Decoder) |
|------|-----------------|-----------------|
| **自注意力** | ✅ 可以看到所有位置 | ✅ 只能看到前面的位置（因果掩码） |
| **交叉注意力** | ❌ | ✅ 关注编码器输出 |
| **前馈网络** | ✅ | ✅ |
| **残差+LayerNorm** | ✅ | ✅ |
| **用途** | 理解输入 | 生成输出 |

### 三种架构类型

| 类型 | 说明 | 代表模型 | 适用场景 |
|------|------|----------|----------|
| **Encoder-only** | 只用编码器 | BERT、RoBERTa、ALBERT | 文本理解、分类、NER、情感分析 |
| **Decoder-only** | 只用解码器 | GPT-3/4、Claude、Llama、Qwen、Mistral | 文本生成、对话、代码、写作 |
| **Encoder-Decoder** | 编码器+解码器 | T5、BART、mBART、翻译模型 | 机器翻译、摘要、问答生成 |

**目前主流大模型都是 Decoder-only 架构，用因果掩码实现自回归生成。**

---

## 学习要点

1. **Transformer 核心是 Self-Attention**，解决长距离依赖和并行训练问题，取代了 RNN/LSTM
2. **Q/K/V 是自注意力的三个核心矩阵**：Q（查询，我在找什么）、K（键，我包含什么）、V（值，我的实际内容）
3. **注意力公式**：`Attention(Q, K, V) = softmax(QK^T / √d_k) V`，除以 √d_k 防止梯度消失
4. **多头注意力**让模型同时关注不同子空间的信息，每个头学习不同的注意力模式
5. **位置编码**让模型知道词的顺序，使用正弦/余弦函数，可以外推到更长序列
6. **前馈网络**增加非线性表达能力，通常是两层全连接，隐藏层维度是模型维度的4倍
7. **残差连接 + LayerNorm**让深层网络可训练，解决梯度消失/爆炸问题
8. **因果掩码**让解码器每个位置只能看到前面的位置，实现自回归生成（一个词一个词生成）
9. **三种架构**：Encoder-only（BERT，理解）、Decoder-only（GPT，生成）、Encoder-Decoder（T5，翻译）
10. **目前主流大模型都是 Decoder-only**，因为生成能力更强，训练和推理更简单

## 扩展方向

- 学习 Transformer 的变体（Linformer、Performer、Longformer、Flash Attention）
- 探索注意力机制的优化（稀疏注意力、线性注意力、滑动窗口注意力）
- 学习位置编码的改进（RoPE 旋转位置编码、ALiBi、NoPE）
- 探索归一化方法（LayerNorm、RMSNorm、Pre-LN vs Post-LN）
- 学习激活函数（ReLU、GELU、SiLU、SwiGLU）
- 探索大模型架构（GPT 系列、Llama 系列、Qwen 系列、Mistral 系列）
- 学习大模型训练（预训练、监督微调 SFT、RLHF、DPO）
- 探索大模型推理优化（KV Cache、量化、投机解码、连续批处理）
- 学习 MoE（混合专家）架构（Switch Transformer、GShard、DeepSeek MoE）
- 探索多模态 Transformer（CLIP、BLIP、LLaVA、Qwen-VL）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/04-storage-monitoring/42-transformer-architecture

包含本文的完整可运行代码示例（位置编码+自注意力+多头注意力+前馈网络+编码器层+完整编码器）。

---

**上一篇**：[LangFuse 可观测平台](./41_LangFuse.md) | **下一篇**：[大模型训练推理全流程](./43_大模型训练推理全流程.md)
