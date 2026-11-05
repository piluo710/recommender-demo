# -*- coding: utf-8 -*-
"""SASRec 模型：Self-Attentive Sequential Recommendation（简化版）

结构（面试口径）：
  输入序列 [u1..u_{t-1}]（item embedding + position embedding）
  → 2 层 Transformer block（pre-LN + causal self-attention + FFN + 残差）
  → 输出层直接和 item embedding 矩阵做点积（权重共享，参数省一半）
  → 每个位置输出"下一个 item"的 logits

关键约束：
- causal mask：位置 t 只能看到 <= t 的历史，上三角 mask 掉未来
- padding mask：补 0 的位置不参与 attention，也不产生 loss
"""

import torch
import torch.nn as nn


class Block(nn.Module):
    """pre-LN 的 Transformer block（比 post-LN 训练更稳）"""

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d * 4, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        # 关键：padding 位置的 query 在可见范围被全屏蔽时 softmax 分母为 0 -> NaN。
        # 必须在这里清零切断 NaN 传播，否则下一层的有效位置会通过 value 被污染。
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = x + self.drop(h)
        h = self.ln2(x)
        x = x + self.drop(self.ffn(h))
        return x


class SASRec(nn.Module):
    def __init__(self, n_items, max_len=50, d=64, n_blocks=2, n_heads=2, dropout=0.2):
        super().__init__()
        self.d = d
        self.item_emb = nn.Embedding(n_items, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.last_ln = nn.LayerNorm(d)
        self.max_len = max_len

    def forward(self, seq):
        """seq: [B, L] item id（0=padding）-> logits [B, L, V]"""
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        # 论文标准做法：item embedding 乘 sqrt(d) 后再叠加位置编码，
        # 否则位置信号会淹没物品信号，模型只学到"位置模式"而不是物品偏好
        x = self.emb_drop(self.item_emb(seq) * (self.d ** 0.5) + self.pos_emb(pos))

        # 上三角 True = 不允许 attend（位置 t 只能看 <= t）
        attn_mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool),
                               diagonal=1)
        key_padding_mask = (seq == 0)   # True = 屏蔽 padding 位置

        for blk in self.blocks:
            x = blk(x, attn_mask, key_padding_mask)
        x = self.last_ln(x)
        x = torch.nan_to_num(x, nan=0.0)   # 双保险：极端输入也不吐 NaN
        # 权重共享：直接和 item embedding 转置做点积
        return x @ self.item_emb.weight.T
