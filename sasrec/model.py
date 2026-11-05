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
import torch.nn.functional as F


class Block(nn.Module):
    """pre-LN 的 Transformer block

    残差缩放（res_scale）是必需的：
    纯 pre-LN 在残差流上没有尺度约束，小数据上每层增量方向会趋同，
    残差流模长从 62 涨到 596/1208，用户自己的物品信号被淹没
    -> last_ln 后用户表示两两余弦 0.995，模型退化成输出全局偏置。
    乘一个小系数把残差流锁住（GPT-2 的 residual scaling 同理）。
    """

    def __init__(self, d, n_heads, dropout, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d * 4, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.ln1(x)
        out, _ = self.attn(h, h, h, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask, need_weights=False)
        # padding 位置的 query 在可见范围被全屏蔽时 softmax 分母为 0 -> NaN，先清零
        out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        # 残差基准取归一化后的 h —— 这是我们的自定义改造，**偏离标准 pre-LN**。
        # 标准 pre-LN 是 x = x + attn(ln(x))（残差基于原始 x）。
        # 我们这样写是为了把残差流模长锁在 LayerNorm 输出量级（约 sqrt(d)=8）。
        # 若不锁，残差流逐层累加、模长从 62 涨到 1208，用户自己的物品信号被淹没
        # -> 表示坍缩成全局偏置。改动有实测依据，但不声称与任何官方实现一致。
        x = h + self.drop(out) * self.res_scale
        h = self.ln2(x)
        x = x + self.drop(self.ffn(h)) * self.res_scale
        # 每层结束整体清零 padding 位置，避免零向量带着位置信号扩散到下一层
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class SASRec(nn.Module):
    def __init__(self, n_items, max_len=50, d=64, n_blocks=2, n_heads=2, dropout=0.2,
                 pos_scale=1.0, res_scale=1.0):
        super().__init__()
        self.d = d
        self.pos_scale = pos_scale   # 位置编码权重（0 表示完全关闭）
        self.item_emb = nn.Embedding(n_items, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads, dropout, res_scale) for _ in range(n_blocks)]
        )
        self.last_ln = nn.LayerNorm(d)
        self.max_len = max_len
        self.logit_scale = 20.0   # 余弦相似度模式下的温度（可调）
        # 输出层模式必须是模型的一部分，不能靠外部动态注入：
        # 否则任何人 load 权重后都会默认走点积分支，推理结果静默错误、且不报错。
        # 训练脚本会把它和 max_len/d 一起写进 best_meta.json，加载方据此还原。
        self.norm_out = False

    def encode(self, seq):
        """返回序列表示 [B, L, d]（last_ln 之后），便于诊断/取特征"""
        B, L = seq.shape
        # pos_emb 只有 max_len 行：L 超过 max_len 会静默越界或报 index 错，
        # 这里提前抛出，避免"推理长度和训练长度不一致"这类问题再次悄悄发生
        if L > self.pos_emb.num_embeddings:
            raise ValueError(
                f'输入长度 {L} 超过位置编码容量 {self.pos_emb.num_embeddings}；'
                f'训练与推理必须使用同一个 max_len')
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        # 论文标准做法：item embedding 乘 sqrt(d) 后再叠加位置编码，
        # 否则位置信号会淹没物品信号，模型只学到"位置模式"而不是物品偏好
        #
        # 关键：position embedding 只加在真实位置上。
        # padding 位置 item_emb(0)=0，若再加位置编码，这一层就变成"纯位置信号"，
        # 经残差多轮携带后会成为所有用户共享的主导方向，
        # 导致 last_ln 之后用户表示两两余弦 0.996（表示坍缩）、排序退化成全局偏置。
        real = (seq != 0).unsqueeze(-1).to(self.pos_emb.weight.dtype)
        x = self.emb_drop(self.item_emb(seq) * (self.d ** 0.5)
                          + self.pos_emb(pos) * real * self.pos_scale)

        # 上三角 True = 不允许 attend（位置 t 只能看 <= t）
        attn_mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool),
                               diagonal=1)
        key_padding_mask = (seq == 0)   # True = 屏蔽 padding 位置

        for blk in self.blocks:
            x = blk(x, attn_mask, key_padding_mask)
        x = self.last_ln(x)
        return torch.nan_to_num(x, nan=0.0)   # 双保险：极端输入也不吐 NaN

    def forward(self, seq, norm_out=False):
        """seq: [B, L] item id（0=padding）-> logits [B, L, V]

        norm_out=True 时用余弦相似度 + 固定温度：
        原始写法 x @ item_emb.T 的点积尺度可达 40-60，softmax 温度极低、
        接近 one-hot，梯度只作用在最高频的几个物品上（loss 卡在 13+，学不动）。
        归一化后尺度约 20，softmax 才有合理的梯度分布。
        """
        x = self.encode(seq)
        if norm_out:
            x = F.normalize(x, dim=-1)
            w = F.normalize(self.item_emb.weight, dim=-1)
            return (x @ w.T) * self.logit_scale
        # 权重共享：直接和 item embedding 转置做点积
        return x @ self.item_emb.weight.T


def build_from_meta(n_items, meta, ckpt_path=None):
    """按 best_meta.json 里记录的配置重建模型（并可顺带加载权重）。

    为什么所有脚本都必须走这个入口：
    norm_out / logit_scale / pos_scale / res_scale / max_len / d / blocks / heads
    都不在 state_dict 里。只要构造参数和训练时不一致，推理就会**静默出错** ——
    不抛异常、不报警，只是分数全错。之前 diag_what 用命令行默认值重建模型，
    就是这类隐患的定时炸弹。
    """
    m = SASRec(n_items,
               max_len=int(meta.get('max_len', 50)),
               d=int(meta.get('d', 64)),
               n_blocks=int(meta.get('blocks', 2)),
               n_heads=int(meta.get('heads', 2)),
               pos_scale=float(meta.get('pos_scale', 1.0)),
               res_scale=float(meta.get('res_scale', 1.0)))
    m.norm_out = bool(meta.get('norm_out', False))
    m.logit_scale = float(meta.get('logit_scale', 20.0))
    if ckpt_path is not None:
        m.load_state_dict(torch.load(ckpt_path, map_location='cpu',
                                     weights_only=True))
    return m
