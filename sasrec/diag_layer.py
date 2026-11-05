# -*- coding: utf-8 -*-
"""诊断：坍缩发生在哪一层

逐层打印 6 个用户（序列长度差异大）在最后位置的表示，
计算两两余弦相似度，定位坍缩点：
  输入层 -> block1 -> block2 -> last_ln 之后
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dataset import load_from_args, pad_seq
from model import SASRec

MAX_LEN = 50


def cos_of(reps):
    reps = np.asarray(reps)
    n = reps / (np.linalg.norm(reps, axis=1, keepdims=True) + 1e-8)
    c = n @ n.T
    off = c[~np.eye(len(c), dtype=bool)]
    return off.mean(), off.min(), off.max()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min_rating', type=int, default=4)
    p.add_argument('--data', type=str, default='')
    p.add_argument('--max_users', type=int, default=0)
    a = p.parse_args()

    uids, train, valid, test, n_items = load_from_args(a)
    model = SASRec(n_items, MAX_LEN, 64, 2, 2, 0.0)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'best.pt'), weights_only=True))
    model.eval()

    # 选 6 个序列长度差异大的用户
    lens = [len(s) for s in train]
    order = np.argsort(lens)
    pick = [order[0], order[100], order[len(order) // 2],
            order[-100], order[-10], order[-1]]
    seqs = [train[u] + [valid[u]] for u in pick]
    print('选中用户序列长度：', [len(s) for s in seqs])

    batch = torch.tensor([pad_seq(s, MAX_LEN) for s in seqs])
    B, L = batch.shape

    stages = {}

    # 逐段复算
    pos = torch.arange(L).unsqueeze(0).expand(B, L)
    real = (batch != 0).unsqueeze(-1).float()
    x = model.item_emb(batch) * (model.d ** 0.5) + model.pos_emb(pos) * real
    stages['0 输入层(item+pos)'] = x
    stages['0b 仅 item_emb'] = model.item_emb(batch) * (model.d ** 0.5)

    attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    kpm = (batch == 0)

    for i, blk in enumerate(model.blocks):
        x = blk(x, attn_mask, kpm)
        stages[f'{i + 1} block{i + 1} 输出'] = x

    x = model.last_ln(x)
    stages['last_ln 之后'] = x

    for name, t in stages.items():
        rep = t[:, -1, :].numpy()
        m, lo, hi = cos_of(rep)
        print(f'{name:22s} 两两余弦 均值={m:.4f}  最小={lo:.4f}  最大={hi:.4f}  '
              f'模长均值={np.linalg.norm(rep, axis=1).mean():.1f}')

    # 最后一层 logits 差异
    xp = stages['last_ln 之后']
    logits = xp[:, -1, :] @ model.item_emb.weight.T
    lg = logits.numpy()
    print(f'\nlogits 跨用户标准差（逐 item 平均）：{lg.std(axis=0).mean():.3f}')
    print(f'logits 单用户标准差（逐用户平均）：{lg.std(axis=1).mean():.3f}')


if __name__ == '__main__':
    main()
