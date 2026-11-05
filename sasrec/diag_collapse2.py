# -*- coding: utf-8 -*-
"""诊断：表示坍缩

1. item embedding 两两余弦相似度的平均值：接近 1 -> 所有 item 向量几乎同向（坍缩）
2. 不同用户的序列表示余弦相似度：接近 1 -> 注意力没学到用户差异（用户表示坍缩）
3. 用户表示与 item embedding 的相似度分布
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dataset import load_from_args
from model import SASRec

MAX_LEN = 50


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

    W = model.item_emb.weight.detach().cpu().numpy()[1:]     # 去掉 padding
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)

    # 1) item embedding 平均两两余弦
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Wn), 1000, replace=False)
    sub = Wn[idx]
    cos = sub @ sub.T
    off = cos[~np.eye(len(sub), dtype=bool)]
    print(f'item embedding 两两余弦相似度：均值 {off.mean():.3f}，'
          f'中位数 {np.median(off):.3f}，标准差 {off.std():.3f}')

    # 2) 不同用户的序列表示余弦相似度
    from dataset import pad_seq
    users = rng.choice(len(train), 100, replace=False)
    reps = []
    for u in users:
        seq = torch.tensor([pad_seq(train[u] + [valid[u]], MAX_LEN)])
        reps.append(model.encode(seq)[0, -1].numpy())
    reps = np.array(reps)
    reps_n = reps / (np.linalg.norm(reps, axis=1, keepdims=True) + 1e-8)
    cos_u = reps_n @ reps_n.T
    off_u = cos_u[~np.eye(len(cos_u), dtype=bool)]
    print(f'\n用户表示两两余弦相似度：均值 {off_u.mean():.4f}，'
          f'最小 {off_u.min():.4f}，最大 {off_u.max():.4f}')
    print('  接近 1 说明不同用户的表示几乎是同一个向量（注意力没学到差异）')

    # 3) 用户表示自身的分散度
    print(f'\n用户表示的逐维标准差（跨用户平均）：{reps.std(axis=0).mean():.4f}')
    print(f'用户表示的模长（均值）：{np.linalg.norm(reps, axis=1).mean():.2f}')
    print(f'item embedding 模长（均值/中位数）：{np.linalg.norm(W, axis=1).mean():.2f} / '
          f'{np.median(np.linalg.norm(W, axis=1)):.2f}')


if __name__ == '__main__':
    main()
