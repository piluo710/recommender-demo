# -*- coding: utf-8 -*-
"""诊断：输出层点积是否被 item embedding 模长主导

SASRec 输出 logits = x @ item_emb.weight.T
  x 经过 LayerNorm，模长固定 ≈ sqrt(d)
  item_emb.weight 的模长完全自由
=> logits ≈ |x| * |w_i| * cos(x, w_i)
   如果 |w_i| 的跨 item 方差远大于 cos 的方差，排序就由"item 向量有多长"决定，
   而模长大的 item 往往是高频 item -> 模型退化成按流行度排序。

这个脚本验证：
1. |w_i| 与 item 流行度的相关性
2. 各用户 logits 的排序，有多少信息来自模长、多少来自方向
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

import numpy as np
import torch

from dataset import default_min_rating_from_meta, pad_seq
from model import SASRec

MAX_LEN = 50


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min_rating', type=int, default=default_min_rating_from_meta(),
                   help='必须与 best.pt 的训练口径一致，否则物品数对不上')
    p.add_argument('--data', type=str, default='')
    p.add_argument('--max_users', type=int, default=0)
    a = p.parse_args()

    from dataset import load_from_args
    uids, train, valid, test, n_items = load_from_args(a)
    model = SASRec(n_items, MAX_LEN, 64, 2, 2, 0.0)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'best.pt'), weights_only=True))
    model.eval()

    W = model.item_emb.weight.detach().cpu().numpy()      # [V, d]
    mod = np.linalg.norm(W, axis=1)                       # [V] 模长
    counts = np.zeros(n_items)
    for s in train:
        for it in s:
            counts[it] += 1
    counts[0] = 0

    # 1) 模长 vs 流行度（Spearman，不假设线性）
    idx = np.arange(1, n_items)
    rho = np.corrcoef(np.argsort(np.argsort(mod[idx])),
                      np.argsort(np.argsort(counts[idx])))[0, 1]
    print(f'item embedding 模长 与 流行度 的 Spearman 相关：{rho:.3f}')

    top_mod = idx[np.argsort(-mod[idx])[:10]]
    top_pop = idx[np.argsort(-counts[idx])[:10]]
    print(f'模长最大的 10 个 item：{sorted(top_mod.tolist())}')
    print(f'最热门的 10 个 item  ：{sorted(top_pop.tolist())}')
    print(f'两者重合：{len(set(top_mod.tolist()) & set(top_pop.tolist()))} / 10')

    # 2) 模长项 vs 方向项的方差对比
    sample = np.random.default_rng(0).choice(len(train), 50, replace=False)
    xs = []
    for u in sample:
        seq = train[u] + [valid[u]]
        x = model(torch.tensor([pad_seq(seq, MAX_LEN)]))  # 只在最后位置有用
        # 重新取 last_ln 之后的表示：用 hook 不方便，这里用近似——
        # 直接比较不同用户的 logits 排序差异（跨用户标准差）
        xs.append(x[0, -1, :].numpy())
    xs = np.array(xs)                                     # [50, V]
    print(f'\n不同用户 logits 的标准差（逐 item 平均）：{xs.std(axis=0).mean():.3f}')
    print(f'单用户 logits 的标准差（逐用户平均）：{xs.std(axis=1).mean():.3f}')
    print('说明：若前者远小于后者，说明用户间几乎没有差异（都在按同一套偏好排序）')


if __name__ == '__main__':
    main()
