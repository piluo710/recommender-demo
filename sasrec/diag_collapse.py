# -*- coding: utf-8 -*-
"""诊断：模型输出是否塌缩到"全局热门"

做法：随机取 30 个用户，各自算 top-20，统计两两平均重合数量。
- 如果平均重合接近 20，说明所有用户拿到几乎相同的推荐 -> 模型只学到流行度
- 如果明显低于 20，说明确实有个性化区分
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dataset import load, pad_seq
from model import SASRec

K, MAX_LEN = 20, 50


@torch.no_grad()
def main():
    uids, train, valid, test, n_items = load()
    model = SASRec(n_items, MAX_LEN, 64, 2, 2, 0.0)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'best.pt'), weights_only=True))
    model.eval()

    rng = np.random.default_rng(0)
    sample = rng.choice(len(train), 30, replace=False)
    lists = []
    for u in sample:
        seq = train[u] + [valid[u]]
        s = model(torch.tensor([pad_seq(seq, MAX_LEN)]))[0, -1, :]
        s = torch.nan_to_num(s, nan=-1e9)
        s[0] = -1e9
        for it in set(train[u]) | {valid[u]}:
            s[it] = -1e9
        lists.append(set(torch.topk(s, K).indices.tolist()))

    overlaps = []
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            overlaps.append(len(lists[i] & lists[j]))
    print(f'用户两两 top20 平均重合：{np.mean(overlaps):.1f} / 20'
          f'（最小 {min(overlaps)}, 最大 {max(overlaps)}）')

    # 与"全局最热 20"的重合
    counts = np.zeros(n_items)
    for s in train:
        for it in s:
            counts[it] += 1
    counts[0] = 0
    hot = set(np.argsort(-counts)[:K].tolist())
    print(f'与全局最热 20 的平均重合：{np.mean([len(l & hot) for l in lists]):.1f} / 20')

    print('\n前 3 个用户的 top5：')
    for u in sample[:3]:
        seq = train[u] + [valid[u]]
        s = model(torch.tensor([pad_seq(seq, MAX_LEN)]))[0, -1, :]
        s = torch.nan_to_num(s, nan=-1e9)
        s[0] = -1e9
        for it in set(train[u]) | {valid[u]}:
            s[it] = -1e9
        top = torch.topk(s, 5)
        print(f'  user{u} target={test[u]} top5={[(int(i), round(float(v), 2)) for i, v in zip(top.indices, top.values)]}')


if __name__ == '__main__':
    main()
