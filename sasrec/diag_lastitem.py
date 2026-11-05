# -*- coding: utf-8 -*-
"""诊断：跳过 attention，直接用"最后一个 item 的 embedding"做相似度打分

目的：把"attention 有没有把信号稀释"和"输出层/评估本身有问题"两件事分开。
- 若本诊断 R@20 明显高于 SASRec 的 0.07（例如 0.10+），说明 attention 在稀释信号
- 若本诊断也约 0.07，说明问题不在 attention，而在输出层尺度/数据本身

跑法：
    & python sasrec\diag_lastitem.py --min_rating 0 --data <ratings.dat>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dataset import pad_seq, load
from model import SASRec

HERE = os.path.dirname(os.path.abspath(__file__))
K = 20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--min_rating', type=int, default=0)
    p.add_argument('--data', type=str, default='')
    p.add_argument('--max_users', type=int, default=0)
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--heads', type=int, default=2)
    return p.parse_args()


def main():
    a = parse_args()
    kw = {'min_rating': a.min_rating}
    if a.data:
        kw['data_path'] = a.data
    if a.max_users:
        kw['max_users'] = a.max_users
    uids, train, valid, test, n_items = load(**kw)

    model = SASRec(n_items, a.max_len, a.d, a.blocks, a.heads)
    ckpt = os.path.join(HERE, 'best.pt')
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    item_emb = model.item_emb.weight.detach()       # [V, d]

    hits, n = 0, 0
    for u in range(len(train)):
        hist = train[u] + [valid[u]]
        last = hist[-1]
        scores = (item_emb @ item_emb[last]).clone()   # 只看"最后一个 item"的相似度
        scores[0] = -1e9
        idx = torch.tensor(sorted(set(hist)), dtype=torch.long)
        scores[idx] = -1e9
        top = torch.topk(scores, K).indices.numpy()
        hits += int(test[u] in top)
        n += 1
    print(f'\n===== 只取最后一个 item 的相似度（跳过 attention）=====')
    print(f'Recall@20={hits / n:.4f}  （对照：SASRec 完整模型约 0.07，ItemCF 最近50 约 0.13–0.16）')


if __name__ == '__main__':
    main()
