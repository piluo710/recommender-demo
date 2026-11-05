# -*- coding: utf-8 -*-
"""融合实验：ItemCF（最近50个行为）+ SASRec

目的：判断序列模型有没有"增量信息"——如果融合能超过 ItemCF 单路，
说明 SASRec 补到了 CF 抓不到的东西（面试可讲的重点）。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dataset import load, pad_seq
from model import SASRec

K, NDCG_K, MAX_LEN = 20, 10, 50
WS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
HERE = os.path.dirname(os.path.abspath(__file__))


def ndcg_at_k(rel, k):
    rel = rel[:k]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def sasrec_scores(train, valid, n_items):
    model = SASRec(n_items, MAX_LEN, 64, 2, 2, 0.0)
    model.load_state_dict(torch.load(os.path.join(HERE, 'best.pt'), weights_only=True))
    model.eval()
    out = np.zeros((len(train), n_items), dtype=np.float32)
    B = 128
    for start in range(0, len(train), B):
        end = min(start + B, len(train))
        seqs = torch.tensor([pad_seq(train[u] + [valid[u]], MAX_LEN) for u in range(start, end)])
        s = model(seqs)[:, -1, :].cpu().numpy()
        out[start:end] = np.nan_to_num(s, nan=-1e9)
    return out


def main():
    uids, train, valid, test, n_items = load()
    n_users = len(train)

    # --- 1. ItemCF（最近 50 个行为）---
    R = np.zeros((n_users, n_items), dtype=np.float32)
    for u in range(n_users):
        for it in train[u][-50:]:
            R[u, it] = 1.0
    norm = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-8)
    S = norm.T @ norm
    np.fill_diagonal(S, 0)
    cf = R @ S

    # --- 2. SASRec 全池分数 ---
    sr = sasrec_scores(train, valid, n_items)

    # 各自 min-max 归一到 [0,1]（逐用户，保证两路尺度可比）
    def user_norm(m):
        lo = m.min(axis=1, keepdims=True)
        hi = m.max(axis=1, keepdims=True)
        return (m - lo) / (hi - lo + 1e-8)

    # 注意顺序：先归一化，再设排除项为 -1e9。
    # 反过来做的话 -1e9 会把 min 拉成 -1e9，所有候选被压到 ≈1，排序直接失真。
    cf_n, sr_n = user_norm(cf), user_norm(sr)
    for u in range(n_users):
        idx = list(set(train[u]) | {valid[u]})
        cf_n[u, idx] = -1e9
        sr_n[u, idx] = -1e9
        cf_n[u, 0] = -1e9
        sr_n[u, 0] = -1e9

    rows = []
    for w in WS:
        score = (1 - w) * cf_n + w * sr_n
        hits, ndcg_sum = 0, 0.0
        for u in range(n_users):
            top = np.argpartition(-score[u], K)[:K]
            top = top[np.argsort(-score[u][top])]
            hits += int(test[u] in top)
            rel = np.array([1 if j == test[u] else 0 for j in top])
            ndcg_sum += ndcg_at_k(rel, NDCG_K)
        r, n_ = hits / n_users, ndcg_sum / n_users
        rows.append({'w_sasrec': w, 'recall20': round(r, 4), 'ndcg10': round(n_, 4)})
        print(f'w(SASRec)={w:.1f}  Recall@20={r:.4f}  NDCG@10={n_:.4f}')

    best = max(rows, key=lambda x: x['recall20'])
    print(f'\n最优权重 w={best["w_sasrec"]}，Recall@20={best["recall20"]}')
    base = [r for r in rows if r['w_sasrec'] == 0.0][0]['recall20']
    if best['w_sasrec'] > 0:
        print(f'相对纯 ItemCF 提升 {(best["recall20"] / base - 1) * 100:.1f}%')
    else:
        print('融合未超过纯 ItemCF，说明 SASRec 增量信息不足')

    with open(os.path.join(HERE, 'fusion_result.json'), 'w', encoding='utf-8') as f:
        json.dump({'sweep': rows, 'note': 'ItemCF只用最近50个行为'}, f,
                  ensure_ascii=False, indent=2)
    print('结果已写入 fusion_result.json')


if __name__ == '__main__':
    main()
