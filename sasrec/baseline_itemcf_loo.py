# -*- coding: utf-8 -*-
"""同口径 ItemCF 基线（leave-one-out）

为什么要有这个：
- 现有 itemcf_recall.py 用 80/20 随机交互划分（简历数字来源）
- SASRec 必须按时间切（否则信息泄漏），用 leave-one-out
- 两种口径不能直接比，所以在 LOO 切分下重跑 ItemCF，做公平对照

--recent N：只用每个用户最近 N 个行为建相似度矩阵（与 SASRec 的窗口对齐），
用来判断"最近 N 个行为"本身包含多少信息。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dataset import load, load_from_args

K, NDCG_K = 20, 10


def ndcg_at_k(rel, k):
    rel = rel[:k]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recent', type=int, default=0,
                   help='只用每个用户最近 N 个训练行为建矩阵（0=全用）')
    p.add_argument('--min_rating', type=int, default=4,
                   help='4=只用 4 分以上；0=用全部评分')
    p.add_argument('--data', type=str, default='')
    p.add_argument('--max_users', type=int, default=0)
    a = p.parse_args()

    uids, train, valid, test, n_items = load_from_args(a)
    n_users = len(train)
    train_use = [s[-a.recent:] if a.recent > 0 else s for s in train]

    # 用 train_use 建用户-物品矩阵（第 0 列留给 padding）
    R = np.zeros((n_users, n_items), dtype=np.float32)
    for u in range(n_users):
        for it in train_use[u]:
            R[u, it] = 1.0

    # ItemCF：item-item 余弦相似度
    norm = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-8)
    S = norm.T @ norm
    np.fill_diagonal(S, 0)

    # 打分 = 用户历史 item 与候选 item 的相似度和
    scores = R @ S

    hits, ndcg_sum, n = 0, 0.0, 0
    for u in range(n_users):
        row = scores[u].copy()
        row[0] = -np.inf
        for it in train[u]:          # 排除全部训练交互（不只最近 N 个）
            row[it] = -np.inf
        top = np.argpartition(row, -K)[-K:]
        top = top[np.argsort(row[top])[::-1]]
        hits += int(test[u] in top)
        rel = np.array([1 if j == test[u] else 0 for j in top])
        ndcg_sum += ndcg_at_k(rel, NDCG_K)
        n += 1

    note = f'最近{a.recent}个行为' if a.recent > 0 else '全部历史'
    print(f'===== ItemCF 基线（leave-one-out｜{note}）=====')
    print(f'Recall@20={hits / n:.4f}  NDCG@10={ndcg_sum / n:.4f}  （用户 {n}）')

    tag = f'_recent{a.recent}' if a.recent > 0 else ''
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'baseline_itemcf{tag}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'test_recall20': round(hits / n, 4),
                   'test_ndcg10': round(ndcg_sum / n, 4),
                   'n_users': n, 'recent': a.recent}, f, ensure_ascii=False, indent=2)
    print(f'结果已写入 {out}')


if __name__ == '__main__':
    main()
