# -*- coding: utf-8 -*-
"""校准基线：纯流行度推荐（全局最热门的 20 个，排除已交互）

用途：判断 LOO 任务里"流行度信号"有多强。
如果纯流行度就能到 0.06-0.08，说明 SASRec 的 0.067 只是学到了热度。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dataset import load

K, NDCG_K = 20, 10


def ndcg_at_k(rel, k):
    rel = rel[:k]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    uids, train, valid, test, n_items = load()

    # 全局流行度
    counts = np.zeros(n_items, dtype=np.int64)
    for s in train:
        for it in s:
            counts[it] += 1
    counts[0] = 0                                    # padding
    pop_order = np.argsort(-counts)

    hits, ndcg_sum, n = 0, 0.0, 0
    for u in range(len(train)):
        seen = set(train[u])
        cand = [it for it in pop_order if it not in seen][:K]
        hits += int(test[u] in cand)
        rel = np.array([1 if j == test[u] else 0 for j in cand])
        ndcg_sum += ndcg_at_k(rel, NDCG_K)
        n += 1

    print('===== 纯流行度基线（leave-one-out）=====')
    print(f'Recall@20={hits / n:.4f}  NDCG@10={ndcg_sum / n:.4f}  （用户 {n}）')


if __name__ == '__main__':
    main()
