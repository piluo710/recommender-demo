# -*- coding: utf-8 -*-
"""独立复算：用与 train.py 完全不同的实现路径重算 Recall@20 / NDCG@10

为什么必须有这个脚本：
train.py 的 evaluate 本身是被怀疑的对象。用同一个函数再跑一遍，什么也证明不了 ——
同一个 bug 会稳定地给出同一个错数。所以这里换三条互不依赖的实现路径：

  1. 排序：不用 torch.topk，改用 numpy.argsort 手工全排序
  2. 排除：不用批量 -1e9 掩码，改成"全排序后按集合过滤"
     （掩码写法一旦漏排/错排，这里会露出差异）
  3. NDCG：用显式位置循环，不用向量化 log2 arange
  4. 批次：一次只算一个用户（batch=1），排除 batch 内互相污染的可能

两者结果一致，才有理由相信 0.1871 这个数。

跑法：
    & python sasrec\verify_metric.py --min_rating 0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dataset import load, pad_seq
from model import build_from_meta

HERE = os.path.dirname(os.path.abspath(__file__))
K, NK = 20, 10


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min_rating', type=int, default=0)
    p.add_argument('--data', type=str, default='')
    p.add_argument('--max_users', type=int, default=0)
    p.add_argument('--ckpt', type=str, default='best.pt')
    p.add_argument('--meta', type=str, default='best_meta.json')
    p.add_argument('--limit', type=int, default=0, help='只算前 N 个用户（快速自测）')
    a = p.parse_args()

    kw = {'min_rating': a.min_rating}
    if a.data:
        kw['data_path'] = a.data
    if a.max_users:
        kw['max_users'] = a.max_users
    _, train, valid, test, n_items = load(**kw)

    meta_path = os.path.join(HERE, a.meta)
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    print(f'推理配置：norm_out={meta.get("norm_out")} '
          f'logit_scale={meta.get("logit_scale", 20.0)} '
          f'max_len={meta.get("max_len")} tag={meta.get("tag")}')

    model = build_from_meta(n_items, meta, os.path.join(HERE, a.ckpt))
    model.eval()

    n = a.limit or len(train)
    hits, ndcg_sum = 0, 0.0
    mismatch = []

    with torch.no_grad():
        for u in range(n):
            hist = train[u] + [valid[u]]
            seq = torch.tensor([pad_seq(hist, meta['max_len'])])
            scores = model(seq, model.norm_out)[0, -1].numpy().astype(np.float64)

            # 路径 2：全排序之后用集合过滤，完全不用掩码
            banned = set(train[u]) | {valid[u]}
            order = np.argsort(-scores, kind='stable')
            cand = [int(i) for i in order if i != 0 and i not in banned][:K]

            target = test[u]
            if target in cand:
                hits += 1
                pos = cand.index(target)          # 路径 3：显式位置算 NDCG
                if pos < NK:
                    ndcg_sum += 1.0 / np.log2(pos + 2.0)

            # 顺手核对：掩码写法算出的 top20 和集合过滤的 top20 必须完全一样
            masked = scores.copy()
            masked[0] = -np.inf
            for it in banned:
                masked[it] = -np.inf
            top_mask = np.argsort(-masked, kind='stable')[:K]
            if len(set(top_mask.tolist()) ^ set(cand)) > 0:
                mismatch.append((u, sorted(set(top_mask.tolist()) ^ set(cand))))

    r, nd = hits / n, ndcg_sum / n
    print('\n===== 独立复算（numpy argsort + 集合过滤 + 逐用户）=====')
    print(f'Recall@20={r:.4f}  NDCG@10={nd:.4f}  （用户 {n}）')
    print(f'过程一致性检查：掩码 top20 与集合过滤 top20 不一致的用户数 = {len(mismatch)}')
    if mismatch[:5]:
        print(f'  前几个不一致样例：{mismatch[:5]}')

    out = os.path.join(HERE, 'verify_metric_result.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'recall20': round(r, 4), 'ndcg10': round(nd, 4), 'n_users': n,
                   'top20_mismatch_users': len(mismatch),
                   'ckpt': a.ckpt, 'tag': meta.get('tag'),
                   'min_rating': a.min_rating}, f, ensure_ascii=False, indent=2)
    print(f'结果已写入 {out}')


if __name__ == '__main__':
    main()
