# -*- coding: utf-8 -*-
"""对照实验：去掉 attention，只保留 item embedding 的"纯矩阵分解"

目的：把"attention 结构有问题"和"sampled softmax 训练目标学不出物品关联"两件事分开。
同一套数据、同一个训练循环（1 正 + neg 负、sampled softmax），只把 forward 换成
    logits = item_emb(seq) @ item_emb.T     # 每个位置独立，无 attention
跑法：
    & python sasrec\diag_fpmc.py --min_rating 0 --epochs 20
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load, pad_seq


def sample_window(seq, max_len):
    """随机截取一段长度 max_len+1 的窗口（训练用，末尾位置的下一个就是标签）"""
    if len(seq) > max_len + 1:
        end = np.random.randint(max_len + 1, len(seq) + 1)
        return seq[end - (max_len + 1):end]
    return seq

HERE = os.path.dirname(os.path.abspath(__file__))


class PureMF(nn.Module):
    """只用 item embedding，不做任何序列建模"""
    def __init__(self, n_items, d=64, dropout=0.0):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, d, padding_idx=0)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.drop(self.item_emb(seq))
        return x @ self.item_emb.weight.T


def evaluate_pairs(model, pairs, max_len, K=20):
    """pairs: [(历史序列, 目标item)]"""
    model.eval()
    hits, ndcg, n = 0, 0.0, 0
    for hist, tgt in pairs:
        seq = torch.tensor([pad_seq(hist, max_len)])
        logits = model(seq)[:, -1, :].clone()
        logits[0, 0] = -1e9
        logits[0, torch.tensor(sorted(set(hist)), dtype=torch.long)] = -1e9
        top = torch.topk(logits[0], K).indices.tolist()
        if tgt in top:
            hits += 1
            ndcg += 1.0 / np.log2(top.index(tgt) + 2)
        n += 1
    return hits / max(n, 1), ndcg / max(n, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--min_rating', type=int, default=0)
    p.add_argument('--data', type=str, default='')
    p.add_argument('--max_users', type=int, default=0)
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--neg', type=int, default=10)
    p.add_argument('--patience', type=int, default=6)
    return p.parse_args()


def main():
    a = parse_args()
    kw = {'min_rating': a.min_rating}
    if a.data:
        kw['data_path'] = a.data
    if a.max_users:
        kw['max_users'] = a.max_users
    uids, train, valid, test, n_items = load(**kw)
    print(f'可用用户 {len(uids)}，物品数 {n_items}')

    # 训练集：每个用户随机窗口，切 x/y
    xs, ys = [], []
    for u in range(len(train)):
        seq = sample_window(train[u], a.max_len)
        if len(seq) < 2:
            continue
        padded = pad_seq(seq, a.max_len + 1)
        xs.append(padded[:-1])
        ys.append(padded[1:])
    X = torch.tensor(np.array(xs), dtype=torch.long)
    Y = torch.tensor(np.array(ys), dtype=torch.long)

    model = PureMF(n_items, a.d, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(42)

    # valid 用 train -> 倒数第2个，test 用 train+valid -> 最后一个。
    # 早停只能看 valid；之前两个都写成 test，等于拿测试集选 epoch，指标会虚高。
    valid_pairs = [(train[u], valid[u]) for u in range(len(train))]
    test_pairs = [(train[u] + [valid[u]], test[u]) for u in range(len(train))]

    best, best_ep, bad = -1, 0, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        idx = torch.randperm(len(X), generator=gen)
        for b in range(0, len(X), a.batch):
            sel = idx[b:b + a.batch]
            x, y = X[sel], Y[sel]
            logits = model(x)
            pos = logits.gather(2, y.unsqueeze(2))
            negs = torch.randint(1, n_items, (x.shape[0], x.shape[1], a.neg),
                                 generator=gen, dtype=torch.long)
            neg_logits = torch.gather(logits, 2, negs)
            all_logits = torch.cat([pos, neg_logits], dim=2)
            loss = F.cross_entropy(all_logits.reshape(-1, a.neg + 1),
                                  torch.zeros(x.shape[0] * x.shape[1], dtype=torch.long),
                                  reduction='none')
            mask = (y != 0).reshape(-1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        r, ng = evaluate_pairs(model, valid_pairs, a.max_len)
        print(f'epoch {ep:02d} loss={loss.item():.4f} valid R@20={r:.4f} NDCG@10={ng:.4f} '
              f'({time.time() - t0:.0f}s)')
        if r > best:
            best, best_ep, bad = r, ep, 0
            torch.save(model.state_dict(), os.path.join(HERE, 'fpmc_best.pt'))
        else:
            bad += 1
            if bad >= a.patience:
                print(f'早停于 epoch {ep}，最优 epoch={best_ep}')
                break

    model.load_state_dict(torch.load(os.path.join(HERE, 'fpmc_best.pt'), weights_only=True))
    r, ng = evaluate_pairs(model, test_pairs, a.max_len)
    print(f'\n===== 纯 item embedding（无 attention）test =====')
    print(f'Recall@20={r:.4f}  NDCG@10={ng:.4f}  最优 epoch={best_ep}')
    with open(os.path.join(HERE, 'result_pure_mf.json'), 'w', encoding='utf-8') as f:
        json.dump({'recall': r, 'ndcg': ng, 'best_epoch': best_ep,
                   'min_rating': a.min_rating, 'model': 'PureMF(no attention)'}, f,
                  ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
