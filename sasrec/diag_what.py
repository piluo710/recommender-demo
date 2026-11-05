# -*- coding: utf-8 -*-
"""诊断：模型到底在推荐什么（参数化，不再写死 min_rating）

一次回答四个问题：
1. 不同用户的 top20 彼此重合多少（越高说明越"千人一面"）
2. top20 与全局最热 top20 重合多少（越高说明在押热门）
3. item_emb 模长与流行度的 Spearman 相关（模长是否主导排序）
4. 模长最大的 10 个 item 是否就是最热的 10 个

跑法：
    & python sasrec\diag_what.py --min_rating 0
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dataset import pad_seq, load
from model import build_from_meta

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
    p.add_argument('--n_users', type=int, default=30)
    return p.parse_args()


@torch.no_grad()
def main():
    a = parse_args()
    kw = {'min_rating': a.min_rating}
    if a.data:
        kw['data_path'] = a.data
    if a.max_users:
        kw['max_users'] = a.max_users
    uids, train, valid, test, n_items = load(**kw)

    # 权重必须按照训练时的配置推理。state_dict 不含 norm_out / logit_scale /
    # 结构参数，所以一律从 best_meta.json 还原；缺失时明确告警，不默认猜。
    meta_path = os.path.join(HERE, 'best_meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        print(f'推理配置：norm_out={meta.get("norm_out")} '
              f'logit_scale={meta.get("logit_scale", 20.0)} '
              f'（来自 best_meta.json，tag={meta.get("tag")}）')
    else:
        print('[警告] 缺少 best_meta.json，按点积模式 + 命令行默认结构推理；'
              '若权重来自余弦输出层版本，以下数字全部失真')
    # 结构参数取自 meta 而不是命令行默认值：默认值和训练配置不一致同样会静默出错
    model = build_from_meta(n_items, meta, os.path.join(HERE, 'best.pt'))
    model.eval()

    # ---- 全局最热 top20（训练集内出现次数）
    cnt = Counter()
    for s in train:
        cnt.update(s)
    hot20 = set([it for it, _ in cnt.most_common(K)])
    pop_of = np.zeros(n_items)
    total = sum(cnt.values())
    for it, c in cnt.items():
        pop_of[it] = c / total

    # ---- 每个用户的 top20
    tops = []
    for u in range(min(a.n_users, len(train))):
        hist = train[u] + [valid[u]]
        seqs = torch.tensor([pad_seq(hist, a.max_len)])
        logits = model(seqs, model.norm_out)[:, -1, :].clone()
        logits[0, 0] = -1e9
        idx = torch.tensor(sorted(set(hist)), dtype=torch.long)
        logits[0, idx] = -1e9
        top = torch.topk(logits[0], K).indices.numpy()
        tops.append(set(top.tolist()))

    # 1) 用户之间
    pair = []
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            pair.append(len(tops[i] & tops[j]))
    print(f'\n[1] {len(tops)} 个用户两两 top20 平均重合：{np.mean(pair):.2f}/{K} '
          f'（越高越"千人一面"）')

    # 2) 与热门
    hot_overlap = [len(t & hot20) for t in tops]
    print(f'[2] top20 与全局最热 top20 平均重合：{np.mean(hot_overlap):.2f}/{K}')

    # 3) 模长 vs 流行度
    w = model.item_emb.weight.detach()
    norm = w.norm(dim=1).numpy()[1:]
    pop = pop_of[1:]
    ok = pop > 0
    from math import isnan

    def spearman(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return np.corrcoef(rx, ry)[0, 1]
    rho = spearman(norm[ok], pop[ok])
    print(f'[3] item_emb 模长 vs 流行度 Spearman：{rho:.3f} '
          f'（接近 1 说明排序被"模长大的热门"主导）')

    # 4) 模长 top10 是不是热门 top10
    top_norm = set((np.argsort(-norm)[:10] + 1).tolist())
    top_hot = set([it for it, _ in cnt.most_common(10)])
    print(f'[4] 模长最大 10 个 item 与最热 10 个重合：{len(top_norm & top_hot)}/10')
    print(f'    模长：均值 {norm.mean():.2f}，最小 {norm.min():.2f}，最大 {norm.max():.2f}')

    # 5) item embedding 之间有没有学到"共现结构"
    wn = w[1:] / (w[1:].norm(dim=1, keepdim=True) + 1e-9)
    rng = np.random.default_rng(0)
    i = rng.integers(0, wn.shape[0], 500)
    j = rng.integers(0, wn.shape[0], 500)
    cos = (wn[i] * wn[j]).sum(dim=1).numpy()
    print(f'[5] item_emb 随机 500 对余弦：均值 {cos.mean():.4f}，'
          f'标准差 {cos.std():.4f}（接近 0 = 方向接近随机，没学到物品关联）')

    # 对照：热门 item 与"用户表示均值"的对齐程度
    with torch.no_grad():
        embs = []
        for u in range(min(200, len(train))):
            seqs = torch.tensor([pad_seq(train[u] + [valid[u]], a.max_len)])
            embs.append(model.encode(seqs)[0, -1])
        mean_u = torch.stack(embs).mean(0)
        mean_u = mean_u / (mean_u.norm() + 1e-9)
        align = (wn.to(mean_u.dtype) @ mean_u).numpy()
    pop_v = pop[ok]
    al_v = align[ok]
    print(f'[6] item_emb 与"用户表示均值方向"的余弦 vs 流行度 Spearman：'
          f'{spearman(al_v, pop_v):.3f}（高 = 热门 item 天然和公共方向对齐）')


if __name__ == '__main__':
    main()
