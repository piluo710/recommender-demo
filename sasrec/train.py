# -*- coding: utf-8 -*-
"""SASRec 训练 + 评估（leave-one-out 口径，参数化版）

跑法（在 recommender-demo 目录下）：
    & 'D:\download\miniconda3\envs\stats\python.exe' sasrec\train.py --max_len 100 --neg 10

产物：
- sasrec\best.pt              最优模型权重
- sasrec\result_<tag>.json    测试指标 + 每轮历史
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from dataset import load, pad_seq, load_from_args
from model import SASRec

SEED, K, NDCG_K = 42, 20, 10
HERE = os.path.dirname(os.path.abspath(__file__))


def setup():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--pos_scale', type=float, default=1.0,
                   help='位置编码权重；0=关闭（验证位置编码是否成了公共成分）')
    p.add_argument('--res_scale', type=float, default=1.0,
                   help='残差缩放系数；1.0=标准（与官方实现一致），小于 1 会削弱每层贡献')
    p.add_argument('--norm_out', action='store_true',
                   help='输出层改用余弦相似度 + 固定温度；原始点积尺度 40+，softmax 接近 one-hot，梯度只作用在最高频物品上')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--neg', type=int, default=10, help='每个正样本的随机负样本数')
    p.add_argument('--neg_mode', type=str, default='uniform',
                   choices=['uniform', 'pop'],
                   help='负采样分布：uniform 均匀（默认，稳定）/ pop 按流行度加权（实测会崩，仅用于复现实验）')
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--window', type=str, default='random', choices=['random', 'fixed'],
                   help='训练序列窗口：random 每轮随机截取 / fixed 固定最近 max_len')
    p.add_argument('--min_rating', type=int, default=4,
                   help='4=只用 4 分以上（原项目口径）；0=用全部评分（序列更长）')
    p.add_argument('--data', type=str, default='',
                   help='ratings.dat 路径（默认 ../data/ml-1m/ratings.dat）')
    p.add_argument('--max_users', type=int, default=0,
                   help='随机采样用户数（0=全部）')
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()


def ndcg_at_k(rel, k):
    """与 itemcf_recall.py 完全一致的口径"""
    rel = rel[:k]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def evaluate(model, user_inputs, targets, exclude_sets, n_items, max_len, batch=128):
    """全池打分：输入序列 -> 预测下一位；排除已交互 item 后算 Recall@20 / NDCG@10"""
    model.eval()
    hits, ndcg_sum, n = 0, 0.0, 0
    for start in range(0, len(user_inputs), batch):
        end = min(start + batch, len(user_inputs))
        seqs = torch.tensor([pad_seq(user_inputs[u], max_len)
                             for u in range(start, end)])
        logits = model(seqs, model.norm_out)[:, -1, :]
        scores = torch.nan_to_num(logits.clone(), nan=-1e9)
        scores[:, 0] = -1e9
        for i, u in enumerate(range(start, end)):
            if u in exclude_sets:
                idx = torch.tensor(sorted(exclude_sets[u]), dtype=torch.long)
                scores[i, idx] = -1e9
            top = torch.topk(scores[i], K).indices.cpu().numpy()
            target = targets[u]
            hits += int(target in top)
            rel = np.array([1 if j == target else 0 for j in top])
            ndcg_sum += ndcg_at_k(rel, NDCG_K)
            n += 1
    return hits / n, ndcg_sum / n


def sample_window(seq, max_len):
    """训练时随机截取长度为 max_len+1 的窗口（每个 epoch 不同）

    固定窗口（最近 N 个）容易让模型记住"这个用户的特定片段"，
    随机窗口迫使它学通用模式，提升泛化。
    """
    if len(seq) > max_len + 1:
        end = np.random.randint(max_len + 1, len(seq) + 1)
        return seq[end - max_len - 1:end]
    return seq


def train_one_epoch(model, opt, train_seqs, n_items, max_len, neg, batch, pop_t=None,
                    random_window=True):
    """sampled softmax：1 个正样本 + neg 个负样本

    pop_t 非空时按流行度分布采负样本：热门 item 更常作为负样本出现，
    逼模型学会"热门但该用户不会看"，否则模型只会押热门（退化成流行度基线）。
    """
    model.train()
    n_users = len(train_seqs)
    perm = np.random.permutation(n_users)
    total, steps = 0.0, 0
    for start in range(0, n_users, batch):
        idx = perm[start:start + batch]
        if random_window:
            seqs = torch.tensor([pad_seq(sample_window(train_seqs[u], max_len), max_len + 1)
                                 for u in idx])
        else:
            seqs = torch.tensor([pad_seq(train_seqs[u], max_len + 1) for u in idx])
        x, y = seqs[:, :-1], seqs[:, 1:]     # 长度 max_len，位置编码全覆盖
        logits = model(x, model.norm_out)    # [B, L, V]
        pos = logits.gather(2, y.unsqueeze(2))                                   # [B, L, 1]
        if pop_t is not None:
            neg_ids = torch.multinomial(pop_t, y.numel() * neg,
                                        replacement=True).view(y.shape[0], y.shape[1], neg)
        else:
            neg_ids = torch.randint(1, n_items, (y.shape[0], y.shape[1], neg))
        neg_logits = logits.gather(2, neg_ids)                                  # [B, L, K]
        cat = torch.cat([pos, neg_logits], dim=2)                               # 正样本在 index 0
        mask = (y != 0)
        loss = F.cross_entropy(cat[mask], torch.zeros(cat[mask].shape[0], dtype=torch.long))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
        steps += 1
    return total / steps


def main():
    a = parse_args()
    setup()
    uids, train, valid, test, n_items = load_from_args(a)
    model = SASRec(n_items, a.max_len, a.d, a.blocks, a.heads, a.dropout,
                   pos_scale=a.pos_scale, res_scale=a.res_scale)
    model.norm_out = a.norm_out
    n_params = sum(p.numel() for p in model.parameters())
    print(f'配置：max_len={a.max_len} d={a.d} neg={a.neg} lr={a.lr} '
          f'epochs={a.epochs}｜参数 {n_params:,}')
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    # 流行度分布（用于按流行度采负样本）
    pop_t = None
    if a.neg_mode == 'pop':
        counts = np.zeros(n_items, dtype=np.float64)
        for s in train:
            for it in s:
                counts[it] += 1
        counts[0] = 0
        pop_t = torch.tensor(counts / counts.sum(), dtype=torch.float32)
        print(f'负采样：按流行度加权（最热 item 占比 {pop_t.max():.4f}）')

    best, best_ep, bad = -1.0, -1, 0
    history = []
    t0 = time.time()

    def save_ckpt(m):
        """权重 + 推理所需元信息一起落盘。

        state_dict 里不含 norm_out，若只存权重，任何脚本加载后都会默认走点积分支，
        拿到的是"看起来合理、其实错误"的分数。所以把推理配置单独写成 best_meta.json。
        """
        torch.save(m.state_dict(), os.path.join(HERE, 'best.pt'))
        with open(os.path.join(HERE, 'best_meta.json'), 'w', encoding='utf-8') as f:
            json.dump({'norm_out': bool(a.norm_out), 'max_len': a.max_len, 'd': a.d,
                       'blocks': a.blocks, 'heads': a.heads, 'pos_scale': a.pos_scale,
                       'res_scale': a.res_scale, 'min_rating': a.min_rating,
                       'tag': a.tag}, f, ensure_ascii=False, indent=2)

    for ep in range(1, a.epochs + 1):
        loss = train_one_epoch(model, opt, train, n_items, a.max_len, a.neg,
                               a.batch, pop_t, a.window == 'random')
        excl = {u: set(train[u]) for u in range(len(train))}
        vr, vn = evaluate(model, train, valid, excl, n_items, a.max_len)
        history.append({'epoch': ep, 'loss': round(loss, 4),
                        'valid_recall20': round(vr, 4), 'valid_ndcg10': round(vn, 4)})
        print(f'epoch {ep:02d} loss={loss:.4f} valid R@20={vr:.4f} '
              f'NDCG@10={vn:.4f} ({time.time() - t0:.0f}s)')
        if vn > best:
            best, best_ep, bad = vn, ep, 0
            save_ckpt(model)
        else:
            bad += 1
            if bad >= a.patience:
                print(f'早停于 epoch {ep}，最优 epoch={best_ep}')
                break

    model.load_state_dict(torch.load(os.path.join(HERE, 'best.pt'), weights_only=True))
    inputs = [train[u] + [valid[u]] for u in range(len(train))]
    excl = {u: set(train[u]) | {valid[u]} for u in range(len(train))}
    tr, tn = evaluate(model, inputs, test, excl, n_items, a.max_len)
    print('\n===== SASRec test（leave-one-out）=====')
    print(f'Recall@20={tr:.4f}  NDCG@10={tn:.4f}  最优 epoch={best_ep}')

    cfg = {k: v for k, v in vars(a).items()}
    result = {'config': cfg, 'test_recall20': round(tr, 4),
              'test_ndcg10': round(tn, 4), 'best_epoch': best_ep,
              'n_users': len(train), 'n_items': n_items - 1,
              'params': n_params, 'history': history}
    tag = a.tag or f'L{a.max_len}_neg{a.neg}_lr{a.lr}'
    out = os.path.join(HERE, f'result_{tag}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'结果已写入 {out}')


if __name__ == '__main__':
    main()
