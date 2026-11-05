# -*- coding: utf-8 -*-
"""SASRec 数据准备：MovieLens-1M，每用户行为序列（leave-one-out 划分）

为什么用 leave-one-out（而不是现有基线的 80/20 随机划分）：
- 序列模型按"行为顺序"建模。随机划分会让训练序列混进时间上更晚的交互，
  等于模型提前看到了测试时刻之后的用户行为（信息泄漏，指标虚高）
- 所以按时间切：每用户最后 1 个交互作测试，倒数第 2 个作验证，其余作训练
- 为保持对比公平，同口径下重跑 ItemCF 基线（baseline_itemcf_loo.py）
"""

import json
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'data', 'ml-1m', 'ratings.dat')
MIN_LEN = 5   # 训练序列最短长度，太短学不到顺序


def default_min_rating_from_meta(here=None):
    """返回 best.pt 的训练口径（best_meta.json 里的 min_rating）。

    所有加载 best.pt 的诊断脚本都应以此作默认口径：
    数据物品数由 min_rating 决定，和权重行数不一致时
    load_state_dict 会 size mismatch 直接崩，而不是给出可用的结果。
    """
    try:
        if here is None:
            here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, 'best_meta.json'), encoding='utf-8') as f:
            return json.load(f).get('min_rating', 4)
    except Exception:
        return 4


def load(data_path=DATA, min_len=MIN_LEN, min_rating=4, max_users=None):
    """返回 (uid_list, train_list, valid_list, test_list, n_items)

    train_list[u] 是用户 u 的训练序列（list of item id，按时间升序）
    valid_list[u] / test_list[u] 是单个 item id（倒数第 2 / 最后一个）

    min_rating=4 只保留 4 分以上（隐式反馈口径，和原项目一致）
    min_rating=0 用全部评分（序列更长，更接近 SASRec 论文设定）
    max_users   用户太多时随机采样（CPU 训练成本控制）
    """
    df = pd.read_csv(data_path, sep='::', header=None, engine='python',
                     names=['uid', 'mid', 'rating', 'ts'])
    if min_rating > 0:
        df = df[df['rating'] >= min_rating]  # 隐式反馈：4 分以上算一次"喜欢"
    df = df.sort_values(['uid', 'ts'])      # 按用户 + 时间排序

    items = sorted(df['mid'].unique())
    mid2j = {m: i + 1 for i, m in enumerate(items)}   # 1 起编，0 留给 padding
    n_items = len(items) + 1

    uid_list, train, valid, test = [], [], [], []
    for uid, grp in df.groupby('uid'):
        seq = [mid2j[m] for m in grp['mid'].values]
        if len(seq) < min_len + 2:          # 至少 min_len 训练 + 1 验证 + 1 测试
            continue
        train.append(seq[:-2])
        valid.append(seq[-2])
        test.append(seq[-1])
        uid_list.append(uid)

    if max_users is not None and len(uid_list) > max_users:
        rng = np.random.default_rng(42)          # 固定种子，保证各脚本采样一致
        keep = np.sort(rng.choice(len(uid_list), max_users, replace=False))
        uid_list = [uid_list[i] for i in keep]
        train = [train[i] for i in keep]
        valid = [valid[i] for i in keep]
        test = [test[i] for i in keep]

    print(f'可用用户 {len(uid_list)} / 原始 {df.uid.nunique()}，'
          f'物品数 {n_items - 1}，训练序列平均长度 '
          f'{np.mean([len(s) for s in train]):.1f}')
    return uid_list, train, valid, test, n_items


def load_from_args(a):
    """统一从 argparse 参数构造 load 调用，保证各脚本用完全相同的采样"""
    kw = {'min_rating': a.min_rating}
    data = getattr(a, 'data', '') or ''
    if data:
        kw['data_path'] = data
    if getattr(a, 'max_users', 0):
        kw['max_users'] = a.max_users
    return load(**kw)


def pad_seq(seq, max_len):
    """左对齐截断 + 左侧补 0（0 是 padding id）"""
    seq = list(seq)[-max_len:]
    return [0] * (max_len - len(seq)) + seq


if __name__ == '__main__':
    _, train, valid, test, n_items = load()
    print('示例：用户 0 训练序列前 5 个 ->', train[0][:5])
    print('示例：用户 0 验证/测试 ->', valid[0], test[0])
