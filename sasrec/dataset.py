# -*- coding: utf-8 -*-
"""SASRec 数据准备：MovieLens-1M，每用户行为序列（leave-one-out 划分）

为什么用 leave-one-out（而不是现有基线的 80/20 随机划分）：
- 序列模型按"行为顺序"建模。随机划分会让训练序列混进时间上更晚的交互，
  等于模型提前看到了测试时刻之后的用户行为（信息泄漏，指标虚高）
- 所以按时间切：每用户最后 1 个交互作测试，倒数第 2 个作验证，其余作训练
- 为保持对比公平，同口径下重跑 ItemCF 基线（baseline_itemcf_loo.py）
"""

import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'data', 'ml-1m', 'ratings.dat')
MIN_LEN = 5   # 训练序列最短长度，太短学不到顺序


def load(data_path=DATA, min_len=MIN_LEN):
    """返回 (uid_list, train_list, valid_list, test_list, n_items)

    train_list[u] 是用户 u 的训练序列（list of item id，按时间升序）
    valid_list[u] / test_list[u] 是单个 item id（倒数第 2 / 最后一个）
    """
    df = pd.read_csv(data_path, sep='::', header=None, engine='python',
                     names=['uid', 'mid', 'rating', 'ts'])
    df = df[df['rating'] >= 4]              # 隐式反馈：4 分以上算一次"喜欢"
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

    print(f'可用用户 {len(uid_list)} / 原始 {df.uid.nunique()}，'
          f'物品数 {n_items - 1}，训练序列平均长度 '
          f'{np.mean([len(s) for s in train]):.1f}')
    return uid_list, train, valid, test, n_items


def pad_seq(seq, max_len):
    """左对齐截断 + 左侧补 0（0 是 padding id）"""
    seq = list(seq)[-max_len:]
    return [0] * (max_len - len(seq)) + seq


if __name__ == '__main__':
    _, train, valid, test, n_items = load()
    print('示例：用户 0 训练序列前 5 个 ->', train[0][:5])
    print('示例：用户 0 验证/测试 ->', valid[0], test[0])
