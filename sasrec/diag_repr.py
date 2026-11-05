# -*- coding: utf-8 -*-
"""诊断：用户表示的数值形态

看 last_ln 输出的具体数值，判断坍缩属于哪种：
- 各维都接近同一个常数（x ≈ beta）  -> LayerNorm 把残差偏置投影出来了
- 只有少数几维有差异               -> 个性化信号被压到低方差方向
- 模长/均值正常但方向一致          -> 注意力输出共模
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dataset import load_from_args
from model import SASRec

MAX_LEN = 50


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min_rating', type=int, default=4)
    p.add_argument('--data', type=str, default='')
    p.add_argument('--max_users', type=int, default=0)
    a = p.parse_args()

    uids, train, valid, test, n_items = load_from_args(a)
    model = SASRec(n_items, MAX_LEN, 64, 2, 2, 0.0)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'best.pt'), weights_only=True))
    model.eval()

    from dataset import pad_seq
    reps, mids = [], []
    for u in range(6):
        seq = train[u] + [valid[u]]
        x = model.encode(torch.tensor([pad_seq(seq, MAX_LEN)]))[0, -1].numpy()
        reps.append(x)
        mids.append(len(seq))
        print(f'用户{u} 序列长={len(seq):3d} 前8维={np.round(x[:8], 3)}')
        print(f'         均值={x.mean():+.3f} 标准差={x.std():.3f} 模长={np.linalg.norm(x):.2f}')

    reps = np.array(reps)
    print(f'\n6 个用户逐维标准差（取均值的跨度）：{reps.std(axis=0).mean():.4f}')
    print(f'6 个用户逐维均值的范围（max-min）：{(reps.max(0) - reps.min(0)).mean():.4f}')

    # 哪几维真正携带用户差异
    sd = reps.std(axis=0)
    order = np.argsort(-sd)
    print(f'\n跨用户标准差最大的 10 维：{order[:10].tolist()}')
    print(f'对应标准差：{np.round(sd[order[:10]], 4).tolist()}')
    print(f'其余 {64 - 10} 维合计方差占比：'
          f'{sd[order[:10]].var() / max(sd.var(), 1e-12) * 0 + sd[order[10:]].sum() / max(sd.sum(), 1e-12):.3f}')


if __name__ == '__main__':
    main()
