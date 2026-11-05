# -*- coding: utf-8 -*-
"""诊断3：极限过拟合（3 个用户），并打印目标 item 的实际排名

3 个用户、400 轮，如果命中率还是 0，说明模型/数据管道有硬 bug。
同时打印：目标 item 的 logit 排名、分数，和 top-5 的对比。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from dataset import load, pad_seq
from model import SASRec

SEED, MAX_LEN, N, NEG = 42, 50, 3, 10
torch.manual_seed(SEED)
np.random.seed(SEED)

uids, train, valid, test, n_items = load()
tr = train[:N]
model = SASRec(n_items, MAX_LEN, 64, 2, 2, 0.0)   # dropout 关掉，纯过拟合测试
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(1, 401):
    model.train()
    seqs = torch.tensor([pad_seq(tr[u], MAX_LEN + 1) for u in range(N)])
    x, y = seqs[:, :-1], seqs[:, 1:]
    logits = model(x)
    pos = logits.gather(2, y.unsqueeze(2))
    neg = logits.gather(2, torch.randint(1, n_items, (y.shape[0], y.shape[1], NEG)))
    cat = torch.cat([pos, neg], dim=2)
    mask = (y != 0)
    loss = F.cross_entropy(cat[mask], torch.zeros(cat[mask].shape[0], dtype=torch.long))
    opt.zero_grad()
    loss.backward()
    opt.step()

    if ep % 50 == 0:
        model.eval()
        with torch.no_grad():
            full = torch.tensor([pad_seq(tr[u], MAX_LEN + 1) for u in range(N)])
            lg = model(full[:, :-1])[:, -1, :].clone()   # 最后位置 -> 预测 full[:,-1]
            for i in range(N):
                lg[i, 0] = -1e9
                hist = set(tr[i][-MAX_LEN:-1])
                if hist:
                    lg[i, torch.tensor(sorted(hist))] = -1e9
                target = tr[i][-1]
                rank = int((lg[i] > lg[i, target]).sum().item()) + 1
                top5 = torch.topk(lg[i], 5)
                print(f'ep {ep:3d} loss={loss.item():.4f} user{i} '
                      f'target={target} rank={rank} '
                      f'score={lg[i, target]:.3f} '
                      f'top5={[(int(v), round(float(s), 2)) for v, s in zip(top5.indices, top5.values)]}')
            print()
