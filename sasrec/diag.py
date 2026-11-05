# -*- coding: utf-8 -*-
"""诊断脚本：小样本过拟合，判断 SASRec 是实现有 bug 还是训练不足

逻辑：取 200 个用户反复训练，看模型能否在"训练序列自己的最后一项"上命中。
- 命中率能冲到接近 1.0  -> 实现没问题，是泛化/超参问题
- 命中率一直很低       -> 模型/数据管道有 bug
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dataset import load, pad_seq
from model import SASRec

SEED, MAX_LEN, N = 42, 50, 200
torch.manual_seed(SEED)
np.random.seed(SEED)

uids, train, valid, test, n_items = load()
tr = train[:N]
model = SASRec(n_items, MAX_LEN, 64, 2, 2, 0.2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()

for ep in range(1, 201):
    model.train()
    perm = np.random.permutation(N)
    tot, cnt = 0.0, 0
    for s in range(0, N, 32):
        idx = perm[s:s + 32]
        seqs = torch.tensor([pad_seq(tr[u], MAX_LEN + 1) for u in idx])
        x, y = seqs[:, :-1], seqs[:, 1:]
        logits = model(x)
        pos = logits.gather(2, y.unsqueeze(2)).squeeze(2)
        neg = logits.gather(2, torch.randint(1, n_items, y.shape).unsqueeze(2)).squeeze(2)
        mask = y != 0
        loss = -(torch.log(torch.sigmoid(pos[mask]) + 1e-8) +
                 torch.log(1 - torch.sigmoid(neg[mask]) + 1e-8)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        cnt += 1

    if ep % 20 == 0:
        model.eval()
        with torch.no_grad():
            seqs = torch.tensor([pad_seq(tr[u], MAX_LEN + 1) for u in range(N)])
            logits = model(seqs[:, :-1])[:, -1, :].clone()
            hits = 0
            for i in range(N):
                logits[i, 0] = -1e9
                hist = set(tr[i][-MAX_LEN:-1])
                if hist:
                    logits[i, torch.tensor(sorted(hist))] = -1e9
                top = torch.topk(logits[i], 20).indices.tolist()
                hits += int(tr[i][-1] in top)
        print(f'ep {ep:3d} loss={tot / cnt:.4f} '
              f'训练集命中@20={hits / N:.3f} ({time.time() - t0:.0f}s)')
