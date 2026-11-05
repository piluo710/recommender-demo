# -*- coding: utf-8 -*-
"""诊断2：把 BCE(1:1) 换成 sampled softmax(1正+K负)，看训练集命中率能否上来

如果命中率明显上升 -> 问题是训练目标太弱，改 train.py
如果仍然很低     -> 继续查模型/数据管道
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from dataset import load, pad_seq
from model import SASRec

SEED, MAX_LEN, N, NEG = 42, 50, 200, 10
torch.manual_seed(SEED)
np.random.seed(SEED)

uids, train, valid, test, n_items = load()
tr = train[:N]
model = SASRec(n_items, MAX_LEN, 64, 2, 2, 0.2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()

for ep in range(1, 121):
    model.train()
    perm = np.random.permutation(N)
    tot, cnt = 0.0, 0
    for s in range(0, N, 32):
        idx = perm[s:s + 32]
        seqs = torch.tensor([pad_seq(tr[u], MAX_LEN + 1) for u in idx])
        x, y = seqs[:, :-1], seqs[:, 1:]
        logits = model(x)                                     # [B, L, V]
        pos = logits.gather(2, y.unsqueeze(2))                 # [B, L, 1]
        neg = logits.gather(
            2, torch.randint(1, n_items, (y.shape[0], y.shape[1], NEG))
        )                                                      # [B, L, K]
        cat = torch.cat([pos, neg], dim=2)                     # 正样本在 index 0
        mask = (y != 0)
        logits_bn = cat[mask]                                  # [N_valid, 1+K]
        target = torch.zeros(logits_bn.shape[0], dtype=torch.long)
        loss = F.cross_entropy(logits_bn, target)
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
            hits10, hits20 = 0, 0
            for i in range(N):
                logits[i, 0] = -1e9
                hist = set(tr[i][-MAX_LEN:-1])
                if hist:
                    logits[i, torch.tensor(sorted(hist))] = -1e9
                top20 = torch.topk(logits[i], 20).indices.tolist()
                hits20 += int(tr[i][-1] in top20)
                hits10 += int(tr[i][-1] in top20[:10])
            print(f'ep {ep:3d} loss={tot / cnt:.4f} '
                  f'训练集命中@20={hits20 / N:.3f} @10={hits10 / N:.3f} '
                  f'({time.time() - t0:.0f}s)')
