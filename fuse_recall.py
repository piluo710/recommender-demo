# 多路召回对比 + 加权融合（Day 2，v2）
# 四路数字同批同口径对比：
#   A. ItemCF          ：电影相似度 = 共现（昨天的方法，baseline）
#   B. ItemMF-ALS      ：隐式反馈矩阵分解（Hu et al. 2008），自己手写，
#                        直接对"喜欢/没喜欢"建模（不是预测 1~5 分）——主推这一路
#   C. ItemMF-SVD(dot) ：surprise 显式评分 SVD 的 user·item 内积（试错记录，弱）
#   D. ItemMF-SVD(cos) ：显式评分 SVD 电影向量余弦（试错记录，弱）
# 融合：A、B 各自按用户归一化到 [0,1]，加权 w*A+(1-w)*B，扫权重
# 跑法：& 'D:\download\miniconda3\envs\stats\python.exe' fuse_recall.py

import time
import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

DATA = r'data\ml-1m\ratings.dat'
K, NDCG_K, SEED = 20, 10, 42
ALS_F, ALS_ALPHA, ALS_REG, ALS_ITERS = 64, 40.0, 0.1, 15
t0 = time.time()

# ---------- 公共：读数据 / 划分 / 建矩阵（与前两天完全一致） ----------
df = pd.read_csv(DATA, sep='::', header=None, engine='python',
                 names=['uid', 'mid', 'rating', 'ts'])
df = df[df['rating'] >= 4]
rng = np.random.default_rng(SEED)
is_train = rng.random(len(df)) < 0.8
train, test = df[is_train], df[~is_train]
users = np.sort(train.uid.unique())
items = np.sort(train.mid.unique())
uid2i = {u: i for i, u in enumerate(users)}
mid2j = {m: j for j, m in enumerate(items)}
R = np.zeros((len(users), len(items)), dtype=np.float32)
for u, m in zip(train.uid.values, train.mid.values):
    R[uid2i[u], mid2j[m]] = 1.0

def ndcg_at_k(rel, k):
    rel = rel[:k]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate(scores):
    s = scores.copy()
    s[R == 1] = -np.inf
    recall_hits, ndcg_sum, n_users, denom = 0, 0.0, 0, 0
    for u, grp in test.groupby('uid'):
        if u not in uid2i:
            continue
        test_items = [mid2j[m] for m in grp.mid.values if m in mid2j]
        if not test_items:
            continue
        row = s[uid2i[u]]
        top_k = np.argpartition(row, -K)[-K:]
        top_k = top_k[np.argsort(row[top_k])[::-1]]
        hits = sum(1 for j in top_k if j in test_items)
        recall_hits += hits
        rel = np.array([1 if j in test_items else 0 for j in top_k])
        ndcg_sum += ndcg_at_k(rel, NDCG_K)
        n_users += 1
        denom += len(test_items)
    return recall_hits / denom if denom else 0.0, ndcg_sum / n_users if n_users else 0.0, n_users

def norm_per_user(scores):
    out = scores.copy()
    for i in range(out.shape[0]):
        row = out[i]
        valid = row[np.isfinite(row)]
        if valid.size == 0 or valid.max() == valid.min():
            out[i] = 0.0
            continue
        lo, hi = valid.min(), valid.max()
        row = (row - lo) / (hi - lo)
        row[np.isinf(out[i])] = -np.inf
        out[i] = row
    return out

# ---------- A. ItemCF：共现余弦相似度 ----------
norm_cf = R / np.linalg.norm(R, axis=0, keepdims=True)
S_cf = norm_cf.T @ norm_cf
np.fill_diagonal(S_cf, 0)
scores_cf = R @ S_cf

# ---------- B. ItemMF-ALS：隐式反馈矩阵分解（自己手写） ----------
def implicit_als(R, factors=ALS_F, alpha=ALS_ALPHA, reg=ALS_REG,
                 iters=ALS_ITERS, seed=SEED):
    """隐式 ALS：最小化 Σ c_ui(p_ui - x_u·y_i)^2 + λ(‖x‖²+‖y‖²)
    p_ui = 1（喜欢过）或 0；c_ui = 1 + α·r_ui（置信度，r 为 0/1）"""
    n_users, n_items = R.shape
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, (n_users, factors))
    Y = rng.normal(0, 0.01, (n_items, factors))
    I = np.eye(factors)
    for it in range(iters):
        YtY = Y.T @ Y
        for u in range(n_users):
            pos = np.nonzero(R[u])[0]
            c = 1.0 + alpha * R[u, pos]
            A = YtY + reg * I
            Yp = Y[pos]
            A += (Yp * (c - 1)[:, None]).T @ Yp
            X[u] = np.linalg.solve(A, Yp.T @ c)
        XtX = X.T @ X
        for j in range(n_items):
            pos = np.nonzero(R[:, j])[0]
            c = 1.0 + alpha * R[pos, j]
            A = XtX + reg * I
            Xp = X[pos]
            A += (Xp * (c - 1)[:, None]).T @ Xp
            Y[j] = np.linalg.solve(A, Xp.T @ c)
    return X, Y

X_als, Y_als = implicit_als(R)
scores_als = X_als @ Y_als.T

# ---------- C/D. 试错记录：显式评分 SVD 两种用法 ----------
reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(
    train[['uid', 'mid', 'rating']], reader).build_full_trainset()
algo = SVD(n_factors=64, random_state=SEED)
algo.fit(trainset)
U_idx = np.array([trainset.to_inner_uid(u) for u in users])
I_idx = np.array([trainset.to_inner_iid(m) for m in items])
scores_svd_dot = algo.pu[U_idx] @ algo.qi[I_idx].T
item_vec = algo.qi[I_idx]
nrm = item_vec / np.linalg.norm(item_vec, axis=1, keepdims=True)
S_cos = nrm @ nrm.T
np.fill_diagonal(S_cos, 0)
scores_svd_cos = R @ S_cos

# ---------- 汇总对比 ----------
rows = [
    ('A. ItemCF（共现相似度）', scores_cf),
    ('B. ItemMF-ALS（隐式，手写）', scores_als),
    ('C. ItemMF-SVD dot（试错）', scores_svd_dot),
    ('D. ItemMF-SVD cos（试错）', scores_svd_cos),
]
print(f'===== 四路召回对比（同批同口径）=====')
results = {}
for name, s in rows:
    r, n, _ = evaluate(s)
    results[name] = (r, n)
    print(f'{name}: Recall@{K}={r:.4f}  NDCG@{NDCG_K}={n:.4f}')

# ---------- 融合：A+B 归一化后加权 ----------
A = norm_per_user(scores_cf)
B = norm_per_user(scores_als)
print(f'\n===== 加权融合 A+B（扫权重）=====')
best = (0.0, None)
for w in [0.2, 0.4, 0.5, 0.6, 0.8]:
    r, n, _ = evaluate(w * A + (1 - w) * B)
    mark = '  <-- 当前最优' if r > best[0] else ''
    if r > best[0]:
        best = (r, (w, n))
    print(f'w(ItemCF)={w}: Recall@{K}={r:.4f}  NDCG@{NDCG_K}={n:.4f}{mark}')

r_best, (w_best, n_best) = best
print(f'\n最优融合 w={w_best}: Recall@{K}={r_best:.4f}  NDCG@{NDCG_K}={n_best:.4f}')
print(f'对比 ItemCF 单路: 0.1636 / 0.5234')
print(f'运行耗时：{time.time() - t0:.1f} 秒')
