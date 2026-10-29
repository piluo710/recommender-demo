# DeepFM 精排（Day3，v1）——在融合召回候选池内重排，补齐简历第三组数字
# 口径：与 Day1/2 完全一致（MovieLens-1M，rating>=4 为正，seed=42 随机 80/20）
# 流程：
#   1) 融合召回（ItemCF + ALS，w=0.4）为每个测试用户生成 top-50 候选池
#   2) DeepFM（PyTorch 手写：一阶 + FM 二阶 + Deep MLP）在候选池内重排
#   3) 同池对比 NDCG@10：召回直接出 top-10  vs  DeepFM 重排 top-10；并报告 AUC
# 跑法：& 'D:\download\miniconda3\envs\stats\python.exe' deepfm_rank.py
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

DATA = r'data\ml-1m\ratings.dat'
MOVIES = r'data\ml-1m\movies.dat'
SEED, POOL_K, NDCG_K, NEG = 42, 50, 10, 4
ALS_F, ALS_ALPHA, ALS_REG, ALS_ITERS = 64, 40.0, 0.1, 15
EMB_DIM, HIDDEN = 32, [256, 128]
EPOCHS, BATCH, LR = 10, 8192, 1e-3
t0 = time.time()

# ---------- 数据（与 fuse_recall.py 完全一致） ----------
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

# ---------- 电影类型（genre 多热） ----------
genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
G = {g: k for k, g in enumerate(genres)}
movies = pd.read_csv(MOVIES, sep='::', header=None, engine='python',
                     names=['mid', 'title', 'genre'], encoding='latin-1')
genre_vec = {}
for _, r in movies.iterrows():
    m = r['mid']
    if m in mid2j:
        v = np.zeros(len(genres), dtype=np.float32)
        for g in str(r['genre']).split('|'):
            if g in G:
                v[G[g]] = 1.0
        genre_vec[mid2j[m]] = v
genre_mat = np.zeros((len(items), len(genres)), dtype=np.float32)
for j, v in genre_vec.items():
    genre_mat[j] = v

# ---------- 稠密特征（只用训练集统计，防泄漏） ----------
item_pop = R.sum(axis=0).astype(np.float32)          # 物品在训练集被交互次数
user_act = R.sum(axis=1).astype(np.float32)          # 用户在训练集交互数
item_pop_n = item_pop / (item_pop.max() + 1e-9)
user_act_n = user_act / (user_act.max() + 1e-9)

# ---------- 融合召回：ItemCF + ALS（复用 Day2 逻辑） ----------
norm_cf = R / np.linalg.norm(R, axis=0, keepdims=True)
S_cf = norm_cf.T @ norm_cf
np.fill_diagonal(S_cf, 0)
scores_cf = R @ S_cf

def implicit_als(R, factors=ALS_F, alpha=ALS_ALPHA, reg=ALS_REG,
                 iters=ALS_ITERS, seed=SEED):
    n_users, n_items = R.shape
    rg = np.random.default_rng(seed)
    X = rg.normal(0, 0.01, (n_users, factors))
    Y = rg.normal(0, 0.01, (n_items, factors))
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

fuse = 0.4 * norm_per_user(scores_cf) + 0.6 * norm_per_user(scores_als)
print(f'[1/4] 融合召回完成（{time.time()-t0:.1f}s），生成测试用户候选池...')

# ---------- 候选池：每测试用户融合 top-POOL_K（排除训练正样本） ----------
test_user_rows = []
pool_scores = []          # 融合分数（池内）
for u, grp in test.groupby('uid'):
    if u not in uid2i:
        continue
    i = uid2i[u]
    row = fuse[i].copy()
    row[R[i] == 1] = -np.inf
    top = np.argpartition(row, -POOL_K)[-POOL_K:]
    top = top[np.argsort(row[top])[::-1]]
    test_user_rows.append((u, i, grp.mid.values))
    pool_scores.append((i, top, row[top]))
print(f'测试用户数：{len(test_user_rows)}')

# ---------- 训练样本：全局负采样（正=训练交互，负=每正 4 个随机未交互），加召回分特征 ----------
n_users, n_items = R.shape
rows_u, rows_m, rows_y, rows_f = [], [], [], []
neg_rng = np.random.default_rng(2026)
for u in users:
    i = uid2i[u]
    pos_items = np.nonzero(R[i])[0]
    for j in pos_items:
        rows_u.append(i); rows_m.append(j); rows_y.append(1)
        rows_f.append(fuse[i][j])
        neg_pool = np.setdiff1d(np.arange(n_items), np.nonzero(R[i])[0])
        if len(neg_pool) < NEG:
            continue
        for _ in range(NEG):
            jn = neg_pool[neg_rng.integers(0, len(neg_pool))]
            rows_u.append(i); rows_m.append(jn); rows_y.append(0)
            rows_f.append(fuse[i][jn])
rows_u = np.array(rows_u); rows_m = np.array(rows_m)
rows_y = np.array(rows_y, dtype=np.float32); rows_f = np.array(rows_f, dtype=np.float32)
print(f'[2/4] 训练样本：{len(rows_y):,}（正 {rows_y.sum():,.0f} / 负 {len(rows_y)-rows_y.sum():,.0f}）')

# 稠密特征拼装（user 活跃度 / item 流行度 / 召回分）
user_act_arr = user_act_n[rows_u]
item_pop_arr = item_pop_n[rows_m]
genre_arr = genre_mat[rows_m]

# ---------- DeepFM 模型 ----------
class DeepFM(nn.Module):
    def __init__(self, n_users, n_items, n_genre, emb_dim=EMB_DIM, hidden=HIDDEN):
        super().__init__()
        self.u_emb = nn.Embedding(n_users, emb_dim)
        self.i_emb = nn.Embedding(n_items, emb_dim)
        self.g_emb = nn.Embedding(n_genre, emb_dim)
        self.u_bias = nn.Embedding(n_users, 1)
        self.i_bias = nn.Embedding(n_items, 1)
        self.linear = nn.Linear(3 + n_genre, 1, bias=True)      # 稠密一阶（含召回分）
        d_in = emb_dim * 3 + 3 + n_genre
        layers = []
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.BatchNorm1d(h)]
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, u, m, g, dense):
        ue, ie = self.u_emb(u), self.i_emb(m)
        ge = g @ self.g_emb.weight                              # genre 多热 → 和向量
        fm = (ue * ie).sum(1) + (ue * ge).sum(1) + (ie * ge).sum(1)
        first = self.u_bias(u).squeeze(1) + self.i_bias(m).squeeze(1) + self.linear(torch.cat([dense, g], dim=1)).squeeze(1)
        deep = self.mlp(torch.cat([ue, ie, ge, dense, g], dim=1)).squeeze(1)
        return first + fm + deep

model = DeepFM(n_users, n_items, len(genres))
opt = torch.optim.Adam(model.parameters(), lr=LR)
lossf = nn.BCEWithLogitsLoss()

# ---------- 训练 ----------
N = len(rows_y)
u_t = torch.tensor(rows_u, dtype=torch.long)
m_t = torch.tensor(rows_m, dtype=torch.long)
g_t = torch.tensor(genre_arr, dtype=torch.float32)
d_t = torch.tensor(np.stack([user_act_arr, item_pop_arr, rows_f], axis=1), dtype=torch.float32)
y_t = torch.tensor(rows_y)
model.train()
for ep in range(EPOCHS):
    perm = torch.randperm(N)
    tot, nb = 0.0, 0
    for s in range(0, N, BATCH):
        idx = perm[s:s + BATCH]
        opt.zero_grad()
        logit = model(u_t[idx], m_t[idx], g_t[idx], d_t[idx])
        loss = lossf(logit, y_t[idx])
        loss.backward()
        opt.step()
        tot += loss.item(); nb += 1
    print(f'epoch {ep+1}/{EPOCHS}  loss={tot/nb:.4f}  ({time.time()-t0:.1f}s)')

# ---------- 评估：候选池内 召回 vs DeepFM 重排 ----------
model.eval()
def ndcg_at_k(rel, k):
    rel = rel[:k]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

ndcg_recall, ndcg_deepfm, cov_pool, n_users_eval = [], [], [], 0
y_true_all, y_score_all = [], []
for idx, (u, i, test_mids) in enumerate(test_user_rows):
    _, pool, s_fuse = pool_scores[idx]
    test_set = {mid2j[m] for m in test_mids if m in mid2j}
    if not test_set:
        continue
    # 池内真实相关
    rel_pool = np.array([1 if j in test_set else 0 for j in pool])
    cov_pool.append(rel_pool.sum() / len(test_set))
    # 召回直接出：按融合分数降序
    order_recall = np.argsort(s_fuse)[::-1]
    ndcg_recall.append(ndcg_at_k(rel_pool[order_recall], NDCG_K))
    # DeepFM 重排
    ut = torch.tensor([i] * len(pool), dtype=torch.long)
    mt = torch.tensor(pool, dtype=torch.long)
    gt = torch.tensor(genre_mat[pool], dtype=torch.float32)
    dt = torch.tensor(np.stack([user_act_n[i].repeat(len(pool)),
                                item_pop_n[pool], s_fuse], axis=1), dtype=torch.float32)
    with torch.no_grad():
        s_deepfm = model(ut, mt, gt, dt).numpy()
    order_deepfm = np.argsort(s_deepfm)[::-1]
    ndcg_deepfm.append(ndcg_at_k(rel_pool[order_deepfm], NDCG_K))
    y_true_all += rel_pool.tolist()
    y_score_all += s_deepfm.tolist()
    n_users_eval += 1

# ---------- 全池 NDCG@10：DeepFM 对全物品打分 top-10（排除训练正样本，与召回全池口径一致） ----------
ndcg_deepfm_full = []
for idx, (u, i, test_mids) in enumerate(test_user_rows):
    test_set = {mid2j[m] for m in test_mids if m in mid2j}
    if not test_set:
        continue
    valid = np.setdiff1d(np.arange(n_items), np.nonzero(R[i])[0])
    scores = np.empty(n_items, dtype=np.float32)
    for s in range(0, len(valid), 8192):
        v = valid[s:s + 8192]
        ut = torch.tensor([i] * len(v), dtype=torch.long)
        mt = torch.tensor(v, dtype=torch.long)
        gt = torch.tensor(genre_mat[v], dtype=torch.float32)
        dt = torch.tensor(np.stack([user_act_n[i].repeat(len(v)),
                                    item_pop_n[v], fuse[i][v]], axis=1), dtype=torch.float32)
        with torch.no_grad():
            scores[v] = model(ut, mt, gt, dt).numpy()
    top = np.argpartition(scores, -NDCG_K)[-NDCG_K:]
    top = top[np.argsort(scores[top])[::-1]]
    rel = np.array([1 if j in test_set else 0 for j in top])
    ndcg_deepfm_full.append(ndcg_at_k(rel, NDCG_K))
ndcg_deepfm_full = np.mean(ndcg_deepfm_full)

ndcg_recall = np.mean(ndcg_recall)
ndcg_deepfm = np.mean(ndcg_deepfm)
auc = roc_auc_score(y_true_all, y_score_all)
print(f'\n[3/4] 同池对比（候选池 top-{POOL_K}，{n_users_eval} 测试用户）')
print(f'  池内测试正样本覆盖率：{np.mean(cov_pool):.3f}')
print(f'  融合召回直接出  NDCG@{NDCG_K} = {ndcg_recall:.4f}')
print(f'  DeepFM 重排     NDCG@{NDCG_K} = {ndcg_deepfm:.4f}  （+{(ndcg_deepfm/ndcg_recall-1)*100:.1f}%）')
print(f'  DeepFM 池内 AUC = {auc:.4f}')
print(f'\n[3b] 全池口径（与召回 0.5644 同口径）')
print(f'  DeepFM 全池打分 NDCG@{NDCG_K} = {ndcg_deepfm_full:.4f}  （+{(ndcg_deepfm_full/0.5644-1)*100:.1f}% vs 融合召回）')
print(f'\n[4/4] 总耗时 {time.time()-t0:.1f}s')
