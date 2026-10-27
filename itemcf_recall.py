# ItemCF 召回基线（Day 1）
# 数据：MovieLens-1M（ratings.dat：UserID::MovieID::Rating::Timestamp）
# 跑法（在 recommender-demo 目录下）：
#   & 'D:\download\miniconda3\envs\stats\python.exe' itemcf_recall.py
# 输出：Recall@20、NDCG@10、运行耗时（这些数字就是简历上的 XX%）

import time
import numpy as np
import pandas as pd

DATA = r'data\ml-1m\ratings.dat'
K = 20          # 召回 20 个候选
NDCG_K = 10     # 只评估前 10 个位置的排序质量
t0 = time.time()

# ---------- 1. 读数据 ----------
# 列名：uid=用户ID, mid=电影ID, rating=1~5分, ts=时间戳
df = pd.read_csv(DATA, sep='::', header=None, engine='python',
                 names=['uid', 'mid', 'rating', 'ts'])
print(f'原始交互数：{len(df)}，用户 {df.uid.nunique()}，电影 {df.mid.nunique()}')

# ---------- 2. 转成隐式反馈 ----------
# 推荐系统里常见的做法：把 4 分以上当成一次"喜欢"（正样本），1~3 分忽略。
# 面试问"为什么用阈值 4"就答：显式评分转隐式喜欢，和广告场景"曝光→点击"
# 的二元标签对齐；想更严谨还可以做"评分>=用户均分"之类的变体。
df = df[df['rating'] >= 4]
print(f'rating>=4 的交互数：{len(df)}')

# ---------- 3. 划分训练/测试 ----------
# 按交互随机 80/20。注意：必须用【同一次】随机抽样得到互补的两份，
# 否则训练/测试会互相"偷看"（数据泄漏），指标会虚高。
rng = np.random.default_rng(42)
is_train = rng.random(len(df)) < 0.8
train, test = df[is_train], df[~is_train]
print(f'训练交互：{len(train)}，测试交互：{len(test)}')

# ---------- 4. 建 user-item 二值矩阵 ----------
# 行=用户，列=电影，R[u, j]=1 表示用户 u 喜欢过电影 j。
# 只保留训练集里出现的用户和电影（测试集里没见过的电影无法推荐，略过）。
users = np.sort(train.uid.unique())
items = np.sort(train.mid.unique())
uid2i = {u: i for i, u in enumerate(users)}
mid2j = {m: j for j, m in enumerate(items)}
R = np.zeros((len(users), len(items)), dtype=np.float32)
for u, m in zip(train.uid.values, train.mid.values):
    R[uid2i[u], mid2j[m]] = 1.0
print(f'用户-物品矩阵：{R.shape}')

# ---------- 5. ItemCF 核心：物品间余弦相似度 ----------
# 每个物品 j 看成一个"用户向量"（谁喜欢过它）。余弦相似度 = 内积 / 各自模长。
# 先按列归一化，再算内积：S = norm.T @ norm，一步得到所有物品两两相似度。
norm = R / np.linalg.norm(R, axis=0, keepdims=True)  # 每列（电影）归一化
S = norm.T @ norm                                     # item-item 相似度矩阵
np.fill_diagonal(S, 0)                                # 自己和自己不算

# ---------- 6. 给每个用户打分 ----------
# ItemCF 打分逻辑：用户 u 对候选电影 j 的得分 = 用户喜欢过的所有电影 i 与 j
# 的相似度之和（权重就是是否喜欢，1 或 0）。用矩阵乘法一次算完：
# score[u, j] = R[u, :] · S[:, j]
scores = R @ S

# 把训练集里已经喜欢过的电影分数压成 -inf，避免"把看过的再推一遍"，
# 这是召回评估的标准做法（排除已知交互）。
scores[R == 1] = -np.inf

# ---------- 7. 评估：Recall@20 和 NDCG@10 ----------
def ndcg_at_k(rel: np.ndarray, k: int) -> float:
    """rel: 0/1 数组（按推荐位置排序），返回前 k 位的 NDCG。"""
    rel = rel[:k]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    # 理想排序：所有正样本排在前面
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

recall_hits, ndcg_sum, eval_users, eval_denom = 0, 0.0, 0, 0
for u, grp in test.groupby('uid'):
    if u not in uid2i:
        continue
    # 该用户在测试期喜欢过的电影（只算训练矩阵里见过的）
    test_items = [mid2j[m] for m in grp.mid.values if m in mid2j]
    if not test_items:
        continue
    # 取分数最高的 K 个（用 argpartition 避免全排序，快）
    row = scores[uid2i[u]]
    top_k = np.argpartition(row, -K)[-K:]
    top_k = top_k[np.argsort(row[top_k])[::-1]]
    hits = sum(1 for j in top_k if j in test_items)
    recall_hits += hits
    rel = np.array([1 if j in test_items else 0 for j in top_k])
    ndcg_sum += ndcg_at_k(rel, NDCG_K)
    eval_users += 1
    eval_denom += len(test_items)

# Recall = 命中数 / 测试正样本总数（分子分母都只算参与评估的用户，口径一致）
recall = recall_hits / eval_denom if eval_denom else 0.0
ndcg = ndcg_sum / eval_users if eval_users else 0.0

print(f'\n===== ItemCF 基线（Day 1）=====')
print(f'评估用户数：{eval_users}（测试集有正样本的用户）')
print(f'Recall@{K}：{recall:.4f}')
print(f'NDCG@{NDCG_K}：{ndcg:.4f}')
print(f'运行耗时：{time.time() - t0:.1f} 秒')
