# recommender-demo

推荐系统个人项目（简历第二项目）：MovieLens-1M 上实现「多路召回 → 排序」链路，产出可讲可查的指标数字。

## 结构

```
recommender-demo/
├── itemcf_recall.py   # Day1：ItemCF 召回基线（Recall@20 / NDCG@10）
├── fuse_recall.py     # Day2：四路召回对比（ItemCF / 手写 ALS / 试错）+ 加权融合
├── deepfm_rank.py     # Day3：DeepFM 精排（候选池内重排实验，PyTorch 手写）
├── notes.md           # 数字账本（简历 XX% 从这里取）
├── data/ml-1m/        # MovieLens-1M（ratings/movies/users）
```

## 跑法

```powershell
cd D:\code\project\recommender-demo
& 'D:\download\miniconda3\envs\stats\python.exe' itemcf_recall.py   # 单路 ItemCF
& 'D:\download\miniconda3\envs\stats\python.exe' fuse_recall.py     # 四路对比 + 融合
& 'D:\download\miniconda3\envs\stats\python.exe' deepfm_rank.py     # DeepFM 精排实验
```

数据说明：MovieLens-1M 从 <https://files.grouplens.org/datasets/movielens/ml-1m.zip> 下载后解压到 `data/ml-1m/`（仓库不包含数据文件）。

## 当前指标（2026-10-28 Day2，同批同口径）

| 方法 | Recall@20 | NDCG@10 |
|---|---|---|
| ItemCF（基线） | 0.1636 | 0.5234 |
| ItemMF-ALS（手写隐式） | 0.1491 | 0.4577 |
| **加权融合（w=0.4）** | **0.1941** | **0.5644** |

召回融合 Recall@20 +18.6%（0.164→0.194）。详细口径与试错记录见 `notes.md`。

## 排序（2026-10-30 Day3）

DeepFM 精排（PyTorch 手写：Embedding + FM 二阶交叉 + MLP，特征含召回分）在融合召回 top-50 候选池内重排：池内 NDCG@10 = 0.4758（召回直接出 0.5643）、池内 AUC = 0.59。MovieLens 隐式任务协同过滤信号已很强，有限特征下排序增益不明显——如实记录，不夸大。
