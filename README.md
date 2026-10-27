# recommender-demo

推荐系统个人项目（简历第二项目）：MovieLens-1M 上实现「多路召回 → 排序」链路，产出可讲可查的指标数字。

## 结构

```
recommender-demo/
├── itemcf_recall.py   # Day1：ItemCF 召回基线（Recall@20 / NDCG@10）
├── notes.md           # 数字账本（简历 XX% 从这里取）
├── data/ml-1m/        # MovieLens-1M（ratings/movies/users）
└── （待加）itemmf_recall.py / fuse_recall.py / deepfm_sort.py
```

## 跑法

```powershell
cd D:\code\project\recommender-demo
& 'D:\download\miniconda3\envs\stats\python.exe' itemcf_recall.py
```

数据说明：MovieLens-1M 从 <https://files.grouplens.org/datasets/movielens/ml-1m.zip> 下载后解压到 `data/ml-1m/`（仓库不包含数据文件）。

## 当前指标（2026-10-27 Day1）

| 方法 | Recall@20 | NDCG@10 |
|---|---|---|
| ItemCF（基线） | 0.1636 | 0.5234 |

详细口径见 `notes.md`。
