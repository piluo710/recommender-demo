# SASRec 序列推荐：实现、对照与归因

在 `recommender-demo` 原有 ItemCF 召回 + DeepFM 精排的基础上，补一个**生成式序列推荐**模块，
并建立同口径对照，回答一个具体问题：**在 ml-1m 这个规模上，序列模型能不能超过传统协同过滤？**

结论先说：**不能**。但这个负结论是实验做出来的，不是猜的，过程里有三个可讲的技术点。

---

## 1. 一句话结论

| 模型 | Recall@20 | NDCG@10 |
|---|---|---|
| ItemCF（全历史，最近 50 行为） | **0.1336** | **0.0384** |
| ItemCF（全历史） | 0.1017 | 0.0301 |
| 纯流行度（全局最热 20） | 0.0723 | 0.0189 |
| **SASRec（最优配置）** | 0.0695 | 0.0184 |
| SASRec + ItemCF 融合 | ≤ 0.1348（任何权重都下降） | — |

SASRec 的水平约等于"只押热门"，比 ItemCF 低 48%，融合也无增益。

---

## 2. 为什么必须自己重跑基线

原项目 `itemcf_recall.py` 用的是 **80/20 随机交互划分**（简历上的 0.1636 来自这里）。
序列模型必须按时间切分——用随机划分的话，训练序列会看到"未来"的交互，信息直接泄漏。

所以这里统一改成 **leave-one-out**：每个用户按时间排序，最后 1 个交互做 test，
倒数第 2 个做 valid，其余做训练。**再在同样切分下重跑 ItemCF 和流行度**，三者在同一张表里才可比。

> 简历上的老数字（ItemCF 0.1636 / 融合 0.1941 / DeepFM 0.4758 等）是 80/20 口径，不要和本表混用。

---

## 3. 文件与运行方式

```
sasrec/
├── dataset.py              # LOO 数据加载（rating>=4，item id 从 1 起，0 留给 padding）
├── model.py                # SASRec：item+pos embedding, 2 层 pre-LN causal attention
├── train.py                # 训练 + 评估（参数化，含早停）
├── baseline_itemcf_loo.py  # 同口径 ItemCF（--recent N 只取最近 N 个行为）
├── baseline_popularity.py  # 纯流行度基线（校准任务难度）
├── fuse_sasrec.py          # 序列 + CF 融合权重扫描
├── diag.py / diag2.py / diag3.py       # 三轮 bug 定位（保留作证据）
├── diag_collapse.py        # 塌缩诊断（是否退化成只推热门）
└── result_*.json           # 各配置结果留档
```

```powershell
# 环境：D:\download\miniconda3\envs\stats\python.exe（torch 2.5.1+cpu）
$env:PYTHONIOENCODING='utf-8'
cd D:\code\project\recommender-demo

# 最优 SASRec 配置
& 'D:\download\miniconda3\envs\stats\python.exe' sasrec\train.py `
  --max_len 50 --neg 10 --neg_mode uniform --dropout 0.3 `
  --lr 5e-4 --epochs 40 --patience 8 --window random

# 三个基线（务必在同一 LOO 切分下）
python sasrec\baseline_itemcf_loo.py                 # 全历史 0.1017
python sasrec\baseline_itemcf_loo.py --recent 50     # 最近50 0.1336
python sasrec\baseline_popularity.py                 # 0.0723
```

数据：`data/ml-1m/ratings.dat`，过滤 rating>=4，可用用户 6016 / 6038，物品 3533，训练序列平均长度 93.6。

---

## 4. 三个值得讲的技术点

### 4.1 padding 全屏蔽导致 NaN，召回从 0.035 修到 0.067

第一版跑完 loss 正常收敛（5.59 → 1.09），但 test 只有 R@20 = 0.0347。

三轮诊断：
1. `diag.py` 小样本训练 → 训练集命中只有 2%，模型只会押热门
2. `diag2.py` 换成 sampled softmax（1 正 + 10 负）→ 命中 0.04，否掉"负采样太弱"
3. `diag3.py` 只喂 3 个用户、打印 target 的排名和分数 → **`score = nan`**

根因：序列左 padding，padding 位置的 query 在 attention 里被 `key_padding_mask` **连自己一起屏蔽**，
softmax 分母为 0 → NaN；第 1 层 NaN 后，第 2 层的有效位置通过 value 被污染 → 整个用户预测作废。
序列短于 max_len 的用户（占多数）全部失效，所以整体命中率只有 3%。

修法两处（`model.py`）：
```python
h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)   # 每层 attention 后切断 NaN
x = torch.nan_to_num(x, nan=0.0)                          # 输出前双保险
```
修复后 0.0347 → 0.0672，翻倍。

### 4.2 位置编码长度必须训练/推理对齐

原实现训练时输入序列长 49（`seq[:, :-1]`），位置编码只训到 0–48；
推理用长度 50 的最后一位（位置 49）——**这个位置编码从没训练过，是随机噪声**。

修法：训练时 `pad_seq(train_seq, max_len + 1)` 再切 `x = seqs[:, :-1] / y = seqs[:, 1:]`，
让训练输入长度 = max_len，与推理索引一致。

### 4.3 流失的 item embedding 缩放（论文细节）

`model.py` 里必须是：
```python
x = emb_drop(item_emb(seq) * (d ** 0.5) + pos_emb(pos))
```
漏掉 `sqrt(d)` 缩放时，位置编码的尺度会淹没物品信号，模型倾向学"位置模式"而不是物品偏好。

---

## 5. 负采样策略对比（三选一，只有一个能用）

| 负采样 | 配置 | Recall@20 | 说明 |
|---|---|---|---|
| 1:1 BCE | L50 | 0.0672 | 论文原方案，只够学到"押热门" |
| 1:10 均匀 | L100 | 0.0665 | 加大负样本数没有提升 |
| 1:10 流行度加权 | L50 | **0.0007** | 崩：热门 item 全被当负样本，模型学会"给热门打低分"，而用户看的新片大多是热门 |

流行度负采样的失败是本次实验最有价值的教训之一：**负采样难度的调整要适度**，
把全局最热的 item 全部推入负样本，等于让模型和真实分布对抗。

---

## 6. 为什么打不过 ItemCF（归因，面试重点）

不是实现错了（3 用户小样本能训练到 loss 0.02、target rank=1，说明模型有能力过拟合）。
是**数据规模和信号类型**的问题：

1. **数据太稀疏**：rating>=4 过滤后用户平均只有 93.6 个交互、物品 3533 个，
   每个物品平均被 160 个用户接触。神经网络学不出稳定的 item-item 转移结构。
2. **共现统计在小数据上是天然优势**：ItemCF 直接从"谁和谁被同一个人看过"精确估计，
   不需要学习，样本少时反而更准。这也是工业界中小规模场景仍大量用 CF 的原因。
3. **序列信号本身弱**：ml-1m 是用户集中评分的数据集，同一用户很多交互的时间戳集中在同一 session，
   "顺序"信息含量低（不像电商点击流有明确的意图递进）。
4. **信息量不是瓶颈**：ItemCF 只用最近 50 个行为就有 0.1336，比全历史还高——
   说明"最近 50 个"已经够用，SASRec 的窗口截断不是主因。

---

## 7. 面试怎么讲（4 点）

1. **为什么换评估口径**：序列模型必须按时间切分，否则训练集看到未来交互造成泄漏；
   换口径后必须在同一口径重跑所有基线，否则对比无意义。
2. **NaN 定位过程**：loss 正常但指标异常 → 小样本诊断 → 打印分数发现 nan →
   定位到 padding query 被全屏蔽导致 softmax 分母为 0 → 用 masked_fill 切断传播。
3. **负采样的取舍**：均匀负采样 → 只学热门；流行度加权 → 直接崩。
   说明"负样本难度"和"真实评测分布"要匹配。
4. **诚实的负结论**：在小规模稀疏数据上，序列模型打不过共现统计；
   如果数据量上去（或换成有点击流的数据集如 RetailRocket/Taobao），结论可能反转。
   **能做公平对照并解释原因，比刷一个好看的数字更重要**。

---

## 8. 简历可以怎么写

推荐（诚实 + 体现工程能力）：

> **序列推荐补做（SASRec）**：自建 leave-one-out 评估体系，复现 SASRec 并做同口径对照；
> 定位并修复了 padding 掩码全屏蔽导致的全用户 NaN 失效问题（Recall@20 0.035→0.067）；
> 对比 3 种负采样策略与 2 种序列窗口，结论：ml-1m 稀疏数据下序列模型（0.070）未超过
> ItemCF（0.134），融合无增益。

面试被追问时补一句：**"我没改简历上的老数字，因为那是 80/20 口径；这套是 LOO 口径，两套不能混比。"**
