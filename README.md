# Amazon Beauty Recommender

A two-stage retrieval-ranking recommendation system built on 3.7M real Amazon Beauty product reviews.

> 基于 370 万条亚马逊美妆真实评论构建的两阶段召回+精排推荐系统。

---

## Demo

```
GET /recommend?user_id=A1B2C3

→ [
    {"rank": 1, "product_id": "B001234", "title": "CeraVe Moisturizing Cream", "score": 0.94},
    {"rank": 2, "product_id": "B005678", "title": "La Roche-Posay Serum",      "score": 0.91},
    ...
  ]
```

---

## Architecture / 架构

```
Raw Data (3.7M reviews)
        ↓
  Data Cleaning Layer
  - Remove nulls & duplicates
  - Detect fake reviews (burst pattern)
  - Temporal train/test split
        ↓
┌──────────────────────────────────┐
│       Two-Stage Pipeline         │
│                                  │
│  Stage 1 — Retrieval             │
│  Two-Tower Model + FAISS ANN     │
│  user_id → Top-200 candidates    │
│              ↓                   │
│  Stage 2 — Ranking               │
│  MLP + text features             │
│  200 candidates → Top-10 result  │
└──────────────────────────────────┘
        ↓
  FastAPI Service
  GET /recommend?user_id=xxx
```

```
原始数据 (371万条评论)
      ↓
  数据清洗层
  - 去空值和重复值
  - 识别刷单评论（短时间内爆发式好评）
  - 按时间切分训练集/测试集
      ↓
  ┌─────────────────────────────┐
  │         两阶段推荐系统        │
  │                             │
  │  第一阶段：召回               │
  │  两塔模型 + FAISS 近邻搜索    │
  │  用户ID → 候选商品 Top-200    │
  │           ↓                 │
  │  第二阶段：精排               │
  │  MLP + 文本特征              │
  │  200个候选 → Top-10 结果     │
  └─────────────────────────────┘
      ↓
  FastAPI 接口服务
```

---

## Results / 实验结果

> Results will be updated as training completes.
> 实验结果训练完成后持续更新。

| Model | RMSE | Precision@10 | Recall@10 | NDCG@10 |
|-------|------|--------------|-----------|---------|
| SVD Baseline | TBD | TBD | TBD | TBD |
| Two-Tower + MLP | TBD | TBD | TBD | TBD |

---

## Tech Stack / 技术栈

| Module / 模块 | Tool / 工具 |
|---------------|-------------|
| Data processing / 数据处理 | pandas, pyarrow |
| Model training / 模型训练 | PyTorch |
| Vector search / 向量检索 | FAISS |
| Text features / 文本特征 | TF-IDF → sentence-transformers |
| Evaluation / 评估 | Custom Precision@K, Recall@K, NDCG@K |
| API serving / 接口服务 | FastAPI |

---

## Project Structure / 项目结构

```
amazon-beauty-rec/
│
├── data/
│   ├── raw/                  # Raw downloaded files / 原始下载文件
│   └── processed/            # Cleaned data / 清洗后数据
│       ├── train.parquet
│       ├── test.parquet
│       └── item_meta.parquet
│
├── src/
│   ├── preprocess.py         # Data cleaning / 数据清洗
│   ├── dataset.py            # PyTorch Dataset class
│   ├── two_tower.py          # Two-Tower retrieval model / 两塔召回模型
│   ├── ranking.py            # MLP ranking model / 精排模型
│   ├── faiss_index.py        # FAISS vector index / 向量索引
│   ├── metrics.py            # Evaluation metrics / 评估指标
│   └── api.py                # FastAPI service / 接口服务
│
├── notebooks/
│   ├── 01_eda.ipynb          # Data exploration / 数据探索分析
│   ├── 02_baseline.ipynb     # SVD baseline
│   └── 03_two_stage.ipynb    # Full two-stage system / 完整两阶段系统
│
├── scripts/
│   ├── train_retrieval.py    # Train Two-Tower model / 训练召回模型
│   ├── train_ranking.py      # Train ranking model / 训练精排模型
│   └── build_index.py        # Build FAISS index / 建立向量索引
│
├── requirements.txt
└── README.md
```

---

## Roadmap / 开发计划

- [x] Project design & README / 项目设计
- [ ] **Week 1** — Data download, cleaning, EDA / 数据下载、清洗、探索分析
- [ ] **Week 2** — SVD baseline + evaluation metrics / SVD 基线 + 评估指标
- [ ] **Week 3** — Two-Tower retrieval + FAISS index / 两塔召回 + FAISS 索引
- [ ] **Week 4** — MLP ranking + FastAPI serving / MLP 精排 + FastAPI 服务
- [ ] **Week 4.5** — README polish + GitHub cleanup / 完善文档

---

## Dataset / 数据集

**Amazon Product Reviews — Beauty Category**
- Source: [UCSD Amazon Review Data](https://nijianmo.github.io/amazon/index.html)
- Reviews: 3,713,939
- Users: ~1.2M
- Items: ~32K
- Time span: 1996–2018

> Data is not included in this repo. See `data/raw/download.sh` for instructions.
> 数据未包含在仓库中，下载方式见 `data/raw/download.sh`。

---

## Quick Start / 快速开始

```bash
# 1. Clone
git clone https://github.com/96528025/amazon-beauty-rec.git
cd amazon-beauty-rec

# 2. Install dependencies / 安装依赖
pip install -r requirements.txt

# 3. Download data / 下载数据
bash data/raw/download.sh

# 4. Preprocess / 数据清洗
python src/preprocess.py

# 5. Train retrieval model / 训练召回模型
python scripts/train_retrieval.py

# 6. Train ranking model / 训练精排模型
python scripts/train_ranking.py

# 7. Start API / 启动服务
uvicorn src.api:app --reload
# → http://localhost:8000/recommend?user_id=A1B2C3
```

---

## What Makes This Non-Trivial / 为什么这个项目不是 toy

- **Real-world scale**: 3.7M reviews that don't fit in memory — chunked loading required
- **Realistic evaluation**: temporal train/test split (not random), mimicking production conditions
- **Cold-start handling**: strategy for users with < 3 interactions
- **Two-stage design**: mirrors architecture used at TikTok, YouTube, and Amazon
- **End-to-end serving**: not just a notebook — a working REST API

---

*Built by Freja Ren · [GitHub](https://github.com/96528025)*
