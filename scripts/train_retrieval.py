"""
Train Two-Tower retrieval model + build FAISS index.
训练两塔召回模型，然后建立 FAISS 向量索引用于快速检索。

流程：
  1. 训练两塔模型（用 BPR Loss）
  2. 提取所有商品的向量 → 存入 FAISS
  3. 对测试用户：用户向量 → FAISS 搜索 → Top-K 商品
  4. 评估 Precision@10 / Recall@10 / NDCG@10
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import faiss
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.two_tower import TwoTowerModel
from src.dataset   import TwoTowerDataset
from src.metrics   import evaluate

DATA_DIR   = Path("data/processed")
MODEL_PATH = Path("data/processed/two_tower.pt")

# 超参数
EMBEDDING_DIM = 32
BATCH_SIZE    = 4096
EPOCHS        = 5
LR            = 1e-3
K             = 10
EVAL_USERS    = 10000

# ── 1. 加载数据 ────────────────────────────────────────────────────────────────

print("Loading data... / 加载数据...")
train = pd.read_parquet(DATA_DIR / "train.parquet")
test  = pd.read_parquet(DATA_DIR / "test.parquet")

num_users = int(train["user_idx"].max()) + 1
num_items = int(train["item_idx"].max()) + 1
print(f"  Users: {num_users:,}  |  Items: {num_items:,}")

# ── 2. 建 Dataset 和 DataLoader ────────────────────────────────────────────────
# DataLoader 会自动把数据切成 batch，并行加载（num_workers）

print("\nBuilding dataset... / 准备训练数据...")
dataset = TwoTowerDataset(train, num_items, num_negatives=4)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"  {len(dataset):,} samples, {len(loader):,} batches per epoch")

# ── 3. 初始化模型 ──────────────────────────────────────────────────────────────

device = torch.device("cpu")   # MPS 在此版本 PyTorch 有兼容问题，用 CPU 稳定
print(f"\nDevice: {device}")

model     = TwoTowerModel(num_users, num_items, EMBEDDING_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── 4. 训练 ───────────────────────────────────────────────────────────────────

print(f"\nTraining Two-Tower ({EPOCHS} epochs)... / 训练中...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_idx, (users, pos_items, neg_items) in enumerate(loader):
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        optimizer.zero_grad()
        loss = model(users, pos_items, neg_items)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"  Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

# ── 5. 建 FAISS 索引 ──────────────────────────────────────────────────────────
# 把所有商品的向量算出来，存进 FAISS
# 之后用户查询时，FAISS 能在毫秒内找出最相似的 Top-200 商品

print("\nBuilding FAISS index... / 建立向量索引...")
model.eval()

all_item_ids = torch.arange(num_items, device=device)

# 分批提取（一次全取可能内存不够）
item_vecs = []
with torch.no_grad():
    for i in range(0, num_items, 4096):
        batch = all_item_ids[i:i+4096]
        vecs  = model.get_item_embedding(batch).cpu().numpy()
        item_vecs.append(vecs)

item_vecs = np.vstack(item_vecs).astype(np.float32)
print(f"  Item vectors shape: {item_vecs.shape}")

# FAISS IndexFlatIP = 内积（Inner Product）搜索
# 向量已 L2 归一化，所以内积 = 余弦相似度
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(item_vecs)
print(f"  FAISS index size: {index.ntotal:,} items")

faiss.write_index(index, str(DATA_DIR / "faiss_item.index"))
print(f"  Saved → data/processed/faiss_item.index")

# ── 6. 生成推荐 + 评估 ────────────────────────────────────────────────────────

print(f"\nEvaluating on {EVAL_USERS:,} sampled users... / 评估中...")

ground_truth  = test.groupby("user_idx")["item_idx"].apply(set).to_dict()
test_user_ids = list(ground_truth.keys())
np.random.seed(42)
sampled_users = np.random.choice(test_user_ids, size=min(EVAL_USERS, len(test_user_ids)), replace=False)

# 训练集里每个用户买过的商品（推荐时要排除）
train_user_items = train.groupby("user_idx")["item_idx"].apply(set).to_dict()

recommended = {}
with torch.no_grad():
    user_tensor = torch.tensor(sampled_users, dtype=torch.long, device=device)
    user_vecs   = model.get_user_embedding(user_tensor).cpu().numpy().astype(np.float32)

# FAISS 一次性搜索所有用户（批量搜索比逐个快很多）
_, top_indices = index.search(user_vecs, K + 50)  # 多取 50 个，过滤后还剩 K 个

for i, user_idx in enumerate(sampled_users):
    seen   = train_user_items.get(int(user_idx), set())
    recs   = [item for item in top_indices[i] if item not in seen][:K]
    recommended[int(user_idx)] = recs

sampled_gt = {int(u): ground_truth[u] for u in sampled_users if u in ground_truth}
scores     = evaluate(recommended, sampled_gt, k=K)

print("\n" + "="*50)
print(f"  Two-Tower Results (K={K}, {EVAL_USERS:,} users)")
print("="*50)
for metric, value in scores.items():
    print(f"  {metric}: {value}")
print("="*50)
print(f"\n  ALS baseline was → Precision@10: 0.0022 | Recall@10: 0.0223")
improvement = (scores[f"Recall@{K}"] - 0.0223) / 0.0223 * 100
print(f"  Two-Tower improvement: {improvement:+.1f}% in Recall@10")
