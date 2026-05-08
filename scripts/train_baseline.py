"""
ALS Baseline: Alternating Least Squares for Implicit Feedback
ALS 基线模型：专为稀疏隐式反馈设计的矩阵分解

为什么用 ALS 不用 SVD：
  SVD 把空白格子当"未知"→ 学不到东西
  ALS 把空白格子当"不感兴趣（低置信度）"→ 更符合真实场景

置信度公式：confidence = 1 + alpha × rating
  5星评价 → confidence = 1 + 40×5 = 201（非常确定用户喜欢）
  1星评价 → confidence = 1 + 40×1 = 41（至少买过，有一定信号）
  没买过  → confidence = 0（不感兴趣）
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.metrics import evaluate

DATA_DIR = Path("data/processed")

# 超参数
FACTORS    = 64    # 用户/商品向量的维度，越大越精细但越慢
ITERATIONS = 20    # 训练轮数
ALPHA      = 40    # 置信度放大系数
K          = 10    # 推荐数量
EVAL_USERS = 10000 # 评估时抽样的用户数（86万全跑太慢）

# ── 1. 加载数据 ────────────────────────────────────────────────────────────────

print("Loading data... / 加载数据...")
train = pd.read_parquet(DATA_DIR / "train.parquet")
test  = pd.read_parquet(DATA_DIR / "test.parquet")

num_users = int(train["user_idx"].max()) + 1
num_items = int(train["item_idx"].max()) + 1
print(f"  Users: {num_users:,}  |  Items: {num_items:,}")
print(f"  Train: {len(train):,} interactions")

# ── 2. 构建置信度矩阵 ──────────────────────────────────────────────────────────
# ALS 输入不是评分，而是置信度：confidence = 1 + alpha × rating
# 形状：(num_items, num_users)  ← implicit 库要求 item × user

print("\nBuilding confidence matrix... / 构建置信度矩阵...")
confidence = (1 + ALPHA * train["rating"].values).astype(np.float32)

# user × item 矩阵（行=用户，列=商品）
user_item_matrix = csr_matrix(
    (confidence, (train["user_idx"].values, train["item_idx"].values)),
    shape=(num_users, num_items)
)
print(f"  Matrix shape (users × items): {user_item_matrix.shape}")

# ── 3. 训练 ALS 模型 ───────────────────────────────────────────────────────────
# 模型学习两组向量：
#   user_factors: (num_users, 64)  每个用户的"口味向量"
#   item_factors: (num_items, 64)  每个商品的"特征向量"
# 内积越大 = 用户越可能喜欢这个商品

print(f"\nTraining ALS (factors={FACTORS}, iterations={ITERATIONS})...")
print("预计 1-3 分钟 / Estimated 1-3 minutes...")

model = AlternatingLeastSquares(
    factors=FACTORS,
    iterations=ITERATIONS,
    regularization=0.01,
    use_gpu=False,
)
model.fit(user_item_matrix)
print("Training complete! / 训练完成！")

# ── 4. 生成推荐（抽样评估）────────────────────────────────────────────────────
# 86万用户全跑太慢，随机抽 10000 个做评估，统计上已经够代表性

print(f"\nGenerating recommendations for {EVAL_USERS:,} sampled users...")

# 构建测试集 ground truth
ground_truth = (
    test.groupby("user_idx")["item_idx"]
    .apply(set)
    .to_dict()
)

# 只评估测试集里有的用户
test_user_ids = list(ground_truth.keys())
np.random.seed(42)
sampled_users = np.random.choice(
    test_user_ids,
    size=min(EVAL_USERS, len(test_user_ids)),
    replace=False
)

recommended = {}
for i, user_idx in enumerate(sampled_users):
    if i % 1000 == 0:
        print(f"  {i}/{len(sampled_users)} users done...")

    # recommend() 自动排除训练集里已买过的商品
    item_ids, _ = model.recommend(
        user_idx,
        user_item_matrix[user_idx],
        N=K,
        filter_already_liked_items=True,
    )
    recommended[user_idx] = item_ids.tolist()

# ── 5. 评估 ───────────────────────────────────────────────────────────────────

print("\nEvaluating... / 评估中...")
sampled_ground_truth = {u: ground_truth[u] for u in sampled_users if u in ground_truth}
scores = evaluate(recommended, sampled_ground_truth, k=K)

print("\n" + "="*45)
print(f"  ALS Baseline Results (K={K}, {EVAL_USERS:,} users)")
print("="*45)
for metric, value in scores.items():
    print(f"  {metric}: {value}")
print("="*45)
print("\n解读：")
print(f"  Precision@10 = {scores[f'Precision@{K}']}")
print(f"  → 平均每推荐10个商品，有 {scores[f'Precision@{K}']*10:.2f} 个是用户真正买过的")
print(f"  Recall@10 = {scores[f'Recall@{K}']}")
print(f"  → 用户测试集里的商品，平均有 {scores[f'Recall@{K}']*100:.1f}% 被推荐出来了")
