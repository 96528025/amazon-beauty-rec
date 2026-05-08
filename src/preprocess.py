"""
Data preprocessing pipeline for Amazon Beauty Reviews.
读取原始数据 → 清洗 → 切分训练/测试集 → 保存为 parquet
"""

import gzip
import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(exist_ok=True)


# ── 1. Load raw data ──────────────────────────────────────────────────────────

def load_reviews_streaming(path: Path) -> pd.DataFrame:
    """
    流式读取，只提取需要的 5 个字段，跳过评论正文、图片等大字段。
    370万条全量读取内存占用从 14GB 降到约 200MB。
    """
    records = []
    with gzip.open(path) as f:
        for line in f:
            obj = json.loads(line)
            records.append({
                "user_id":          obj.get("user_id"),
                "parent_asin":      obj.get("parent_asin"),
                "rating":           obj.get("rating"),
                "timestamp":        obj.get("timestamp"),
                "verified_purchase": obj.get("verified_purchase"),
            })
    return pd.DataFrame(records)


def load_meta_streaming(path: Path) -> pd.DataFrame:
    """只提取商品元数据中需要的字段"""
    records = []
    with gzip.open(path) as f:
        for line in f:
            obj = json.loads(line)
            records.append({
                "parent_asin":    obj.get("parent_asin"),
                "title":          obj.get("title"),
                "average_rating": obj.get("average_rating"),
                "rating_number":  obj.get("rating_number"),
                "store":          obj.get("store"),
                "price":          obj.get("price"),
            })
    return pd.DataFrame(records)


print("Loading reviews (streaming, ~5 mins)... / 流式加载评论数据（约5分钟）...")
reviews = load_reviews_streaming(RAW_DIR / "Beauty_and_Personal_Care.jsonl.gz")
print(f"  Raw rows: {len(reviews):,}")

print("Loading metadata... / 加载商品元数据...")
meta = load_meta_streaming(RAW_DIR / "meta_All_Beauty.jsonl.gz")
print(f"  Raw items: {len(meta):,}")


# ── 2. Clean reviews ──────────────────────────────────────────────────────────

print("\nCleaning reviews... / 清洗评论数据...")

# Keep only needed columns
reviews = reviews[["user_id", "parent_asin", "rating", "timestamp", "verified_purchase"]]

# Drop nulls
reviews = reviews.dropna(subset=["user_id", "parent_asin", "rating", "timestamp"])

# Convert timestamp (milliseconds → seconds)
reviews["timestamp"] = reviews["timestamp"] / 1000
reviews["timestamp"] = reviews["timestamp"].astype(int)

# Remove duplicate (user, item) pairs — keep the latest review
reviews = reviews.sort_values("timestamp").drop_duplicates(
    subset=["user_id", "parent_asin"], keep="last"
)

# Filter: keep users with >= 5 reviews, items with >= 10 reviews
# 数据量大了可以用更严格的门槛，保留行为更丰富的用户和商品
user_counts = reviews["user_id"].value_counts()
item_counts = reviews["parent_asin"].value_counts()
reviews = reviews[
    reviews["user_id"].isin(user_counts[user_counts >= 5].index) &
    reviews["parent_asin"].isin(item_counts[item_counts >= 10].index)
]

print(f"  After cleaning: {len(reviews):,} reviews")
print(f"  Users: {reviews['user_id'].nunique():,}")
print(f"  Items: {reviews['parent_asin'].nunique():,}")


# ── 3. Encode user_id and item_id to integers ─────────────────────────────────
# 模型需要整数 ID，这里建立映射表

user2idx = {u: i for i, u in enumerate(reviews["user_id"].unique())}
item2idx = {p: i for i, p in enumerate(reviews["parent_asin"].unique())}

reviews["user_idx"] = reviews["user_id"].map(user2idx)
reviews["item_idx"] = reviews["parent_asin"].map(item2idx)

print(f"\n  num_users = {len(user2idx):,}")
print(f"  num_items = {len(item2idx):,}")


# ── 4. Temporal train/test split ──────────────────────────────────────────────
# 按时间切分：最后一条交互作为测试集，其余为训练集
# 这比随机切分更贴近真实推荐场景

print("\nSplitting train/test by time... / 按时间切分训练/测试集...")

reviews = reviews.sort_values(["user_idx", "timestamp"])
test = reviews.groupby("user_idx").tail(1)
train = reviews.drop(test.index)

print(f"  Train: {len(train):,} rows")
print(f"  Test:  {len(test):,} rows")


# ── 5. Clean metadata ─────────────────────────────────────────────────────────

print("\nCleaning metadata... / 清洗商品元数据...")

meta = meta.dropna(subset=["parent_asin", "title"])
meta = meta.drop_duplicates(subset="parent_asin")

# Only keep items that appear in our filtered reviews
meta = meta[meta["parent_asin"].isin(item2idx.keys())].copy()
meta["item_idx"] = meta["parent_asin"].map(item2idx)

print(f"  Items with metadata: {len(meta):,}")


# ── 6. Save ───────────────────────────────────────────────────────────────────

print("\nSaving to parquet... / 保存数据...")

train.to_parquet(OUT_DIR / "train.parquet", index=False)
test.to_parquet(OUT_DIR / "test.parquet", index=False)
meta.to_parquet(OUT_DIR / "item_meta.parquet", index=False)

# Save ID mappings for inference
pd.DataFrame(list(user2idx.items()), columns=["user_id", "user_idx"]).to_parquet(
    OUT_DIR / "user_map.parquet", index=False
)
pd.DataFrame(list(item2idx.items()), columns=["parent_asin", "item_idx"]).to_parquet(
    OUT_DIR / "item_map.parquet", index=False
)

print("\nDone! / 完成！")
print(f"  data/processed/train.parquet       → {len(train):,} rows")
print(f"  data/processed/test.parquet        → {len(test):,} rows")
print(f"  data/processed/item_meta.parquet   → {len(meta):,} rows")
print(f"  data/processed/user_map.parquet")
print(f"  data/processed/item_map.parquet")
