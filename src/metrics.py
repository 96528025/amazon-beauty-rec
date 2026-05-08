"""
Evaluation metrics for recommendation systems.
推荐系统评估指标：Precision@K, Recall@K, NDCG@K
"""

import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """推荐的 Top-K 里有多少是用户真正交互过的"""
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """用户真正交互过的里，有多少被推荐到了"""
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """排名越靠前的命中，得分越高"""
    rec_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(rec_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(recommended: dict, ground_truth: dict, k: int = 10) -> dict:
    """
    对所有用户计算平均指标
    recommended:  {user_idx: [item_idx, ...]}  按分数排好序的推荐列表
    ground_truth: {user_idx: set(item_idx)}    测试集里用户真实交互的商品
    """
    precisions, recalls, ndcgs = [], [], []

    for user_idx, relevant_items in ground_truth.items():
        if user_idx not in recommended:
            continue
        recs = recommended[user_idx]
        relevant = set(relevant_items)
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    return {
        f"Precision@{k}": round(np.mean(precisions), 4),
        f"Recall@{k}":    round(np.mean(recalls), 4),
        f"NDCG@{k}":      round(np.mean(ndcgs), 4),
        "num_users":      len(precisions),
    }
