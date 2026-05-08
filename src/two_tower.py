"""
Two-Tower Model for candidate retrieval.
两塔召回模型：用户塔 + 商品塔，各自输出 64 维向量。

用户塔：user_id → Embedding → MLP → L2归一化 → 64维向量
商品塔：item_id → Embedding → MLP → L2归一化 → 64维向量
相似度：内积（向量归一化后内积 = 余弦相似度）

损失函数：BPR Loss
  score_pos = dot(user_vec, pos_item_vec)
  score_neg = dot(user_vec, neg_item_vec)
  loss = -log(sigmoid(score_pos - score_neg))
  目标：让正样本分数远高于负样本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(user_ids)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)   # L2 归一化，让向量长度 = 1


class ItemTower(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(item_ids)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_items, embedding_dim)

    def forward(self, user_ids, pos_item_ids, neg_item_ids):
        user_vec     = self.user_tower(user_ids)
        pos_item_vec = self.item_tower(pos_item_ids)
        neg_item_vec = self.item_tower(neg_item_ids)

        # 内积 = 相似度分数
        pos_score = (user_vec * pos_item_vec).sum(dim=-1)
        neg_score = (user_vec * neg_item_vec).sum(dim=-1)

        # BPR Loss：正样本分数应高于负样本
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return loss

    def get_user_embedding(self, user_ids):
        return self.user_tower(user_ids)

    def get_item_embedding(self, item_ids):
        return self.item_tower(item_ids)
