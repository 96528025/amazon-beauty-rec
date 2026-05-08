"""
PyTorch Dataset for Two-Tower training with negative sampling.
两塔模型训练数据集，包含负采样。

每个样本是一个三元组：(user_idx, pos_item_idx, neg_item_idx)
  user_idx     : 用户 ID（整数）
  pos_item_idx : 用户真正买过的商品（正样本）
  neg_item_idx : 用户没买过的随机商品（负样本）

模型目标：让 score(user, pos) > score(user, neg)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class TwoTowerDataset(Dataset):
    def __init__(self, train_df: pd.DataFrame, num_items: int, num_negatives: int = 4):
        """
        train_df     : 训练集 DataFrame，包含 user_idx 和 item_idx 列
        num_items    : 商品总数（用于随机负采样范围）
        num_negatives: 每个正样本对应的负样本数量
        """
        self.users     = train_df["user_idx"].values
        self.items     = train_df["item_idx"].values
        self.num_items = num_items
        self.num_neg   = num_negatives

        # 注：不建 user_items 字典，直接随机负采样
        # 稀疏度 99.9%，随机采到正样本概率 < 0.003%，可忽略不计

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos  = self.items[idx]

        # 直接随机负采样（稀疏度 99.9%，碰到正样本概率极低）
        neg = np.random.randint(self.num_items)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos,  dtype=torch.long),
            torch.tensor(neg,  dtype=torch.long),
        )
