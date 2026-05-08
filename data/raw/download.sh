#!/bin/bash
# Download Amazon Beauty Reviews dataset
# Source: https://nijianmo.github.io/amazon/index.html
#
# 下载亚马逊美妆评论数据集

set -e

DATA_DIR="$(dirname "$0")"

echo "Downloading reviews (~700MB, may take a few minutes)..."
echo "正在下载评论数据（约700MB，需要几分钟）..."
curl -L -o "$DATA_DIR/All_Beauty.jsonl.gz" \
  "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl.gz"

echo "Downloading product metadata..."
echo "正在下载商品元数据..."
curl -L -o "$DATA_DIR/meta_All_Beauty.jsonl.gz" \
  "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_All_Beauty.jsonl.gz"

echo "Done! / 下载完成！"
echo "Next step: python src/preprocess.py"
