#!/bin/bash
# Download Amazon Beauty Reviews dataset
# Source: https://nijianmo.github.io/amazon/index.html
#
# 下载亚马逊美妆评论数据集

set -e

DATA_DIR="$(dirname "$0")"

echo "Downloading reviews (3.7M rows, ~500MB)..."
echo "正在下载评论数据（370万条，约500MB）..."
curl -L -o "$DATA_DIR/All_Beauty.jsonl.gz" \
  "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz"

echo "Downloading product metadata..."
echo "正在下载商品元数据..."
curl -L -o "$DATA_DIR/meta_All_Beauty.jsonl.gz" \
  "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz"

echo "Done! / 下载完成！"
echo "Next step: python src/preprocess.py"
