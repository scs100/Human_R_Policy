#!/bin/bash
# 20维度模型推理脚本
# 使用修改后的 eval_6d.py 进行推理

# 激活h_policy conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate h_policy

cd /home/testuser/code/opensource/human-policy

# ========== 配置参数 ==========
# H5数据文件路径
HDF_FILE="data/recordings/processed/robot_val/processed_episode_6.hdf5"

# 模型checkpoint目录
CKPT_DIR="human_and_robot_20d_mixed_20251027_213938_ckpt"

# 选择模型iteration（可选: 10000, 20000, ..., 260000）
MODEL_ITER="280000"

# 模型配置文件（20维配置）
CONFIG_FILE="hdt/configs/models/act_resnet_vr_robot20.yaml"

# 语言嵌入文件（可以使用空的）
LANG_EMBED="hdt/empty_lang_embed.pt"

# Chunk size（预测序列长度）
CHUNK_SIZE=15

# ========== 开始推理 ==========
echo "===================================="
echo "     20维度模型推理"
echo "===================================="
echo "数据文件: $HDF_FILE"
echo "模型checkpoint: policy_iter_${MODEL_ITER}_seed_0"
echo "配置文件: $CONFIG_FILE"
echo "Chunk size: $CHUNK_SIZE"
echo "===================================="
echo ""

# 检查文件是否存在
if [ ! -f "$HDF_FILE" ]; then
    echo "错误: 数据文件不存在: $HDF_FILE"
    exit 1
fi

if [ ! -f "$CKPT_DIR/policy_iter_${MODEL_ITER}_seed_0/pytorch_model.bin" ]; then
    echo "错误: 模型文件不存在: $CKPT_DIR/policy_iter_${MODEL_ITER}_seed_0/pytorch_model.bin"
    echo "可用的checkpoints:"
    ls -d $CKPT_DIR/policy_iter_*/ 2>/dev/null | tail -5
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 运行推理
python cet/eval_6d.py \
  --hdf_file_path "$HDF_FILE" \
  --norm_stats_path "$CKPT_DIR/dataset_stats.pkl" \
  --model_path "$CKPT_DIR/policy_iter_${MODEL_ITER}_seed_0/pytorch_model.bin" \
  --lang_embeddings_path "$LANG_EMBED" \
  --model_cfg_path "$CONFIG_FILE" \
  --chunk_size $CHUNK_SIZE \
  --plot

echo ""
echo "===================================="
echo "推理完成！"
echo "===================================="
echo "对比图已保存: prediction_vs_gt.png"

