#!/bin/bash

# 训练20维人类+机器人混合数据集的脚本
# 使用说明: bash train_20d_human_robot.sh

# 设置实验ID（会创建对应的checkpoint目录）
EXPT_ID="human_and_robot_20d_mixed_$(date +%Y%m%d_%H%M%S)"

# 训练参数
BATCH_SIZE=64              # 批次大小，根据GPU显存调整
NUM_EPOCHS=1000000         # 训练轮数
LR=1e-4                   # 学习率
CHUNK_SIZE=15            # 预测未来的步数
COND_MASK_PROB=0.1        # 条件dropout概率
HUMAN_SLOWDOWN=1          # 人类数据减速倍数（数据已在转换时减速，这里设为1避免重复减速）

# 配置文件路径
DATASET_JSON="hdt/configs/datasets/human_2000_robot_200.json"
MODEL_CFG="hdt/configs/models/act_resnet_vr_robot20.yaml"

# WandB设置（如果不想使用wandb，添加 --no_wandb）
# export WANDB_PROJECT="human-policy-20d"
# export WANDB_ENTITY="your-wandb-username"

echo "=========================================="
echo "训练配置信息："
echo "  实验ID: $EXPT_ID"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  学习率: $LR"
echo "  Chunk大小: $CHUNK_SIZE"
echo "  数据集配置: $DATASET_JSON"
echo "  模型配置: $MODEL_CFG"
echo "=========================================="

# 启动训练
python hdt/main.py \
  --exptid $EXPT_ID \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --lr $LR \
  --chunk_size $CHUNK_SIZE \
  --cond_mask_prob $COND_MASK_PROB \
  --dataset_json_path $DATASET_JSON \
  --model_cfg_path $MODEL_CFG \
  --human_slow_down_factor $HUMAN_SLOWDOWN \
  --no_wandb

# 如果想从checkpoint恢复训练，添加：
# --load_pretrained_path ${EXPT_ID}_ckpt/policy_epoch_1000_seed_0.ckpt

echo "=========================================="
echo "训练完成！"
echo "Checkpoint保存在: ${EXPT_ID}_ckpt/"
echo "=========================================="

