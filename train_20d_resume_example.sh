#!/bin/bash

# ========================================
# 训练脚本 Resume 使用示例
# ========================================

# ---------------------------------------
# 方式 1: 自动恢复（最常用）✅
# ---------------------------------------
# 场景：训练中断后继续训练
# 优点：完全恢复所有状态（模型、优化器、迭代次数）

# 第一次训练
EXPT_ID="human_and_robot_20d_mixed_20251027_213938"

echo "=== 方式 1: 自动恢复 ==="
echo "直接运行训练脚本，使用相同的 EXPT_ID 即可自动恢复"
echo ""

python hdt/main.py \
  --exptid $EXPT_ID \
  --batch_size 64 \
  --num_epochs 1000000 \
  --lr 1e-4 \
  --chunk_size 15 \
  --cond_mask_prob 0.1 \
  --dataset_json_path hdt/configs/datasets/human_2000_robot_200.json \
  --model_cfg_path hdt/configs/models/act_resnet_vr_robot20.yaml \
  --human_slow_down_factor 1 \
  --no_wandb

# 脚本会自动检测 ${EXPT_ID}_ckpt/ 目录中最新的 checkpoint：
# - policy_iter_10000_seed_0/
# - policy_iter_20000_seed_0/
# - ...
# - policy_iter_230000_seed_0/  <-- 自动恢复最新的这个

echo "训练会从最新的 checkpoint 自动恢复！"
echo ""

# ---------------------------------------
# 方式 2: 从指定 checkpoint Fine-tune 🔧
# ---------------------------------------
# 场景：使用之前训练好的模型作为预训练模型，开始新的训练
# 优点：可以修改学习率等超参数

echo "=== 方式 2: 从预训练模型 Fine-tune ==="
echo "适用于：从其他实验的 checkpoint 开始训练"
echo ""

# 创建新的实验 ID
NEW_EXPT_ID="finetuned_from_230k_$(date +%Y%m%d_%H%M%S)"

# 从之前实验的 checkpoint 加载
PRETRAINED_PATH="human_and_robot_20d_mixed_20251027_213938_ckpt/policy_iter_230000_seed_0/pytorch_model.bin"

python hdt/main.py \
  --exptid $NEW_EXPT_ID \
  --batch_size 64 \
  --num_epochs 1000000 \
  --lr 5e-5 \
  --chunk_size 15 \
  --cond_mask_prob 0.1 \
  --dataset_json_path hdt/configs/datasets/human_2000_robot_200.json \
  --model_cfg_path hdt/configs/models/act_resnet_vr_robot20.yaml \
  --human_slow_down_factor 1 \
  --load_pretrained_path $PRETRAINED_PATH \
  --no_wandb

echo "从 $PRETRAINED_PATH 加载权重，但使用新的优化器状态！"
echo ""

# ---------------------------------------
# 方式 3: 混合使用 🚀
# ---------------------------------------
# 场景：从预训练模型开始，训练中断后自动恢复

echo "=== 方式 3: 混合使用 ==="
echo "第一次运行时从预训练模型加载，之后自动恢复"
echo ""

MIXED_EXPT_ID="mixed_resume_$(date +%Y%m%d_%H%M%S)"
PRETRAINED_PATH="human_and_robot_20d_mixed_20251027_213938_ckpt/policy_iter_100000_seed_0/pytorch_model.bin"

python hdt/main.py \
  --exptid $MIXED_EXPT_ID \
  --batch_size 64 \
  --num_epochs 1000000 \
  --lr 1e-4 \
  --chunk_size 15 \
  --cond_mask_prob 0.1 \
  --dataset_json_path hdt/configs/datasets/human_2000_robot_200.json \
  --model_cfg_path hdt/configs/models/act_resnet_vr_robot20.yaml \
  --human_slow_down_factor 1 \
  --load_pretrained_path $PRETRAINED_PATH \
  --no_wandb

# 逻辑：
# 1. 首先检查 ${MIXED_EXPT_ID}_ckpt/ 中是否有 checkpoint
#    - 如果有：优先恢复（忽略 --load_pretrained_path）
#    - 如果没有：从 --load_pretrained_path 加载
# 2. 训练中断后再次运行，会自动从 checkpoint 恢复

echo "智能恢复：优先使用 checkpoint，否则使用预训练模型！"

