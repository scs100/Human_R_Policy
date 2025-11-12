#!/bin/bash
# ROS2实时推理脚本

echo "========================================"
echo "ROS2实时推理系统"
echo "========================================"
echo ""

# 检查ROS2环境
if [ -z "$ROS_DISTRO" ]; then
    echo "尝试自动source ROS2环境..."
    for distro in humble iron foxy galactic; do
        if [ -f "/opt/ros/$distro/setup.bash" ]; then
            source /opt/ros/$distro/setup.bash
            break
        fi
    done
    
    if [ -z "$ROS_DISTRO" ]; then
        echo "错误: 无法找到ROS2环境"
        echo "请手动运行: source /opt/ros/<your_distro>/setup.bash"
        exit 1
    fi
fi

echo "使用ROS2: $ROS_DISTRO"
echo ""

cd /home/q/code/human-policy

# 配置路径
# MODEL_PATH="/home/q/code/human-policy/h_p_cpkt/ros_robot20_training_fixdim_b512_ckpt/policy_iter_120000_seed_0/pytorch_model.bin"
# MODEL_CFG_PATH="/home/q/code/human-policy/hdt/configs/models/act_resnet_vr_robot20.yaml"
# NORM_STATS_PATH="/home/q/code/human-policy/h_p_cpkt/ros_robot20_training_fixdim_b512_ckpt/dataset_stats.pkl"
MODEL_PATH="/home/q/code/human-policy/h_r_ckpt/policy_iter_670000_seed_0/pytorch_model.bin"
MODEL_CFG_PATH="/home/q/code/human-policy/hdt/configs/models/act_resnet_vr_robot20.yaml"
NORM_STATS_PATH="/home/q/code/human-policy/h_r_ckpt/dataset_stats.pkl"

# # 检查文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请修改脚本中的MODEL_PATH"
    exit 1
fi

if [ ! -f "$MODEL_CFG_PATH" ]; then
    echo "错误: 配置文件不存在: $MODEL_CFG_PATH"
    echo "请修改脚本中的MODEL_CFG_PATH"
    exit 1
fi

if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "错误: 归一化统计文件不存在: $NORM_STATS_PATH"
    echo "请修改脚本中的NORM_STATS_PATH"
    exit 1
fi

echo "模型路径: $MODEL_PATH"
echo "配置路径: $MODEL_CFG_PATH"
echo "统计路径: $NORM_STATS_PATH"
echo ""

# 运行推理
python3 cet/inference_6d_ros_realtime.py \
    --model_path "$MODEL_PATH" \
    --model_cfg_path "$MODEL_CFG_PATH" \
    --norm_stats_path "$NORM_STATS_PATH" \
    --base_frame base \
    --chunk_size 15 \
    --rate 10.0

