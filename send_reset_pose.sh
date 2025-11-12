#!/bin/bash
# 从h5文件发送reset pose的便捷脚本
# 使用方法:
#   直接发送:  ./send_reset_pose.sh <h5文件> [帧索引]
#   插值发送:  ./send_reset_pose.sh <h5文件> [帧索引] --interpolate [时长] [频率]
#
# 示例:
#   ./send_reset_pose.sh data.h5 10
#   ./send_reset_pose.sh data.h5 10 --interpolate
#   ./send_reset_pose.sh data.h5 10 --interpolate 5.0 50

set -e

# 解析参数
H5_FILE=$1
FRAME_IDX=${2:-10}  # 默认第10帧
MODE=${3:-""}       # 可选 --interpolate
DURATION=${4:-3.0}  # 默认3秒
HZ=${5:-50}         # 默认50Hz
BASE_FRAME=${6:-base}  # 默认base

if [ -z "$H5_FILE" ]; then
    echo "错误: 请提供h5文件路径"
    echo ""
    echo "使用方法:"
    echo "  直接发送:  $0 <h5文件> [帧索引]"
    echo "  插值发送:  $0 <h5文件> [帧索引] --interpolate [时长] [频率]"
    echo ""
    echo "示例:"
    echo "  $0 data.h5 10                      # 直接发送第10帧"
    echo "  $0 data.h5 10 --interpolate        # 插值发送（3秒，50Hz）"
    echo "  $0 data.h5 10 --interpolate 5.0    # 插值发送（5秒，50Hz）"
    echo "  $0 data.h5 10 --interpolate 5.0 30 # 插值发送（5秒，30Hz）"
    exit 1
fi

if [ ! -f "$H5_FILE" ]; then
    echo "错误: 文件不存在: $H5_FILE"
    exit 1
fi

echo "============================================"
echo "发送Reset Pose"
echo "============================================"
echo "H5文件: $H5_FILE"
echo "帧索引: $FRAME_IDX"

if [ "$MODE" = "--interpolate" ]; then
    echo "模式: 插值发送（平滑运动）"
    echo "时长: ${DURATION}秒"
    echo "频率: ${HZ}Hz"
else
    echo "模式: 直接发送"
fi
echo "Base Frame: $BASE_FRAME"

echo "============================================"

# 切换到项目目录
cd "$(dirname "$0")"

# 运行Python脚本
if [ "$MODE" = "--interpolate" ]; then
    python3 cet/send_reset_pose.py "$H5_FILE" --frame "$FRAME_IDX" --interpolate --duration "$DURATION" --hz "$HZ" --base-frame "$BASE_FRAME"
else
    python3 cet/send_reset_pose.py "$H5_FILE" --frame "$FRAME_IDX" --base-frame "$BASE_FRAME"
fi

echo ""
echo "✓ 完成"

