import os
import numpy as np

# ===== 20维动作空间定义 =====
# 你的20维结构：
# [0-2]: 左臂位置 (x, y, z)
# [3-8]: 左臂旋转6D表示 (6维)
# [9]: 左夹爪
# [10-12]: 右臂位置 (x, y, z)  
# [13-18]: 右臂旋转6D表示 (6维)
# [19]: 右夹爪

ACTION_STATE_VEC_SIZE = 20
QPOS_INDICES = np.array([])  # 没有关节位置数据
OUTPUT_LEFT_EEF = np.arange(0, 9)   # 左臂位置+旋转6D (9维)
OUTPUT_RIGHT_EEF = np.arange(10, 19) # 右臂位置+旋转6D (9维)
OUTPUT_LEFT_GRIPPER = np.array([9])  # 左夹爪
OUTPUT_RIGHT_GRIPPER = np.array([19])  # 右夹爪
# 20维数据不需要头部和关键点
OUTPUT_HEAD_EEF = np.array([])
OUTPUT_LEFT_KEYPOINTS = np.array([])
OUTPUT_RIGHT_KEYPOINTS = np.array([])

OUTPUT_INDEX = np.concatenate([OUTPUT_LEFT_EEF, OUTPUT_RIGHT_EEF])

# ===== 20维数据不需要关键点重定向 =====
RETARGETTING_INDICES = []  # 20维数据没有关键点
VALID_RETARGETTING_INDICES = []  # 20维数据没有关键点

# 机器人控制器索引 - 只有双手数据
CONTROLLER_UPPERBODY_INDICES = [*range(0, 20)]

# URDF索引 - 6自由度机械臂+夹爪
H1_HEAD_POS = [0.0, 0.0, 0.0]  # 没有头部
H1_ALL_INDICES = [i for i in range(20)]
H1_BODY_INDICES = []  # 没有身体
H1_LEFT_ARM_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 左臂位置+旋转6D (9维)
H1_RIGHT_ARM_INDICES = [10, 11, 12, 13, 14, 15, 16, 17, 18]  # 右臂位置+旋转6D (9维)
H1_LEFT_HAND_INDICES = [9]  # 左夹爪
H1_RIGHT_HAND_INDICES = [19]  # 右夹爪
H1_MOTOR_INDICES = [*range(0, 20)]

# 真实机器人接口索引 - 6自由度机械臂+夹爪
H1_QPOS_LEFT_ELBOW_INDICES = [0, 1, 2]  # 左臂位置 (3维)
H1_QPOS_LEFT_WRIST_INDICES = [3, 4, 5, 6, 7, 8]  # 左臂旋转6D (6维)
H1_QPOS_LEFT_ARM_INDICES = H1_QPOS_LEFT_ELBOW_INDICES + H1_QPOS_LEFT_WRIST_INDICES
H1_QPOS_LEFT_HAND_INDICES = [9]  # 左夹爪
H1_QPOS_RIGHT_ELBOW_INDICES = [10, 11, 12]  # 右臂位置 (3维)
H1_QPOS_RIGHT_WRIST_INDICES = [13, 14, 15, 16, 17, 18]  # 右臂旋转6D (6维)
H1_QPOS_RIGHT_ARM_INDICES = H1_QPOS_RIGHT_ELBOW_INDICES + H1_QPOS_RIGHT_WRIST_INDICES
H1_QPOS_RIGHT_HAND_INDICES = [19]  # 右夹爪

# 其他设置
STILL_HEAD_MAT = np.array([[0.95533649, 0, 0.29552021], [0, 1, 0], [-0.29552021, 0, 0.95533649]])
