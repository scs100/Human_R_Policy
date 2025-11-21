import os
import numpy as np

# ===== 16维关节角动作空间定义 =====
# 数据格式（来自 convert_robot_data_joint.py）：
# [0-6]: 左臂关节角 (7维)
# [7]: 左夹爪 (1维)
# [8-14]: 右臂关节角 (7维)
# [15]: 右夹爪 (1维)
# 总共: 7 + 1 + 7 + 1 = 16维

ACTION_STATE_VEC_SIZE = 16

# ===== 动作空间索引定义 =====
# 对于关节角数据，没有EEF（末端执行器）位置，只有关节角和夹爪
# 但为了兼容训练代码中的损失计算，我们需要定义这些索引
# 注意：关节角数据中没有EEF位置，所以这些索引可能不适用
# 如果需要计算损失，可能需要使用关节角索引

# 左臂关节角索引
OUTPUT_LEFT_ARM_JOINTS = np.arange(0, 7)  # 左臂关节角 (7维)
OUTPUT_LEFT_GRIPPER = np.array([7])  # 左夹爪

# 右臂关节角索引
OUTPUT_RIGHT_ARM_JOINTS = np.arange(8, 15)  # 右臂关节角 (7维)
OUTPUT_RIGHT_GRIPPER = np.array([15])  # 右夹爪

# 为了兼容现有代码，定义这些（虽然关节角数据中没有EEF）
# 如果训练代码需要这些，可能需要根据实际情况调整
OUTPUT_LEFT_EEF = np.array([])  # 关节角数据中没有EEF位置
OUTPUT_RIGHT_EEF = np.array([])  # 关节角数据中没有EEF位置
OUTPUT_HEAD_EEF = np.array([])  # 没有头部
OUTPUT_LEFT_KEYPOINTS = np.array([])  # 没有关键点
OUTPUT_RIGHT_KEYPOINTS = np.array([])  # 没有关键点

# 所有输出索引
OUTPUT_INDEX = np.concatenate([
    OUTPUT_LEFT_ARM_JOINTS, 
    OUTPUT_LEFT_GRIPPER,
    OUTPUT_RIGHT_ARM_JOINTS, 
    OUTPUT_RIGHT_GRIPPER
])

# ===== 状态空间索引定义 =====
# State数据格式：左臂关节角(puppet) + 左臂夹爪(master复用) + 右臂关节角(puppet) + 右臂夹爪(master复用)
# 与Action格式相同，所以索引也相同
QPOS_INDICES = np.arange(0, 16)  # State就是关节角+夹爪数据

# ===== 数据重定向索引 =====
# 关节角数据不需要关键点重定向
RETARGETTING_INDICES = []  # 关节角数据没有关键点
VALID_RETARGETTING_INDICES = []  # 关节角数据没有关键点

# ===== 机器人控制器索引 =====
# 16维关节角数据：左右臂各7个关节 + 左右夹爪各1个
CONTROLLER_UPPERBODY_INDICES = [*range(0, 16)]

# ===== URDF索引 =====
# 对于关节角数据，这些索引对应到实际的机器人关节
H1_HEAD_POS = [0.0, 0.0, 0.0]  # 没有头部
H1_ALL_INDICES = [i for i in range(16)]
H1_BODY_INDICES = []  # 没有身体
H1_LEFT_ARM_INDICES = list(range(0, 7))  # 左臂关节角 (7维)
H1_RIGHT_ARM_INDICES = list(range(8, 15))  # 右臂关节角 (7维)
H1_LEFT_HAND_INDICES = [7]  # 左夹爪
H1_RIGHT_HAND_INDICES = [15]  # 右夹爪
H1_MOTOR_INDICES = [*range(0, 16)]  # 所有16维都是电机控制

# ===== 真实机器人接口索引 =====
# 这些索引用于从qpos中提取对应的关节角数据
H1_QPOS_LEFT_ARM_INDICES = list(range(0, 7))  # 左臂关节角 (7维)
H1_QPOS_LEFT_HAND_INDICES = [7]  # 左夹爪
H1_QPOS_RIGHT_ARM_INDICES = list(range(8, 15))  # 右臂关节角 (7维)
H1_QPOS_RIGHT_HAND_INDICES = [15]  # 右夹爪

# 为了兼容性，定义这些（虽然关节角数据中没有肘部和腕部分离）
H1_QPOS_LEFT_ELBOW_INDICES = list(range(0, 4))  # 左臂前4个关节（可以理解为肘部相关）
H1_QPOS_LEFT_WRIST_INDICES = list(range(4, 7))  # 左臂后3个关节（可以理解为腕部相关）
H1_QPOS_RIGHT_ELBOW_INDICES = list(range(8, 12))  # 右臂前4个关节
H1_QPOS_RIGHT_WRIST_INDICES = list(range(12, 15))  # 右臂后3个关节

# ===== 其他设置 =====
STILL_HEAD_MAT = np.array([[0.95533649, 0, 0.29552021], [0, 1, 0], [-0.29552021, 0, 0.95533649]])

