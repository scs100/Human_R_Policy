#!/usr/bin/env python3
"""
Robot数据转换脚本 - 16维转20维
将原始Robot数据（16维 xyz+quat+gripper）转换为HAT训练格式（20维 xyz+rotation6d+gripper）
专门用于转换机器人数据

注意：
- 机器人数据有真实的action和state数据
- state来自机器人的关节状态信息

使用方法:
    # 查看帮助
    python convert_robot_data_20d.py --help
    
    # 转换机器人数据
    python convert_robot_data_20d.py --input ./robot/data --output ./processed
    
示例:
    python convert_robot_data_20d.py \
      --input /path/to/robot_data \
      --output /path/to/output
"""

import h5py
import numpy as np
import os
import cv2
import glob
import argparse
from scipy.spatial.transform import Rotation as R
from pathlib import Path

def quat_to_rot6d(quat_data):
    """
    将四元数转换为6D旋转表示
    
    Args:
        quat_data: shape (N, 4) 的四元数数据 [qx, qy, qz, qw]
        
    Returns:
        rot6d_data: shape (N, 6) 的6D旋转数据
    """
    N = quat_data.shape[0]
    rot6d_data = np.zeros((N, 6), dtype=np.float32)
    
    for i in range(N):
        quat = quat_data[i]
        
        # 检查四元数范数，如果为零或接近零，使用单位四元数
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-8:
            # 使用单位四元数 [0, 0, 0, 1]
            quat_normalized = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            # 归一化四元数
            quat_normalized = quat / quat_norm
        
        # 确保四元数格式正确 [qx, qy, qz, qw] -> [qw, qx, qy, qz] 给scipy
        quat_scipy = np.array([quat_normalized[3], quat_normalized[0], quat_normalized[1], quat_normalized[2]])
        
        try:
            # 转换为旋转矩阵
            rotation = R.from_quat(quat_scipy)
            rotation_matrix = rotation.as_matrix()
            
            # 提取前两列作为6D表示
            rot6d_data[i] = rotation_matrix[:, :2].flatten()
        except Exception as e:
            # 如果转换失败，使用单位矩阵的前两列
            print(f"警告：第{i}帧四元数转换失败，使用单位旋转: {e}")
            rot6d_data[i] = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    
    return rot6d_data

def convert_16d_to_20d(data_16d):
    """
    将16维数据转换为20维数据
    
    Args:
        data_16d: shape (T, 16) 的16维数据
                 左臂: [0:3]xyz + [3:7]quat + [7]gripper
                 右臂: [8:11]xyz + [11:15]quat + [15]gripper
    
    Returns:
        data_20d: shape (T, 20) 的20维数据
                 左臂: [0:3]xyz + [3:9]rot6d + [9]gripper  
                 右臂: [10:13]xyz + [13:19]rot6d + [19]gripper
    """
    T = data_16d.shape[0]
    data_20d = np.zeros((T, 20), dtype=np.float32)
    
    # 左臂数据转换
    left_xyz = data_16d[:, 0:3]  # xyz
    left_quat = data_16d[:, 3:7]  # quat
    left_gripper = data_16d[:, 7:8]  # gripper
    
    # 转换四元数为6D旋转
    left_rot6d = quat_to_rot6d(left_quat)
    
    # 组装左臂20维数据
    data_20d[:, 0:3] = left_xyz
    data_20d[:, 3:9] = left_rot6d
    data_20d[:, 9:10] = left_gripper
    
    # 右臂数据转换
    right_xyz = data_16d[:, 8:11]  # xyz
    right_quat = data_16d[:, 11:15]  # quat
    right_gripper = data_16d[:, 15:16]  # gripper
    
    # 转换四元数为6D旋转
    right_rot6d = quat_to_rot6d(right_quat)
    
    # 组装右臂20维数据
    data_20d[:, 10:13] = right_xyz
    data_20d[:, 13:19] = right_rot6d
    data_20d[:, 19:20] = right_gripper
    
    return data_20d

def slow_down_human_data(data, slow_factor=2):
    """
    对人类数据进行减速处理
    
    Args:
        data: 输入数据 (T, D)
        slow_factor: 减速因子，默认2倍减速
    
    Returns:
        slowed_data: 减速后的数据
    """
    if slow_factor <= 1:
        return data
    
    # 使用线性插值进行减速
    T_original = data.shape[0]
    T_slowed = int(T_original * slow_factor)
    
    # 创建新的时间索引
    original_indices = np.linspace(0, T_original - 1, T_original)
    slowed_indices = np.linspace(0, T_original - 1, T_slowed)
    
    # 对每个维度进行插值
    slowed_data = np.zeros((T_slowed, data.shape[1]), dtype=data.dtype)
    for d in range(data.shape[1]):
        slowed_data[:, d] = np.interp(slowed_indices, original_indices, data[:, d])
    
    return slowed_data

def convert_robot_file(input_file, output_file):
    """
    转换单个机器人数据文件到训练格式
    
    Args:
        input_file: 输入HDF5文件路径
        output_file: 输出HDF5文件路径
    """
    print(f"转换文件: {input_file}")
    
    # 读取时间戳文件（如果存在）
    timestamp_file = input_file.replace('.h5', '_timestamps.txt')
    timestamps = []
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # 跳过标题行
                if line.strip():
                    timestamps.append(float(line.split('\t')[0]))
        print(f"读取到 {len(timestamps)} 个时间戳")
    
    # 打开输入文件获取数据
    with h5py.File(input_file, 'r') as f_in:
        # 读取16维数据
        action_16d = f_in['action/data'][:].astype(np.float32)
        status_16d = f_in['status/data'][:].astype(np.float32)
        camera_data = f_in['camera/data'][:]
        
        T = action_16d.shape[0]
        print(f"原始数据形状: action={action_16d.shape}, status={status_16d.shape}, camera={camera_data.shape}")
        
        # 转换为20维数据
        action_20d = convert_16d_to_20d(action_16d)
        status_20d = convert_16d_to_20d(status_16d)
        
        print(f"转换后数据形状: action={action_20d.shape}, status={status_20d.shape}, camera={camera_data.shape}")
    
    # 处理图像数据
    print("处理图像数据...")
    max_len = 0
    sample_size = min(camera_data.shape[0], 30)
    
    # 估算JPEG最大长度
    for i in range(sample_size):
        img_flat = camera_data[i]
        img = img_flat.reshape(480, 640, 3)
        _, img_encode = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if len(img_encode) > max_len:
            max_len = len(img_encode)
    max_len = int(max_len * 1.2)  # 留20%余量
    print(f"估算JPEG最大长度: {max_len}")
    
    # 保存转换后的数据
    with h5py.File(output_file, 'w') as f_out:
        # 保存20维动作和状态数据
        f_out.create_dataset('action', data=action_20d)
        f_out.create_dataset('observation.state', data=status_20d)
        
        # 创建图像数据集
        img_dataset = f_out.create_dataset(
            'observation.image.left',
            shape=(camera_data.shape[0], max_len),
            dtype=np.uint8,
            chunks=(1, max_len)
        )
        
        # 逐帧编码并写入图像
        for i in range(camera_data.shape[0]):
            img_flat = camera_data[i]
            img = img_flat.reshape(480, 640, 3)
            _, img_encode = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            enc_len = len(img_encode)
            
            if enc_len > max_len:
                print(f"警告：第{i}帧JPEG长度{enc_len}超出预估{max_len}，跳过")
                continue
            
            img_dataset[i, :enc_len] = img_encode.flatten()
            
            if (i + 1) % 50 == 0:
                print(f"已保存 {i+1}/{camera_data.shape[0]} 帧")
        
        # 保存时间戳
        if timestamps:
            f_out.create_dataset('timestamps', data=np.array(timestamps))
        
        # 添加完整的属性信息
        f_out.attrs['description'] = 'Robot demonstration data with 20D action space (xyz+rotation6d+gripper)'
        f_out.attrs['embodiment'] = 'robot'
        f_out.attrs['sim'] = False
        f_out.attrs['data_type'] = 'robot'
        
        # 添加技术属性
        f_out.attrs['action_dim'] = 20
        f_out.attrs['state_dim'] = 20
        f_out.attrs['original_dim'] = 16
        f_out.attrs['conversion'] = '16d_to_20d_xyz_rot6d_gripper'
        f_out.attrs['data_source'] = 'robot_data'
        f_out.attrs['action_space'] = 'xyz_rotation6d_gripper'
        f_out.attrs['version'] = '1.0'
    
    print(f"转换完成，保存到: {output_file}\n")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Convert robot data from 16D to 20D format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', 
                        required=True,
                        help='输入目录路径（包含.h5文件的目录）')
    
    parser.add_argument('--output', '-o',
                        required=True,
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    # 获取参数
    input_dir = args.input
    output_dir = args.output
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理机器人数据
    print(f"=== 转换Robot数据 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    episode_idx = 0
    
    h5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    print(f"找到 {len(h5_files)} 个数据文件")
    
    if not h5_files:
        print(f"警告: 在 {input_dir} 中没有找到 .h5 文件")
        return
    
    for h5_file in h5_files:
        output_file = os.path.join(output_dir, f"processed_episode_{episode_idx}.hdf5")
        
        if os.path.exists(output_file):
            print(f"跳过已存在文件: processed_episode_{episode_idx}.hdf5")
            episode_idx += 1
            continue
        
        try:
            convert_robot_file(h5_file, output_file)
            episode_idx += 1
        except Exception as e:
            print(f"转换失败 {h5_file}: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("=" * 60)
    print(f"所有Robot数据转换完成！")
    print(f"输出目录: {output_dir}")
    print(f"总共处理了 {episode_idx} 个episode")

if __name__ == "__main__":
    main()
