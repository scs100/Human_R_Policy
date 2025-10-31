#!/usr/bin/env python3
"""
Human数据转换脚本 - 16维转20维（自动降速版本）
将原始Human数据（16维 xyz+quat+gripper）转换为HAT训练格式（20维 xyz+rotation6d+gripper）
专门用于转换human目录下的数据

处理流程：
1. 将16维数据转换为20维数据
2. 如果原始帧数 < 200，自动计算slow_factor进行降速处理
3. 降速后删除最后45action、camera、timestamps都删除）
4. 生成state（通过action向后移一帧）

自动降速功能：
- 当原始h5文件小于200帧时，自动计算降速因子使其达到约260帧
- slow_factor = 260 / 原始帧数
- 原始帧数 >= 200 时，不进行降速（slow_factor = 1.0）

注意：
- 人类数据只有action数据，没有独立的state数据
- state是通过action向后移一帧生成的：state[t] = action[t-1] (t>0), state[0] = action[0]
- 这与机器人数据不同，机器人有真实的关节状态（qpos）作为state

使用方法:
    # 查看帮助
    python convert_human_data_20d_autoslow.py --help
    
    # 自动降速（对小于200帧的文件自动降速到260帧左右）
    python convert_human_data_20d_autoslow.py --input ./human/slow --output ./processed
    
    # 手动指定减速倍数
    python convert_human_data_20d_autoslow.py --input ./human/slow --output ./processed --slow_factor 2.0
    
    # 自定义目标帧数和最小帧数阈值
    python convert_human_data_20d_autoslow.py --input ./human/slow --output ./processed --target_frames 300 --min_frames 150
    
示例:
    python convert_human_data_20d_autoslow.py \
      --input /media/testuser/data_department2/vr_robot/human/human_white_long \
      --output /home/testuser/code/opensource/human-policy/data/recordings/processed/human_white_long


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
    对人类数据进行减速处理（使用线性插值）
    
    Args:
        data: 输入数据 (T, D)
        slow_factor: 减速因子，可以是任意大于1的值
                    例如: 2.0表示2倍减速, 3.0表示3倍减速, 4.0表示4倍减速
    
    Returns:
        slowed_data: 减速后的数据
    """
    if slow_factor <= 1:
        return data
    
    # 使用线性插值进行减速
    T_original = data.shape[0]
    T_slowed = int(T_original * slow_factor)
    
    print(f"  原始帧数: {T_original} -> 减速后帧数: {T_slowed} ({slow_factor}x)")
    
    # 创建新的时间索引
    # 原始索引: [0, 1, 2, ..., T_original-1]
    # 减速索引: [0, 1, 2, ..., T_slowed-1] (更长，对应更慢的时间尺度)
    original_indices = np.arange(T_original, dtype=np.float32)
    slowed_indices = np.linspace(0, T_original - 1, T_slowed)
    
    # 对每个维度进行线性插值
    slowed_data = np.zeros((T_slowed, data.shape[1]), dtype=data.dtype)
    for d in range(data.shape[1]):
        slowed_data[:, d] = np.interp(slowed_indices, original_indices, data[:, d])
    
    return slowed_data

def get_file_frame_count(input_file):
    """
    获取文件的帧数
    
    Args:
        input_file: 输入HDF5文件路径
        
    Returns:
        frame_count: 帧数
    """
    with h5py.File(input_file, 'r') as f_in:
        action_16d = f_in['action/data'][:].astype(np.float32)
        return action_16d.shape[0]

def convert_human_file(input_file, output_file, slow_factor=2):
    """
    转换单个Human数据文件到训练格式
    
    Args:
        input_file: 输入HDF5文件路径
        output_file: 输出HDF5文件路径
        slow_factor: 人类数据减速因子
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
        camera_data = f_in['camera/data'][:]
        
        T = action_16d.shape[0]
        print(f"原始数据形状: action={action_16d.shape}, camera={camera_data.shape}")
        
        # 转换为20维数据
        action_20d = convert_16d_to_20d(action_16d)
        
        # 对人类数据进行减速处理
        print(f"对人类动作数据进行 {slow_factor}x 减速处理...")
        action_20d = slow_down_human_data(action_20d, slow_factor)
        
        T_slowed = action_20d.shape[0]
        print(f"减速后帧数: {T_slowed}")
        
        # 对图像数据进行减速处理（重复帧以匹配减速后的时间步）
        print(f"对图像数据进行减速处理...")
        print(f"  原始帧数: {T} -> 减速后帧数: {T_slowed} ({slow_factor}x)")
        camera_slowed = np.zeros((T_slowed, camera_data.shape[1]), dtype=camera_data.dtype)
        
        # 使用与动作数据相同的插值方式处理图像
        # 使用与slow_down_human_data相同的插值逻辑
        for i in range(T_slowed):
            # 计算对应的原始帧索引（与slow_down_human_data中的插值逻辑一致）
            if T_slowed > 1:
                # 映射到[0, T-1]的区间
                orig_float_idx = i * (T - 1) / (T_slowed - 1)
            else:
                orig_float_idx = 0
            
            orig_idx = int(np.round(orig_float_idx))
            orig_idx = min(max(orig_idx, 0), T - 1)  # 确保在有效范围内
            camera_slowed[i] = camera_data[orig_idx]
        
        camera_data = camera_slowed
        
        # 处理时间戳
        if timestamps:
            timestamps_slowed = []
            for i in range(T_slowed):
                # 使用与图像处理相同的逻辑
                if T_slowed > 1:
                    orig_float_idx = i * (len(timestamps) - 1) / (T_slowed - 1)
                else:
                    orig_float_idx = 0
                
                orig_idx = int(np.round(orig_float_idx))
                orig_idx = min(max(orig_idx, 0), len(timestamps) - 1)
                timestamps_slowed.append(timestamps[orig_idx])
            
            timestamps = timestamps_slowed
        
        # 删除最后45帧（减速后的action、camera、timestamps都需要删除）
        frames_to_remove = 45
        if T_slowed > frames_to_remove:
            action_20d = action_20d[:-frames_to_remove]
            camera_data = camera_data[:-frames_to_remove]
            print(f"删除最后{frames_to_remove}帧，action剩余帧数: {action_20d.shape[0]}, camera剩余帧数: {camera_data.shape[0]}")
            if timestamps:
                timestamps = timestamps[:-frames_to_remove]
                print(f"timestamps剩余帧数: {len(timestamps)}")
        else:
            print(f"警告：减速后帧数 ({T_slowed}) 小于等于 {frames_to_remove}，跳过删除操作")
        
        T_final = action_20d.shape[0]
        
        # 对于人类数据，state是action往后移一帧（根据cet/utils.py的逻辑）
        # policy_states[1:] = policy_action[:-1], policy_states[0] = policy_action[0]
        status_20d = np.zeros_like(action_20d)
        status_20d[1:] = action_20d[:-1]
        status_20d[0] = action_20d[0]
        
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
        skipped_frames = 0
        for i in range(camera_data.shape[0]):
            img_flat = camera_data[i]
            img = img_flat.reshape(480, 640, 3)
            _, img_encode = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            enc_len = len(img_encode)
            
            if enc_len > max_len:
                print(f"警告：第{i}帧JPEG长度{enc_len}超出预估{max_len}，使用零填充")
                # 用零填充而不是跳过
                img_dataset[i, :] = 0
                skipped_frames += 1
            else:
                img_dataset[i, :enc_len] = img_encode.flatten()
            
            if (i + 1) % 50 == 0:
                print(f"已保存 {i+1}/{camera_data.shape[0]} 帧")
        
        if skipped_frames > 0:
            print(f"警告：共有 {skipped_frames} 帧图像因长度超限被零填充")
        
        # 保存时间戳
        if timestamps:
            f_out.create_dataset('timestamps', data=np.array(timestamps))
        
        # 添加完整的属性信息
        f_out.attrs['description'] = 'Human demonstration data with 20D action space (xyz+rotation6d+gripper)'
        f_out.attrs['embodiment'] = 'human'
        f_out.attrs['sim'] = False
        f_out.attrs['data_type'] = 'human'
        f_out.attrs['slow_factor'] = slow_factor
        
        # 添加技术属性
        f_out.attrs['action_dim'] = 20
        f_out.attrs['state_dim'] = 20
        f_out.attrs['original_dim'] = 16
        f_out.attrs['conversion'] = '16d_to_20d_xyz_rot6d_gripper'
        f_out.attrs['data_source'] = 'human_data'
        f_out.attrs['action_space'] = 'xyz_rotation6d_gripper'
        f_out.attrs['version'] = '1.0'
    
    print(f"转换完成，保存到: {output_file}\n")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Convert human data from 16D to 20D format with automatic slowdown',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', 
                        required=True,
                        help='输入目录路径（包含.h5文件的目录）')
    
    parser.add_argument('--output', '-o',
                        required=True,
                        help='输出目录路径')
    
    parser.add_argument('--slow_factor', '-f',
                        type=float,
                        default=None,
                        help='减速倍数 (例如: 2.0 表示2倍减速, 1.0 表示不减速)。如果未指定，将自动计算')
    
    parser.add_argument('--target_frames', '-t',
                        type=int,
                        default=260,
                        help='自动降速目标帧数 (默认: 260)')
    
    parser.add_argument('--min_frames', '-m',
                        type=int,
                        default=200,
                        help='触发自动降速的最小帧数阈值 (默认: 200)')
    
    args = parser.parse_args()
    
    # 获取参数
    input_dir = args.input
    output_base_dir = args.output
    slow_factor = args.slow_factor
    target_frames = args.target_frames
    min_frames = args.min_frames
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 处理数据
    print(f"=== 转换Human数据 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_base_dir}")
    if slow_factor is None:
        print(f"减速模式: 自动（目标帧数: {target_frames}, 最小帧数阈值: {min_frames}）")
    else:
        print(f"减速倍数: {slow_factor}x")
    print()
    
    episode_idx = 0
    
    h5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    print(f"找到 {len(h5_files)} 个数据文件")
    
    if not h5_files:
        print(f"警告: 在 {input_dir} 中没有找到 .h5 文件")
        return
    
    # 如果未指定slow_factor，检查每个文件并自动计算
    auto_slow = (slow_factor is None)
    
    for h5_file in h5_files:
        output_file = os.path.join(output_base_dir, f"processed_episode_{episode_idx}.hdf5")
        
        if os.path.exists(output_file):
            print(f"跳过已存在文件: processed_episode_{episode_idx}.hdf5")
            episode_idx += 1
            continue
        
        # 自动计算slow_factor
        if auto_slow:
            try:
                frame_count = get_file_frame_count(h5_file)
                print(f"文件 {h5_file} 原始帧数: {frame_count}")
                
                if frame_count < min_frames:
                    # 计算slow_factor以达到target_frames
                    calculated_slow_factor = target_frames / frame_count
                    print(f"  原始帧数 ({frame_count}) < {min_frames}，自动降速到 {target_frames} 帧")
                    print(f"  计算出的slow_factor: {calculated_slow_factor:.3f}")
                    file_slow_factor = calculated_slow_factor
                else:
                    print(f"  原始帧数 ({frame_count}) >= {min_frames}，不进行降速")
                    file_slow_factor = 1.0
            except Exception as e:
                print(f"读取文件帧数失败 {h5_file}: {e}")
                file_slow_factor = 1.0
        else:
            file_slow_factor = slow_factor
        
        try:
            convert_human_file(h5_file, output_file, slow_factor=file_slow_factor)
            episode_idx += 1
        except Exception as e:
            print(f"转换失败 {h5_file}: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("=" * 60)
    print(f"所有Human数据转换完成！")
    print(f"输出目录: {output_base_dir}")
    print(f"总共处理了 {episode_idx} 个episode")

if __name__ == "__main__":
    main()
