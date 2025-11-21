#!/usr/bin/env python3
"""
Robot数据转换脚本 - 关节角数据格式
将master_puppet格式的HDF5数据转换为训练格式（关节角+夹爪）
专门用于转换机器人关节角数据

注意：
- 机器人数据有真实的action和state数据
- action来自控制端（master）的关节角和夹爪数据
- state来自执行端（puppet）的关节角数据和控制端的夹爪数据

数据格式：
- 图像：/camera_observations/color_images/camera_head（JPEG编码的变长数组）
- Action：左臂关节角 + 左臂夹爪 + 右臂关节角 + 右臂夹爪
- State：左臂关节角(puppet) + 左臂夹爪(master复用) + 右臂关节角(puppet) + 右臂夹爪(master复用)

使用方法:
    # 查看帮助
    python convert_robot_data_joint.py --help
    
    # 转换机器人数据
    python convert_robot_data_joint.py --input ./robot/data --output ./processed
    
示例:
    python convert_robot_data_joint.py \
      --input /path/to/robot_data \
      --output /path/to/output
"""

import h5py
import numpy as np
import os
import cv2
import glob
import argparse
from pathlib import Path


def convert_robot_file(input_file, output_file):
    """
    转换单个机器人数据文件到训练格式（关节角数据）
    
    Args:
        input_file: 输入HDF5文件路径
        output_file: 输出HDF5文件路径
    """
    print(f"转换文件: {input_file}")
    
    # 打开输入文件获取数据
    with h5py.File(input_file, 'r') as f_in:
        # 读取图像数据 - 参考h5_read_14d_joint.py的实现
        # 实际图像数据路径是 /camera_observations/color_images/camera_head
        camera_path = '/camera_observations/color_images/camera_head'
        encoded_images = []
        image_height = None
        image_width = None
        
        try:
            if camera_path in f_in:
                image_dataset = f_in[camera_path]
                num_frames = len(image_dataset)
                print(f"找到图像数据集，总帧数: {num_frames}")
                
                # 读取所有帧的编码图像数据
                for i in range(num_frames):
                    encoded_img = image_dataset[i]
                    if not isinstance(encoded_img, np.ndarray):
                        encoded_img = np.array(encoded_img, dtype=np.uint8)
                    encoded_images.append(encoded_img)
                    
                    # 从第一帧解码获取图像尺寸
                    if i == 0 and image_height is None:
                        decoded_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                        if decoded_image is not None:
                            image_height = decoded_image.shape[0]
                            image_width = decoded_image.shape[1]
                            print(f"从第一帧解码获取图像尺寸: {image_height} x {image_width}")
                
                print(f"已读取 {len(encoded_images)} 帧编码图像数据")
            else:
                print(f"警告: 未找到图像路径 {camera_path}")
        except Exception as e:
            print(f"警告: 读取图像数据时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 读取Action数据：左臂关节角 + 左臂夹爪 + 右臂关节角 + 右臂夹爪
        action_parts = []
        
        # 左臂关节角
        try:
            left_arm = f_in['master/arm_left_position_align/data'][:].astype(np.float32)
            action_parts.append(left_arm)
            print(f"左臂关节角形状: {left_arm.shape}")
        except KeyError:
            print("错误: 未找到 master/arm_left_position_align/data")
            return False
        
        # 左臂夹爪
        try:
            left_gripper = f_in['master/end_effector_left_position_align/data'][:].astype(np.float32)
            action_parts.append(left_gripper)
            print(f"左臂夹爪形状: {left_gripper.shape}")
        except KeyError:
            print("错误: 未找到 master/end_effector_left_position_align/data")
            return False
        
        # 右臂关节角
        try:
            right_arm = f_in['master/arm_right_position_align/data'][:].astype(np.float32)
            action_parts.append(right_arm)
            print(f"右臂关节角形状: {right_arm.shape}")
        except KeyError:
            print("错误: 未找到 master/arm_right_position_align/data")
            return False
        
        # 右臂夹爪
        try:
            right_gripper = f_in['master/end_effector_right_position_align/data'][:].astype(np.float32)
            action_parts.append(right_gripper)
            print(f"右臂夹爪形状: {right_gripper.shape}")
        except KeyError:
            print("错误: 未找到 master/end_effector_right_position_align/data")
            return False
        
        # 拼接Action数据
        action_data = np.concatenate(action_parts, axis=1)
        print(f"Action数据形状: {action_data.shape}")
        
        # 读取State数据：左臂关节角(puppet) + 左臂夹爪(master复用) + 右臂关节角(puppet) + 右臂夹爪(master复用)
        state_parts = []
        
        # 左臂关节角 (puppet)
        try:
            left_arm_puppet = f_in['puppet/arm_left_position_align/data'][:].astype(np.float32)
            state_parts.append(left_arm_puppet)
            print(f"左臂关节角(puppet)形状: {left_arm_puppet.shape}")
        except KeyError:
            print("错误: 未找到 puppet/arm_left_position_align/data")
            return False
        
        # 左臂夹爪 (复用master的)
        state_parts.append(left_gripper)
        print(f"左臂夹爪(复用master)形状: {left_gripper.shape}")
        
        # 右臂关节角 (puppet)
        try:
            right_arm_puppet = f_in['puppet/arm_right_position_align/data'][:].astype(np.float32)
            state_parts.append(right_arm_puppet)
            print(f"右臂关节角(puppet)形状: {right_arm_puppet.shape}")
        except KeyError:
            print("错误: 未找到 puppet/arm_right_position_align/data")
            return False
        
        # 右臂夹爪 (复用master的)
        state_parts.append(right_gripper)
        print(f"右臂夹爪(复用master)形状: {right_gripper.shape}")
        
        # 拼接State数据
        state_data = np.concatenate(state_parts, axis=1)
        print(f"State数据形状: {state_data.shape}")
        
        # 检查数据长度一致性
        T = action_data.shape[0]
        if state_data.shape[0] != T:
            print(f"警告: Action和State的帧数不一致 (action={T}, state={state_data.shape[0]})")
            T = min(T, state_data.shape[0])
            action_data = action_data[:T]
            state_data = state_data[:T]
        
        if len(encoded_images) > 0 and len(encoded_images) != T:
            print(f"警告: 图像帧数与动作帧数不一致 (images={len(encoded_images)}, action={T})")
            min_frames = min(len(encoded_images), T)
            encoded_images = encoded_images[:min_frames]
            action_data = action_data[:min_frames]
            state_data = state_data[:min_frames]
            T = min_frames
    
    # 处理图像数据 - 计算最大JPEG长度
    print("处理图像数据...")
    max_len = 0
    if len(encoded_images) > 0:
        sample_size = min(len(encoded_images), 30)
        for i in range(sample_size):
            if len(encoded_images[i]) > max_len:
                max_len = len(encoded_images[i])
        max_len = int(max_len * 1.2)  # 留20%余量
        print(f"估算JPEG最大长度: {max_len}")
    else:
        print("警告: 没有图像数据")
        max_len = 100000  # 默认值
    
    # 保存转换后的数据
    with h5py.File(output_file, 'w') as f_out:
        # 保存动作和状态数据
        f_out.create_dataset('action', data=action_data)
        f_out.create_dataset('observation.state', data=state_data)
        
        # 创建图像数据集（如果存在图像数据）
        if len(encoded_images) > 0:
            img_dataset = f_out.create_dataset(
                'observation.image.left',
                shape=(len(encoded_images), max_len),
                dtype=np.uint8,
                chunks=(1, max_len)
            )
            
            # 逐帧写入编码图像
            for i in range(len(encoded_images)):
                enc_len = len(encoded_images[i])
                if enc_len > max_len:
                    print(f"警告：第{i}帧JPEG长度{enc_len}超出预估{max_len}，截断")
                    enc_len = max_len
                
                img_dataset[i, :enc_len] = encoded_images[i][:enc_len]
                
                if (i + 1) % 50 == 0:
                    print(f"已保存 {i+1}/{len(encoded_images)} 帧图像")
        
        # 添加完整的属性信息
        action_dim = action_data.shape[1]
        state_dim = state_data.shape[1]
        
        f_out.attrs['description'] = f'Robot demonstration data with {action_dim}D action space (joint angles+gripper)'
        f_out.attrs['embodiment'] = 'robot'
        f_out.attrs['sim'] = False
        f_out.attrs['data_type'] = 'robot'
        
        # 添加技术属性
        f_out.attrs['action_dim'] = action_dim
        f_out.attrs['state_dim'] = state_dim
        f_out.attrs['conversion'] = 'master_puppet_to_joint_angles'
        f_out.attrs['data_source'] = 'robot_joint_data'
        f_out.attrs['action_space'] = 'joint_angles_gripper'
        f_out.attrs['version'] = '1.0'
        
        if image_height is not None and image_width is not None:
            f_out.attrs['image_height'] = image_height
            f_out.attrs['image_width'] = image_width
    
    print(f"转换完成，保存到: {output_file}\n")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Convert robot joint angle data from master_puppet format to training format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', 
                        required=True,
                        help='输入目录路径（包含.h5文件的目录）')
    
    parser.add_argument('--output', '-o',
                        required=True,
                        help='输出目录路径')
    
    parser.add_argument('--recursive', '-r',
                        action='store_true',
                        help='递归遍历子目录查找h5文件')
    
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
    print(f"=== 转换Robot数据（关节角格式） ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"递归遍历: {'是' if args.recursive else '否'}")
    print()
    
    episode_idx = 0
    
    # 查找所有h5文件
    if args.recursive:
        # 递归查找所有子目录中的h5文件
        h5_files = sorted(glob.glob(os.path.join(input_dir, "**/*.h5"), recursive=True))
        # 也查找.hdf5文件
        hdf5_files = sorted(glob.glob(os.path.join(input_dir, "**/*.hdf5"), recursive=True))
        h5_files = sorted(list(set(h5_files + hdf5_files)))
    else:
        # 只查找当前目录下的h5文件
        h5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
        hdf5_files = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
        h5_files = sorted(list(set(h5_files + hdf5_files)))
    
    print(f"找到 {len(h5_files)} 个数据文件")
    
    if not h5_files:
        print(f"警告: 在 {input_dir} 中没有找到 .h5 或 .hdf5 文件")
        return
    
    # 显示文件列表
    print("\n待处理文件列表:")
    for i, h5_file in enumerate(h5_files[:10]):  # 只显示前10个
        print(f"  {i+1}. {h5_file}")
    if len(h5_files) > 10:
        print(f"  ... 还有 {len(h5_files) - 10} 个文件")
    print()
    
    # 转换每个文件
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for idx, h5_file in enumerate(h5_files):
        output_file = os.path.join(output_dir, f"processed_episode_{episode_idx}.hdf5")
        
        print(f"[{idx+1}/{len(h5_files)}] 处理: {os.path.basename(h5_file)}")
        
        if os.path.exists(output_file):
            print(f"  跳过已存在文件: processed_episode_{episode_idx}.hdf5")
            episode_idx += 1
            skip_count += 1
            continue
        
        try:
            convert_robot_file(h5_file, output_file)
            episode_idx += 1
            success_count += 1
            print(f"  ✓ 成功转换为: processed_episode_{episode_idx-1}.hdf5")
        except Exception as e:
            print(f"  ✗ 转换失败: {e}")
            fail_count += 1
            import traceback
            traceback.print_exc()
            continue
        print()
    
    print()
    print("=" * 60)
    print(f"所有Robot数据转换完成！")
    print(f"输出目录: {output_dir}")
    print(f"统计信息:")
    print(f"  - 成功转换: {success_count} 个文件")
    print(f"  - 跳过已存在: {skip_count} 个文件")
    print(f"  - 转换失败: {fail_count} 个文件")
    print(f"  - 总共处理: {episode_idx} 个episode")
    print(f"  - 输出文件编号: 0 到 {episode_idx-1}")

if __name__ == "__main__":
    main()
