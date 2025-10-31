#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人类数据读取和可视化脚本 - 20维度版本
读取HDF5格式的人类数据，提取图像和动作数据，生成可视化对比图
支持processed_episode格式，专注于20维度的action和state数据对比

用法:
  # 基本用法
  python read_human_data.py --input /path/to/data --episode processed_episode_0.hdf5
  
  # 指定输出目录
  python read_human_data.py -i ./processed -e processed_episode_0.hdf5 -o ./output
  
  # 指定维度和帧率
  python read_human_data.py -i ./processed -e processed_episode_0.hdf5 -d left_arm --fps 60
  
  # 完整示例
python read_human_data.py \
  -i  /home/testuser/code/opensource/human-policy/data/recordings/processed/human_circle1025 \
  -e processed_episode_14.hdf5 \
  -o /home/testuser/code/opensource/human-policy/read_h5/human_circle1025 \
  -d all_20d \
  --fps 15

"""

import h5py
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path


class HumanDataReader:
    def __init__(self, h5_file_path, specific_dimensions=None, output_dir=None, fps=30):
        """
        初始化人类数据读取器 - 20维度版本
        
        Args:
            h5_file_path: HDF5文件路径
            specific_dimensions: 要对比的特定维度列表，如[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]表示所有20个维度
            output_dir: 输出目录路径（可选）
            fps: 视频帧率（默认30）
        """
        self.h5_file_path = h5_file_path
        self.h5_file = None
        self.action_data = None
        self.state_data = None
        self.camera_data = None
        self.specific_dimensions = specific_dimensions
        self.custom_output_dir = output_dir
        self.fps = fps
        
        # 图像信息
        self.image_height = None
        self.image_width = None
        self.image_channels = None
        
        # 列名
        self.action_columns = None
        self.state_columns = None
        
        # 输出目录
        self.output_dir = None
        self.pic_dir = None
        self.motion_dir = None
        
        # 20维度数据含义
        # [0:3] 左臂位置xyz, [3:9] 左臂旋转6D, [9] 左夹爪
        # [10:13] 右臂位置xyz, [13:19] 右臂旋转6D, [19] 右夹爪
        self.dim_names = [
            'left_x', 'left_y', 'left_z',  # 0-2: 左臂位置
            'left_rot_0', 'left_rot_1', 'left_rot_2', 'left_rot_3', 'left_rot_4', 'left_rot_5',  # 3-8: 左臂旋转6D
            'left_gripper',  # 9: 左夹爪
            'right_x', 'right_y', 'right_z',  # 10-12: 右臂位置
            'right_rot_0', 'right_rot_1', 'right_rot_2', 'right_rot_3', 'right_rot_4', 'right_rot_5',  # 13-18: 右臂旋转6D
            'right_gripper',  # 19: 右夹爪
        ]
        
    def load_data(self):
        """加载HDF5文件数据"""
        try:
            self.h5_file = h5py.File(self.h5_file_path, 'r')
            print(f"成功打开HDF5文件: {self.h5_file_path}")
            
            # 读取action数据
            self.action_data = self.h5_file['action'][:]
            print(f"Action数据形状: {self.action_data.shape}")
            
            # 读取observation.state数据作为state
            self.state_data = self.h5_file['observation.state'][:]
            print(f"State数据形状: {self.state_data.shape}")
            
            # 读取左相机数据
            self.camera_data = self.h5_file['observation.image.left'][:]
            print(f"Left Camera数据形状: {self.camera_data.shape}")
            
            # 尝试解码JPEG图像
            self.decode_jpeg_images()
            
            # 生成默认列名
            self.action_columns = [f'action_{self.dim_names[i]}' for i in range(self.action_data.shape[1])]
            self.state_columns = [f'state_{self.dim_names[i]}' for i in range(self.state_data.shape[1])]
            
            print(f"Action列数: {len(self.action_columns)}")
            print(f"State列数: {len(self.state_columns)}")
            
            # 打印文件属性
            print("\n文件属性:")
            for key, value in self.h5_file.attrs.items():
                print(f"  {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"加载HDF5文件出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def decode_jpeg_images(self):
        """解码JPEG压缩图像"""
        print("\n尝试解码JPEG图像...")
        
        # 尝试解码第一张图像来获取尺寸
        sample_size = min(5, self.camera_data.shape[0])
        decoded_images = []
        
        for i in range(sample_size):
            try:
                # 获取压缩数据（可能是变长的）
                compressed_data = self.camera_data[i]
                
                # 找到非零数据的长度
                if compressed_data.dtype == np.uint8:
                    # 计算实际数据长度
                    # 寻找最后一个非零字节
                    nonzero_indices = np.nonzero(compressed_data)[0]
                    if len(nonzero_indices) > 0:
                        actual_length = nonzero_indices[-1] + 1
                        actual_data = compressed_data[:actual_length]
                        
                        # 尝试解码
                        image = cv2.imdecode(actual_data, cv2.IMREAD_COLOR)
                        if image is not None:
                            decoded_images.append(image)
                            if i == 0:
                                self.image_height = image.shape[0]
                                self.image_width = image.shape[1]
                                self.image_channels = image.shape[2]
                                print(f"检测到图像尺寸: {self.image_height}x{self.image_width}x{self.image_channels}")
            except Exception as e:
                print(f"解码第{i}张图像失败: {e}")
                continue
        
        if decoded_images:
            print(f"成功解码 {len(decoded_images)} 张样本图像")
        else:
            print("无法解码JPEG图像，可能需要其他解码方法")
    
    def setup_output_dirs(self):
        """设置输出目录"""
        # 获取h5文件的目录和文件名
        h5_path = Path(self.h5_file_path)
        h5_name = h5_path.stem  # 不带扩展名的文件名
        
        # 使用自定义输出目录或默认目录
        if self.custom_output_dir:
            self.output_dir = Path(self.custom_output_dir)
        else:
            h5_dir = h5_path.parent
            self.output_dir = h5_dir / f"{h5_name}_human"
        
        self.pic_dir = self.output_dir / 'pictures'
        self.motion_dir = self.output_dir / 'motion_comparison'
        
        # 创建目录
        self.output_dir.mkdir(exist_ok=True)
        self.pic_dir.mkdir(exist_ok=True)
        self.motion_dir.mkdir(exist_ok=True)
        
        print(f"\n输出目录: {self.output_dir}")
        print(f"图像目录: {self.pic_dir}")
        print(f"运动对比图目录: {self.motion_dir}")
    
    def save_images(self):
        """保存所有图像为PNG文件"""
        if self.image_height is None:
            print("\n无法确定图像尺寸，尝试保存原始数据...")
            return
            
        print("\n开始保存图像...")
        
        num_images = self.camera_data.shape[0]
        
        for i in range(num_images):
            try:
                # 获取压缩数据
                compressed_data = self.camera_data[i]
                
                # 找到实际数据长度
                nonzero_indices = np.nonzero(compressed_data)[0]
                if len(nonzero_indices) > 0:
                    actual_length = nonzero_indices[-1] + 1
                    actual_data = compressed_data[:actual_length]
                    
                    # 解码JPEG
                    image = cv2.imdecode(actual_data, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # 保存为PNG文件
                        image_filename = self.pic_dir / f"frame_{i:06d}.png"
                        success = cv2.imwrite(str(image_filename), image)
                        
                        if not success:
                            print(f"警告: 无法保存图像 {image_filename}")
                    else:
                        print(f"警告: 无法解码第{i}张图像")
                else:
                    print(f"警告: 第{i}张图像数据为空")
                
                if (i + 1) % 50 == 0:
                    print(f"已保存 {i + 1}/{num_images} 张图像")
                    
            except Exception as e:
                print(f"保存第 {i} 张图像时出错: {e}")
                continue
        
        print(f"完成！共保存 {num_images} 张图像到 {self.pic_dir}")
    
    def create_video_from_images(self):
        """将保存的图像帧转换为视频"""
        if self.image_height is None:
            print("\n无法生成视频：图像尺寸未知")
            return
        
        print(f"\n开始生成视频（FPS={self.fps}）...")
        
        # 获取所有图像文件
        image_files = sorted(list(self.pic_dir.glob("frame_*.png")))
        
        if not image_files:
            print("警告: 没有找到图像文件，无法生成视频")
            return
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 输出视频文件路径
        video_filename = self.output_dir / "video.mp4"
        
        # 使用OpenCV创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_filename),
            fourcc,
            self.fps,
            (self.image_width, self.image_height)
        )
        
        if not video_writer.isOpened():
            print(f"错误: 无法创建视频文件 {video_filename}")
            return
        
        # 逐帧写入视频
        for i, image_file in enumerate(image_files):
            try:
                # 读取图像
                frame = cv2.imread(str(image_file))
                
                if frame is None:
                    print(f"警告: 无法读取图像 {image_file}")
                    continue
                
                # 写入视频
                video_writer.write(frame)
                
                if (i + 1) % 50 == 0:
                    print(f"已处理 {i + 1}/{len(image_files)} 帧")
                    
            except Exception as e:
                print(f"处理第 {i} 帧时出错: {e}")
                continue
        
        # 释放视频写入器
        video_writer.release()
        
        print(f"视频生成完成: {video_filename}")
        print(f"  - 分辨率: {self.image_width}x{self.image_height}")
        print(f"  - 帧率: {self.fps} FPS")
        print(f"  - 总帧数: {len(image_files)}")
        print(f"  - 时长: {len(image_files)/self.fps:.2f} 秒")
    
    def plot_dimension_comparison(self, dimension_idx):
        """
        绘制单个维度的action vs state对比图（包含差值）
        
        Args:
            dimension_idx: 维度索引 (0-19)
        """
        # 获取该维度的数据
        action_values = self.action_data[:, dimension_idx]
        state_values = self.state_data[:, dimension_idx]
        
        # 生成时间轴（帧数）
        frames = np.arange(len(action_values))
        
        # 创建双子图：上图显示action和state，下图显示差值
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        
        # 上图：绘制action和state
        ax1.plot(frames, action_values, 'b-', label='Action', alpha=0.7, linewidth=1.5)
        ax1.plot(frames, state_values, 'r-', label='State (action shifted by 1)', alpha=0.7, linewidth=1.5)
        
        # 获取列名
        action_col_name = self.action_columns[dimension_idx] if self.action_columns is not None else f'Action_{dimension_idx}'
        state_col_name = self.state_columns[dimension_idx] if self.state_columns is not None else f'State_{dimension_idx}'
        dim_name = self.dim_names[dimension_idx]
        
        # 设置上图标题和标签
        ax1.set_title(f'Human Data - Dimension {dimension_idx}: {dim_name}\n{action_col_name} vs {state_col_name}', 
                    fontsize=14, fontweight='bold')
        ax1.set_xlabel('Frame Number', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 下图：绘制差值 (action - state)
        # 从第1帧开始计算差值，因为第0帧的state和action相同
        diff_values = action_values[1:] - state_values[1:]
        ax2.plot(frames[1:], diff_values, 'g-', label='Difference (Action - State)', alpha=0.8, linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # 计算统计信息
        diff_mean = np.mean(diff_values)
        diff_std = np.std(diff_values)
        diff_max = np.max(np.abs(diff_values))
        
        # 设置下图标题和标签
        ax2.set_title(f'Difference (Action[t] - State[t]) | Mean: {diff_mean:.6f}, Std: {diff_std:.6f}, Max: {diff_max:.6f}', 
                     fontsize=11)
        ax2.set_xlabel('Frame Number', fontsize=12)
        ax2.set_ylabel('Difference', fontsize=12)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 调整子图间距
        plt.tight_layout()
        
        # 保存图形
        plot_filename = self.motion_dir / f"dim_{dimension_idx:02d}_{dim_name}.png"
        plt.savefig(str(plot_filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_filename
    
    def plot_all_dimensions(self):
        """绘制所有维度的对比图"""
        print("\n开始生成运动对比图...")
        
        if self.specific_dimensions is not None:
            # 绘制特定维度
            dimensions_to_plot = self.specific_dimensions
            print(f"绘制特定维度: {dimensions_to_plot}")
        else:
            # 绘制所有维度
            num_dimensions = min(self.action_data.shape[1], self.state_data.shape[1])
            dimensions_to_plot = list(range(num_dimensions))
            print(f"绘制所有维度: {num_dimensions}个")
        
        for i, dim_idx in enumerate(dimensions_to_plot):
            if dim_idx >= self.action_data.shape[1] or dim_idx >= self.state_data.shape[1]:
                print(f"警告: 维度 {dim_idx} 超出数据范围，跳过")
                continue
                
            plot_filename = self.plot_dimension_comparison(dim_idx)
            print(f"已生成对比图 {i + 1}/{len(dimensions_to_plot)}: {plot_filename.name}")
        
        print(f"完成！共生成 {len(dimensions_to_plot)} 张对比图到 {self.motion_dir}")
    
    def create_summary_plot(self):
        """创建一个包含所有20维度的汇总图"""
        print("\n生成汇总对比图...")
        
        if self.specific_dimensions is not None:
            dimensions_to_plot = self.specific_dimensions
            title_suffix = f"Specific Dimensions {dimensions_to_plot}"
        else:
            num_dimensions = min(self.action_data.shape[1], self.state_data.shape[1])
            dimensions_to_plot = list(range(num_dimensions))
            title_suffix = f"All {num_dimensions} Dimensions"
        
        frames = np.arange(len(self.action_data))
        
        # 计算子图布局 - 针对20维度优化
        n_dims = len(dimensions_to_plot)
        if n_dims <= 4:
            rows, cols = 1, n_dims
        elif n_dims <= 8:
            rows, cols = 2, 4
        elif n_dims <= 12:
            rows, cols = 3, 4
        elif n_dims <= 16:
            rows, cols = 4, 4
        elif n_dims <= 20:
            rows, cols = 4, 5  # 4行5列，适合20维度
        else:
            rows, cols = (n_dims + 4) // 5, 5
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        fig.suptitle(f'Human Data - Action vs State Comparison - {title_suffix}', fontsize=16, fontweight='bold')
        
        # 确保axes是二维数组
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dim_idx in enumerate(dimensions_to_plot):
            if dim_idx >= self.action_data.shape[1] or dim_idx >= self.state_data.shape[1]:
                continue
            
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # 获取该维度的数据
            action_values = self.action_data[:, dim_idx]
            state_values = self.state_data[:, dim_idx]
            
            # 绘制
            ax.plot(frames, action_values, 'b-', label='Action', alpha=0.7, linewidth=1)
            ax.plot(frames, state_values, 'r-', label='State', alpha=0.7, linewidth=1)
            
            # 获取维度名称
            dim_name = self.dim_names[dim_idx]
            
            # 设置标题
            ax.set_title(f'Dim {dim_idx}: {dim_name}', fontsize=10)
            ax.set_xlabel('Frame', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(dimensions_to_plot), rows * cols):
            row = i // cols
            col = i % cols
            if row < rows and col < cols:
                axes[row, col].set_visible(False)
        
        # 保存汇总图
        summary_filename = self.motion_dir / "summary_human_20d_dimensions.png"
        plt.savefig(str(summary_filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"汇总图已保存: {summary_filename}")
    
    def generate_statistics(self):
        """生成统计信息"""
        print("\n生成统计信息...")
        
        stats_filename = self.output_dir / "statistics_human.txt"
        
        with open(stats_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("人类数据统计信息 - 20维度版本\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"数据文件: {self.h5_file_path}\n")
            f.write(f"总帧数: {len(self.action_data)}\n\n")
            
            # 文件属性
            f.write("文件属性:\n")
            for key, value in self.h5_file.attrs.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write(f"图像尺寸: {self.image_height} x {self.image_width} x {self.image_channels}\n\n")
            
            # 计算每个维度的统计信息
            f.write("=" * 80 + "\n")
            f.write("各维度统计信息 (Action vs State)\n")
            f.write("注意: State = Action向右移一帧\n")
            f.write("=" * 80 + "\n\n")
            
            if self.specific_dimensions is not None:
                dimensions_to_analyze = self.specific_dimensions
                f.write(f"分析特定维度: {dimensions_to_analyze}\n\n")
            else:
                num_dimensions = min(self.action_data.shape[1], self.state_data.shape[1])
                dimensions_to_analyze = list(range(num_dimensions))
                f.write(f"分析所有维度: {num_dimensions}个\n\n")
            
            for i in dimensions_to_analyze:
                if i >= self.action_data.shape[1] or i >= self.state_data.shape[1]:
                    f.write(f"维度 {i}: 超出数据范围，跳过\n\n")
                    continue
                    
                action_col = self.action_columns[i] if self.action_columns is not None else f'Dim_{i}'
                state_col = self.state_columns[i] if self.state_columns is not None else f'Dim_{i}'
                dim_name = self.dim_names[i]
                
                action_vals = self.action_data[:, i]
                state_vals = self.state_data[:, i]
                
                # 计算误差（理论上state应该是action移一位，应该接近但不会完全一样）
                error = action_vals[1:] - state_vals[1:]  # 忽略第一帧
                mae = np.mean(np.abs(error))
                rmse = np.sqrt(np.mean(error ** 2))
                
                f.write(f"维度 {i}: {dim_name} ({action_col} vs {state_col})\n")
                f.write(f"  Action - Mean: {np.mean(action_vals):.6f}, Std: {np.std(action_vals):.6f}, "
                       f"Min: {np.min(action_vals):.6f}, Max: {np.max(action_vals):.6f}\n")
                f.write(f"  State  - Mean: {np.mean(state_vals):.6f}, Std: {np.std(state_vals):.6f}, "
                       f"Min: {np.min(state_vals):.6f}, Max: {np.max(state_vals):.6f}\n")
                f.write(f"  误差(忽略第1帧) - MAE: {mae:.6f}, RMSE: {rmse:.6f}\n")
                f.write("\n")
        
        print(f"统计信息已保存: {stats_filename}")
    
    def create_difference_summary_plot(self):
        """创建一个包含所有维度差值的汇总图"""
        print("\n生成差值汇总对比图...")
        
        if self.specific_dimensions is not None:
            dimensions_to_plot = self.specific_dimensions
            title_suffix = f"Specific Dimensions {dimensions_to_plot}"
        else:
            num_dimensions = min(self.action_data.shape[1], self.state_data.shape[1])
            dimensions_to_plot = list(range(num_dimensions))
            title_suffix = f"All {num_dimensions} Dimensions"
        
        frames = np.arange(len(self.action_data))
        
        # 计算子图布局 - 针对20维度优化
        n_dims = len(dimensions_to_plot)
        if n_dims <= 4:
            rows, cols = 1, n_dims
        elif n_dims <= 8:
            rows, cols = 2, 4
        elif n_dims <= 12:
            rows, cols = 3, 4
        elif n_dims <= 16:
            rows, cols = 4, 4
        elif n_dims <= 20:
            rows, cols = 4, 5  # 4行5列，适合20维度
        else:
            rows, cols = (n_dims + 4) // 5, 5
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        fig.suptitle(f'Difference (Action - State) - {title_suffix}', fontsize=16, fontweight='bold')
        
        # 确保axes是二维数组
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dim_idx in enumerate(dimensions_to_plot):
            if dim_idx >= self.action_data.shape[1] or dim_idx >= self.state_data.shape[1]:
                continue
            
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # 获取该维度的数据
            action_values = self.action_data[:, dim_idx]
            state_values = self.state_data[:, dim_idx]
            
            # 计算差值（从第1帧开始）
            diff_values = action_values[1:] - state_values[1:]
            
            # 绘制差值
            ax.plot(frames[1:], diff_values, 'g-', alpha=0.8, linewidth=1)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            
            # 获取维度名称
            dim_name = self.dim_names[dim_idx]
            
            # 计算统计信息
            diff_mean = np.mean(diff_values)
            diff_std = np.std(diff_values)
            
            # 设置标题
            ax.set_title(f'Dim {dim_idx}: {dim_name}\nμ={diff_mean:.4f}, σ={diff_std:.4f}', fontsize=9)
            ax.set_xlabel('Frame', fontsize=8)
            ax.set_ylabel('Diff', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        # 隐藏多余的子图
        for i in range(len(dimensions_to_plot), rows * cols):
            row = i // cols
            col = i % cols
            if row < rows and col < cols:
                axes[row, col].set_visible(False)
        
        # 保存汇总图
        summary_filename = self.motion_dir / "summary_difference_20d_dimensions.png"
        plt.savefig(str(summary_filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"差值汇总图已保存: {summary_filename}")
    
    def process(self):
        """处理完整的数据读取和可视化流程"""
        # 加载数据
        if not self.load_data():
            print("数据加载失败，退出")
            return False
        
        # 设置输出目录
        self.setup_output_dirs()
        
        # 保存图像
        self.save_images()
        
        # 生成视频
        self.create_video_from_images()
        
        # 绘制所有维度的对比图
        self.plot_all_dimensions()
        
        # 创建汇总图
        self.create_summary_plot()
        
        # 创建差值汇总图
        self.create_difference_summary_plot()
        
        # 生成统计信息
        self.generate_statistics()
        
        # 关闭HDF5文件
        if self.h5_file:
            self.h5_file.close()
        
        print("\n" + "=" * 80)
        print("人类数据处理完成！")
        print("=" * 80)
        print(f"输出目录: {self.output_dir}")
        print(f"  - 图像保存在: {self.pic_dir}")
        print(f"  - 视频文件: {self.output_dir / 'video.mp4'}")
        print(f"  - 运动对比图保存在: {self.motion_dir}")
        print(f"  - 差值汇总图: {self.motion_dir / 'summary_difference_20d_dimensions.png'}")
        print(f"  - 统计信息: {self.output_dir / 'statistics_human.txt'}")
        
        return True
    
    def __del__(self):
        """析构函数，确保HDF5文件被关闭"""
        if self.h5_file:
            self.h5_file.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Read and visualize human data from HDF5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
20维度说明:
  0-2:   左臂位置 (x, y, z)
  3-8:   左臂旋转 (6D)
  9:     左夹爪
  10-12: 右臂位置 (x, y, z)
  13-18: 右臂旋转 (6D)
  19:    右夹爪

预定义维度组:
  all_20d    - 所有20个维度
  first_10d  - 前10个维度(左臂)
  last_10d   - 后10个维度(右臂)
  left_arm   - 左臂数据
  right_arm  - 右臂数据
  positions  - 位置信息
  grippers   - 夹爪信息
        """
    )
    
    parser.add_argument('--input', '-i',
                        required=True,
                        help='输入目录路径（包含HDF5文件的目录）')
    
    parser.add_argument('--episode', '-e',
                        required=True,
                        help='要读取的episode文件名（例如: processed_episode_0.hdf5）')
    
    parser.add_argument('--output', '-o',
                        default=None,
                        help='输出目录路径（默认为输入文件同目录下的_human文件夹）')
    
    parser.add_argument('--dimensions', '-d',
                        default=None,
                        help='要可视化的维度 (例如: 0,1,2,3 或 left_arm 或 first_10d)')
    
    parser.add_argument('--fps',
                        type=int,
                        default=30,
                        help='生成视频的帧率（默认: 30）')
    
    args = parser.parse_args()
    
    # 构建完整的文件路径
    input_dir = args.input
    episode_file = args.episode
    h5_file_path = os.path.join(input_dir, episode_file)
    
    # 解析特定维度参数
    specific_dimensions = None
    if args.dimensions:
        dim_str = args.dimensions
        
        # 检查是否是预定义的维度组
        if dim_str.lower() == 'all_20d':
            specific_dimensions = list(range(20))
            print(f"所有20个维度: {specific_dimensions}")
        elif dim_str.lower() == 'first_10d':
            specific_dimensions = list(range(10))
            print(f"前10个维度: {specific_dimensions}")
        elif dim_str.lower() == 'last_10d':
            specific_dimensions = list(range(10, 20))
            print(f"后10个维度: {specific_dimensions}")
        elif dim_str.lower() == 'left_arm':
            specific_dimensions = list(range(10))
            print(f"左臂(前10个维度): {specific_dimensions}")
        elif dim_str.lower() == 'right_arm':
            specific_dimensions = list(range(10, 20))
            print(f"右臂(后10个维度): {specific_dimensions}")
        elif dim_str.lower() == 'positions':
            specific_dimensions = [0, 1, 2, 10, 11, 12]
            print(f"位置(xyz): {specific_dimensions}")
        elif dim_str.lower() == 'grippers':
            specific_dimensions = [9, 19]
            print(f"夹爪: {specific_dimensions}")
        else:
            try:
                specific_dimensions = [int(x.strip()) for x in dim_str.split(',')]
                print(f"指定维度: {specific_dimensions}")
            except ValueError:
                print("错误: 维度参数格式不正确")
                print("可用选项: all_20d, first_10d, last_10d, left_arm, right_arm, positions, grippers")
                print("或指定维度: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
                sys.exit(1)
    
    # 检查文件是否存在
    if not os.path.exists(h5_file_path):
        print(f"错误: 文件不存在: {h5_file_path}")
        sys.exit(1)
    
    # 创建读取器并处理
    reader = HumanDataReader(h5_file_path, specific_dimensions, args.output, args.fps)
    success = reader.process()
    
    if success:
        print("\n人类数据处理成功完成！")
    else:
        print("\n人类数据处理失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()
