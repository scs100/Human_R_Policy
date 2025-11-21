#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5数据读取和可视化脚本 - 14维度版本（关节角数据）
读取HDF5文件，提取图像和动作数据，生成可视化对比图
支持processed_episode格式、ros_data_sync格式和master_puppet格式
支持14维度的action和status数据对比（关节角数据）

python h5_read_14d_joint.py processed_episode_0.hdf5
python h5_read_14d_joint.py processed_episode_0.hdf5 0,1,2,3,4,5,6,7,8,9,10,11,12,13
"""

import h5py
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R



class H5DataReader14D:
    def __init__(self, h5_file_path, specific_dimensions=None):
        """
        初始化HDF5数据读取器 - 14维度版本（关节角数据）
        
        Args:
            h5_file_path: HDF5文件路径
            specific_dimensions: 要对比的特定维度列表，如[0,1,2,3,4,5,6,7,8,9,10,11,12,13]表示所有14个维度
        """
        self.h5_file_path = h5_file_path
        self.h5_file = None
        self.action_data = None
        self.camera_data = None
        self.status_data = None
        self.specific_dimensions = specific_dimensions
        
        # 四元数数据
        self.action_quat_data = None
        self.status_quat_data = None
        
        # 图像信息
        self.image_height = None
        self.image_width = None
        self.image_channels = None
        
        # 列名
        self.action_columns = None
        self.status_columns = None
        
        # 输出目录
        self.output_dir = None
        self.pic_dir = None
        self.motion_dir = None
        
        # 文件格式检测
        self.file_format = None  # 'processed_episode', 'ros_data_sync', or 'master_puppet'
        
    def detect_file_format(self):
        """检测HDF5文件格式"""
        keys = list(self.h5_file.keys())
        
        if 'observation.state' in keys and 'observation.image.left' in keys:
            self.file_format = 'processed_episode'
            print("检测到文件格式: processed_episode")
        elif 'action' in keys and 'camera' in keys and 'status' in keys:
            self.file_format = 'ros_data_sync'
            print("检测到文件格式: ros_data_sync")
        elif 'master' in keys and 'puppet' in keys and 'camera_color_channel' in keys:
            self.file_format = 'master_puppet'
            print("检测到文件格式: master_puppet")
        else:
            print(f"未知文件格式，键: {keys}")
            return False
        
        return True
    
    def load_data(self):
        """加载HDF5文件数据"""
        try:
            self.h5_file = h5py.File(self.h5_file_path, 'r')
            print(f"成功打开HDF5文件: {self.h5_file_path}")
            
            # 检测文件格式
            if not self.detect_file_format():
                return False
            
            if self.file_format == 'processed_episode':
                return self._load_processed_episode_data()
            elif self.file_format == 'ros_data_sync':
                return self._load_ros_data_sync_data()
            elif self.file_format == 'master_puppet':
                return self._load_master_puppet_data()
            
            return False
            
        except Exception as e:
            print(f"加载HDF5文件出错: {e}")
            return False
    
    def _load_processed_episode_data(self):
        """加载processed_episode格式数据"""
        try:
            # 读取action数据
            self.action_data = self.h5_file['action'][:]
            print(f"Action数据形状: {self.action_data.shape}")
            
            # 读取observation.state数据作为status
            self.status_data = self.h5_file['observation.state'][:]
            print(f"Status数据形状: {self.status_data.shape}")
            
            # 读取左相机数据
            self.camera_data = self.h5_file['observation.image.left'][:]
            print(f"Left Camera数据形状: {self.camera_data.shape}")
            
            # 对于processed_episode格式，图像可能是压缩格式
            # 先尝试常见的图像尺寸
            image_size = self.camera_data.shape[1]
            print(f"图像数据大小: {image_size}")
            
            # 尝试常见的图像尺寸
            possible_sizes = [
                # 常见RGB格式
                (84, 84, 3), (128, 128, 3), (64, 64, 3), (256, 256, 3), (512, 512, 3),
                (224, 224, 3), (480, 640, 3), (720, 1280, 3), (1080, 1920, 3),
                # 常见灰度格式
                (84, 84, 1), (128, 128, 1), (64, 64, 1), (256, 256, 1), (512, 512, 1),
                (224, 224, 1), (480, 640, 1), (720, 1280, 1), (1080, 1920, 1),
                # 常见RGBA格式
                (84, 84, 4), (128, 128, 4), (64, 64, 4), (256, 256, 4), (512, 512, 4),
                (224, 224, 4), (480, 640, 4), (720, 1280, 4), (1080, 1920, 4),
            ]
            
            for h, w, c in possible_sizes:
                if h * w * c == image_size:
                    self.image_height = h
                    self.image_width = w
                    self.image_channels = c
                    print(f"检测到图像尺寸: {h}x{w}x{c}")
                    break
            
            if self.image_height is None:
                # 尝试自动推断图像尺寸
                print(f"尝试自动推断图像尺寸...")
                # 尝试常见的宽高比
                common_ratios = [(1, 1), (4, 3), (16, 9), (3, 2), (5, 4)]
                for ratio_h, ratio_w in common_ratios:
                    for c in [1, 3, 4]:  # 灰度、RGB、RGBA
                        # 计算可能的尺寸
                        area = image_size // c
                        if area * c == image_size:
                            # 尝试不同的尺寸
                            for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                                h = ratio_h * scale
                                w = ratio_w * scale
                                if h * w == area:
                                    self.image_height = h
                                    self.image_width = w
                                    self.image_channels = c
                                    print(f"自动推断图像尺寸: {h}x{w}x{c}")
                                    break
                            if self.image_height is not None:
                                break
                    if self.image_height is not None:
                        break
            
            if self.image_height is None:
                # 如果还是找不到匹配的尺寸，可能是压缩格式
                print(f"警告: 无法确定图像尺寸 ({image_size}像素)，可能是压缩格式")
                print("将尝试直接保存原始数据...")
                # 尝试保存为原始数据文件
                self.image_height = None
                self.image_width = None
                self.image_channels = None
            
            print(f"图像尺寸: {self.image_height} x {self.image_width} x {self.image_channels}")
            
            # 生成默认列名 - 支持14维度
            self.action_columns = [f'action_dim_{i}' for i in range(self.action_data.shape[1])]
            self.status_columns = [f'state_dim_{i}' for i in range(self.status_data.shape[1])]
            
            print(f"Action列数: {len(self.action_columns)}")
            print(f"Status列数: {len(self.status_columns)}")
            
            return True
            
        except Exception as e:
            print(f"加载processed_episode数据出错: {e}")
            return False
    
    def _load_ros_data_sync_data(self):
        """加载ros_data_sync格式数据"""
        try:
            # 读取action数据
            self.action_data = self.h5_file['action']['data'][:]
            print(f"Action数据形状: {self.action_data.shape}")
            
            # 读取camera数据
            self.camera_data = self.h5_file['camera']['data'][:]
            print(f"Camera数据形状: {self.camera_data.shape}")
            
            # 读取status数据
            self.status_data = self.h5_file['status']['data'][:]
            print(f"Status数据形状: {self.status_data.shape}")
            
            # 读取图像尺寸信息
            self.image_height = self.h5_file['camera'].attrs['image_height']
            self.image_width = self.h5_file['camera'].attrs['image_width']
            self.image_channels = self.h5_file['camera'].attrs['image_channels']
            print(f"图像尺寸: {self.image_height} x {self.image_width} x {self.image_channels}")
            
            # 读取列名
            if 'columns' in self.h5_file['action'].attrs:
                self.action_columns = self.h5_file['action'].attrs['columns']
                print(f"Action列名: {self.action_columns}")
            
            if 'columns' in self.h5_file['status'].attrs:
                self.status_columns = self.h5_file['status'].attrs['columns']
                print(f"Status列名: {self.status_columns}")
            
            return True
            
        except Exception as e:
            print(f"加载ros_data_sync数据出错: {e}")
            return False
    
    def _load_master_puppet_data(self):
        """加载master_puppet格式数据"""
        try:
            # 读取图像数据 - 参考h5_image_parser.py的实现
            # 实际图像数据路径是 /camera_observations/color_images/camera_head
            camera_path = '/camera_observations/color_images/camera_head'
            try:
                # 检查路径是否存在
                if camera_path in self.h5_file:
                    image_dataset = self.h5_file[camera_path]
                    num_frames = len(image_dataset)
                    print(f"找到图像数据集，总帧数: {num_frames}")
                    
                    # 读取第一帧以确定图像尺寸
                    if num_frames > 0:
                        try:
                            # 获取第一帧的编码图像数据
                            encoded_image = image_dataset[0]
                            if not isinstance(encoded_image, np.ndarray):
                                encoded_image = np.array(encoded_image, dtype=np.uint8)
                            
                            # 解码第一帧以获取图像尺寸
                            decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
                            if decoded_image is not None:
                                # cv2.imdecode返回BGR格式，转换为RGB
                                decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
                                self.image_height = decoded_image.shape[0]
                                self.image_width = decoded_image.shape[1]
                                self.image_channels = decoded_image.shape[2] if len(decoded_image.shape) == 3 else 1
                                print(f"从第一帧解码获取图像尺寸: {self.image_height} x {self.image_width} x {self.image_channels}")
                                
                                # 存储所有帧的编码图像数据（保持编码格式，后续需要时再解码）
                                # 由于是变长数组，我们需要逐个读取
                                encoded_images = []
                                for i in range(num_frames):
                                    encoded_img = image_dataset[i]
                                    if not isinstance(encoded_img, np.ndarray):
                                        encoded_img = np.array(encoded_img, dtype=np.uint8)
                                    encoded_images.append(encoded_img)
                                
                                # 将编码图像数据存储为列表（因为每帧的JPEG长度不同）
                                self.camera_data = encoded_images
                                print(f"已读取 {len(encoded_images)} 帧编码图像数据")
                            else:
                                print("警告: 第一帧图像解码失败")
                                self.camera_data = None
                                self.image_height = None
                                self.image_width = None
                                self.image_channels = None
                        except Exception as e:
                            print(f"警告: 读取/解码图像时出错: {e}")
                            self.camera_data = None
                            self.image_height = None
                            self.image_width = None
                            self.image_channels = None
                    else:
                        print("警告: 图像数据集为空")
                        self.camera_data = None
                        self.image_height = None
                        self.image_width = None
                        self.image_channels = None
                else:
                    print(f"警告: 未找到图像路径 {camera_path}")
                    # 尝试其他可能的路径
                    if 'camera_observations' in self.h5_file:
                        obs_keys = list(self.h5_file['camera_observations'].keys())
                        print(f"camera_observations下的键: {obs_keys}")
                    self.camera_data = None
                    self.image_height = None
                    self.image_width = None
                    self.image_channels = None
            except Exception as e:
                print(f"警告: 读取图像数据时出错: {e}")
                import traceback
                traceback.print_exc()
                self.camera_data = None
                self.image_height = None
                self.image_width = None
                self.image_channels = None
            
            # 读取Action数据：左臂关节角 + 左臂夹爪 + 右臂关节角 + 右臂夹爪
            action_parts = []
            
            # 左臂关节角
            try:
                left_arm = self.h5_file['master/arm_left_position_align/data'][:]
                action_parts.append(left_arm)
                print(f"左臂关节角形状: {left_arm.shape}")
            except KeyError:
                print("错误: 未找到 master/arm_left_position_align/data")
                return False
            
            # 左臂夹爪
            try:
                left_gripper = self.h5_file['master/end_effector_left_position_align/data'][:]
                action_parts.append(left_gripper)
                print(f"左臂夹爪形状: {left_gripper.shape}")
            except KeyError:
                print("错误: 未找到 master/end_effector_left_position_align/data")
                return False
            
            # 右臂关节角
            try:
                right_arm = self.h5_file['master/arm_right_position_align/data'][:]
                action_parts.append(right_arm)
                print(f"右臂关节角形状: {right_arm.shape}")
            except KeyError:
                print("错误: 未找到 master/arm_right_position_align/data")
                return False
            
            # 右臂夹爪
            try:
                right_gripper = self.h5_file['master/end_effector_right_position_align/data'][:]
                action_parts.append(right_gripper)
                print(f"右臂夹爪形状: {right_gripper.shape}")
            except KeyError:
                print("错误: 未找到 master/end_effector_right_position_align/data")
                return False
            
            # 拼接Action数据
            self.action_data = np.concatenate(action_parts, axis=1)
            print(f"Action数据形状: {self.action_data.shape}")
            
            # 读取State数据：左臂关节角(puppet) + 左臂夹爪(master复用) + 右臂关节角(puppet) + 右臂夹爪(master复用)
            state_parts = []
            
            # 左臂关节角 (puppet)
            try:
                left_arm_puppet = self.h5_file['puppet/arm_left_position_align/data'][:]
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
                right_arm_puppet = self.h5_file['puppet/arm_right_position_align/data'][:]
                state_parts.append(right_arm_puppet)
                print(f"右臂关节角(puppet)形状: {right_arm_puppet.shape}")
            except KeyError:
                print("错误: 未找到 puppet/arm_right_position_align/data")
                return False
            
            # 右臂夹爪 (复用master的)
            state_parts.append(right_gripper)
            print(f"右臂夹爪(复用master)形状: {right_gripper.shape}")
            
            # 拼接State数据
            self.status_data = np.concatenate(state_parts, axis=1)
            print(f"Status数据形状: {self.status_data.shape}")
            
            # 如果图像尺寸还没有设置（例如图像读取失败），尝试从元数据获取
            if self.image_height is None and self.camera_data is not None:
                # 尝试从camera_color_resolution获取图像尺寸
                try:
                    if 'camera_color_resolution' in self.h5_file:
                        resolution = self.h5_file['camera_color_resolution'][:]
                        if len(resolution) >= 2:
                            self.image_height = int(resolution[0])
                            self.image_width = int(resolution[1])
                            self.image_channels = 3  # RGB
                            print(f"从camera_color_resolution获取图像尺寸: {self.image_height} x {self.image_width} x {self.image_channels}")
                except Exception as e:
                    print(f"警告: 无法从camera_color_resolution获取图像尺寸: {e}")
            
            # 生成默认列名
            self.action_columns = [f'action_dim_{i}' for i in range(self.action_data.shape[1])]
            self.status_columns = [f'state_dim_{i}' for i in range(self.status_data.shape[1])]
            
            print(f"Action列数: {len(self.action_columns)}")
            print(f"Status列数: {len(self.status_columns)}")
            
            return True
            
        except Exception as e:
            print(f"加载master_puppet数据出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_output_dirs(self):
        """设置输出目录"""
        # 获取h5文件的目录和文件名
        h5_path = Path(self.h5_file_path)
        h5_dir = h5_path.parent
        h5_name = h5_path.stem  # 不带扩展名的文件名
        
        # 创建同名文件夹
        self.output_dir = h5_dir / f"{h5_name}_14d"
        self.pic_dir = self.output_dir / 'pic'
        self.motion_dir = self.output_dir / 'motion'
        
        # 创建目录
        self.output_dir.mkdir(exist_ok=True)
        self.pic_dir.mkdir(exist_ok=True)
        self.motion_dir.mkdir(exist_ok=True)
        
        print(f"输出目录: {self.output_dir}")
        print(f"图像目录: {self.pic_dir}")
        print(f"运动对比图目录: {self.motion_dir}")
    
    def save_images(self):
        """保存所有图像为PNG文件"""
        if self.camera_data is None:
            print("\n警告: 没有图像数据可保存")
            return
        
        # 检查camera_data是列表（编码图像）还是numpy数组（原始图像）
        if isinstance(self.camera_data, list):
            # 处理编码的JPEG图像列表
            print("\n开始保存编码图像（需要解码）...")
            num_images = len(self.camera_data)
            print(f"图像数据: {num_images} 帧编码图像")
            
            for i in range(num_images):
                try:
                    # 获取编码图像数据
                    encoded_image = self.camera_data[i]
                    if not isinstance(encoded_image, np.ndarray):
                        encoded_image = np.array(encoded_image, dtype=np.uint8)
                    
                    # 解码JPEG图像
                    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
                    if decoded_image is not None:
                        # cv2.imdecode返回BGR格式，转换为RGB
                        decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
                        
                        # 保存为PNG文件
                        image_filename = self.pic_dir / f"frame_{i:06d}.png"
                        # cv2.imwrite需要BGR格式
                        image_bgr = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)
                        success = cv2.imwrite(str(image_filename), image_bgr)
                        
                        if not success:
                            print(f"警告: 无法保存图像 {image_filename}")
                        
                        if (i + 1) % 100 == 0:
                            print(f"已保存 {i + 1}/{num_images} 张图像")
                    else:
                        print(f"警告: 第 {i} 帧图像解码失败")
                        
                except Exception as e:
                    print(f"保存第 {i} 张图像时出错: {e}")
                    continue
            
            print(f"完成！共保存 {num_images} 张图像到 {self.pic_dir}")
            return
        
        # 处理原始numpy数组格式的图像数据
        if self.image_height is None or self.image_width is None or self.image_channels is None:
            print("\n尝试保存压缩图像数据...")
            self.save_compressed_images()
            return
            
        print("\n开始保存图像...")
        
        num_images = self.camera_data.shape[0]
        print(f"图像数据形状: {self.camera_data.shape}")
        print(f"目标图像尺寸: {self.image_height} x {self.image_width} x {self.image_channels}")
        
        for i in range(num_images):
            try:
                # 获取扁平化的图像数据
                flat_image = self.camera_data[i]
                
                # 重塑为原始图像尺寸
                image = flat_image.reshape(
                    self.image_height, 
                    self.image_width, 
                    self.image_channels
                )
                
                # 确保图像数据类型正确
                if image.dtype != np.uint8:
                    # 如果数据是浮点数，假设范围是0-1，转换为0-255
                    if image.dtype in [np.float32, np.float64]:
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # 保存为PNG文件
                image_filename = self.pic_dir / f"frame_{i:06d}.png"
                success = cv2.imwrite(str(image_filename), image)
                
                if not success:
                    print(f"警告: 无法保存图像 {image_filename}")
                
                if (i + 1) % 100 == 0:
                    print(f"已保存 {i + 1}/{num_images} 张图像")
                    
            except Exception as e:
                print(f"保存第 {i} 张图像时出错: {e}")
                continue
        
        print(f"完成！共保存 {num_images} 张图像到 {self.pic_dir}")
    
    def save_compressed_images(self):
        """保存压缩图像数据"""
        print("\n开始保存压缩图像数据...")
        
        num_images = self.camera_data.shape[0]
        print(f"压缩图像数据形状: {self.camera_data.shape}")
        
        for i in range(num_images):
            try:
                # 获取压缩图像数据
                compressed_data = self.camera_data[i]
                
                # 尝试直接解码压缩数据
                if compressed_data.dtype == np.uint8:
                    # 尝试使用OpenCV解码
                    try:
                        # 假设数据是JPEG或PNG格式的压缩数据
                        image = cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)
                        if image is not None:
                            # 保存解码后的图像
                            image_filename = self.pic_dir / f"frame_{i:06d}.png"
                            success = cv2.imwrite(str(image_filename), image)
                            if success:
                                if (i + 1) % 100 == 0:
                                    print(f"已保存 {i + 1}/{num_images} 张图像")
                                continue
                    except:
                        pass
                
                # 如果解码失败，保存原始数据
                raw_filename = self.pic_dir / f"frame_{i:06d}.raw"
                with open(raw_filename, 'wb') as f:
                    f.write(compressed_data.tobytes())
                
                if (i + 1) % 100 == 0:
                    print(f"已保存 {i + 1}/{num_images} 张原始数据")
                    
            except Exception as e:
                print(f"保存第 {i} 张图像时出错: {e}")
                continue
        
        print(f"完成！共保存 {num_images} 张图像/原始数据到 {self.pic_dir}")
    
    def plot_dimension_comparison(self, dimension_idx):
        """
        绘制单个维度的action vs status对比图
        
        Args:
            dimension_idx: 维度索引 (0-19)
        """
        # 获取该维度的数据
        action_values = self.action_data[:, dimension_idx]
        status_values = self.status_data[:, dimension_idx]
        
        # 生成时间轴（帧数）
        frames = np.arange(len(action_values))
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制action和status
        plt.plot(frames, action_values, 'b-', label='Action', alpha=0.7, linewidth=1.5)
        plt.plot(frames, status_values, 'r-', label='Status', alpha=0.7, linewidth=1.5)
        
        # 获取列名
        action_col_name = self.action_columns[dimension_idx] if self.action_columns is not None else f'Action_{dimension_idx}'
        status_col_name = self.status_columns[dimension_idx] if self.status_columns is not None else f'Status_{dimension_idx}'
        
        # 设置标题和标签
        plt.title(f'Dimension {dimension_idx}: {action_col_name} vs {status_col_name}', fontsize=14)
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        plot_filename = self.motion_dir / f"dim_{dimension_idx:02d}_{action_col_name}_vs_{status_col_name}.png"
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
            num_dimensions = min(self.action_data.shape[1], self.status_data.shape[1])
            dimensions_to_plot = list(range(num_dimensions))
            print(f"绘制所有维度: {num_dimensions}个")
        
        for i, dim_idx in enumerate(dimensions_to_plot):
            if dim_idx >= self.action_data.shape[1] or dim_idx >= self.status_data.shape[1]:
                print(f"警告: 维度 {dim_idx} 超出数据范围，跳过")
                continue
                
            plot_filename = self.plot_dimension_comparison(dim_idx)
            print(f"已生成对比图 {i + 1}/{len(dimensions_to_plot)}: {plot_filename.name}")
        
        print(f"完成！共生成 {len(dimensions_to_plot)} 张对比图到 {self.motion_dir}")
    
    def create_summary_plot(self):
        """创建一个包含所有14维度的汇总图"""
        print("\n生成汇总对比图...")
        
        if self.specific_dimensions is not None:
            dimensions_to_plot = self.specific_dimensions
            title_suffix = f"Specific Dimensions {dimensions_to_plot}"
        else:
            num_dimensions = min(self.action_data.shape[1], self.status_data.shape[1])
            dimensions_to_plot = list(range(num_dimensions))
            title_suffix = f"All {num_dimensions} Dimensions"
        
        frames = np.arange(len(self.action_data))
        
        # 计算子图布局 - 针对14维度优化
        n_dims = len(dimensions_to_plot)
        if n_dims <= 4:
            rows, cols = 1, n_dims
        elif n_dims <= 8:
            rows, cols = 2, 4
        elif n_dims <= 12:
            rows, cols = 3, 4
        elif n_dims <= 14:
            rows, cols = 2, 7  # 2行7列，适合14维度
        else:
            rows, cols = (n_dims + 6) // 7, 7
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        fig.suptitle(f'Action vs Status - {title_suffix}', fontsize=16)
        
        # 确保axes是二维数组
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dim_idx in enumerate(dimensions_to_plot):
            if dim_idx >= self.action_data.shape[1] or dim_idx >= self.status_data.shape[1]:
                continue
            
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # 获取该维度的数据
            action_values = self.action_data[:, dim_idx]
            status_values = self.status_data[:, dim_idx]
            
            # 绘制
            ax.plot(frames, action_values, 'b-', label='Action', alpha=0.7, linewidth=1)
            ax.plot(frames, status_values, 'r-', label='Status', alpha=0.7, linewidth=1)
            
            # 获取列名
            action_col_name = self.action_columns[dim_idx] if self.action_columns is not None else f'Dim_{dim_idx}'
            
            # 设置标题
            ax.set_title(f'Dim {dim_idx}: {action_col_name}', fontsize=10)
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
        summary_filename = self.motion_dir / "summary_14d_dimensions.png"
        plt.savefig(str(summary_filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"汇总图已保存: {summary_filename}")
    
    def generate_statistics(self):
        """生成统计信息"""
        print("\n生成统计信息...")
        
        stats_filename = self.output_dir / "statistics_14d.txt"
        
        with open(stats_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HDF5数据统计信息 - 14维度版本（关节角数据）\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"数据文件: {self.h5_file_path}\n")
            f.write(f"总帧数: {len(self.action_data)}\n\n")
            
            f.write("Action数据形状: {}\n".format(self.action_data.shape))
            f.write("Status数据形状: {}\n".format(self.status_data.shape))
            # 处理camera_data可能是列表（编码图像）或numpy数组的情况
            if isinstance(self.camera_data, list):
                f.write("Camera数据: {} 帧编码图像（列表格式）\n".format(len(self.camera_data)))
            elif self.camera_data is not None:
                f.write("Camera数据形状: {}\n".format(self.camera_data.shape))
            else:
                f.write("Camera数据: 无\n")
            f.write("\n")
            
            f.write(f"图像尺寸: {self.image_height} x {self.image_width} x {self.image_channels}\n\n")
            
            # 计算每个维度的统计信息
            f.write("=" * 80 + "\n")
            f.write("各维度统计信息 (Action vs Status)\n")
            f.write("=" * 80 + "\n\n")
            
            if self.specific_dimensions is not None:
                dimensions_to_analyze = self.specific_dimensions
                f.write(f"分析特定维度: {dimensions_to_analyze}\n\n")
            else:
                num_dimensions = min(self.action_data.shape[1], self.status_data.shape[1])
                dimensions_to_analyze = list(range(num_dimensions))
                f.write(f"分析所有维度: {num_dimensions}个\n\n")
            
            for i in dimensions_to_analyze:
                if i >= self.action_data.shape[1] or i >= self.status_data.shape[1]:
                    f.write(f"维度 {i}: 超出数据范围，跳过\n\n")
                    continue
                    
                action_col = self.action_columns[i] if self.action_columns is not None else f'Dim_{i}'
                status_col = self.status_columns[i] if self.status_columns is not None else f'Dim_{i}'
                
                action_vals = self.action_data[:, i]
                status_vals = self.status_data[:, i]
                
                # 计算误差
                error = action_vals - status_vals
                mae = np.mean(np.abs(error))
                rmse = np.sqrt(np.mean(error ** 2))
                
                f.write(f"维度 {i}: {action_col} vs {status_col}\n")
                f.write(f"  Action - Mean: {np.mean(action_vals):.6f}, Std: {np.std(action_vals):.6f}, "
                       f"Min: {np.min(action_vals):.6f}, Max: {np.max(action_vals):.6f}\n")
                f.write(f"  Status - Mean: {np.mean(status_vals):.6f}, Std: {np.std(status_vals):.6f}, "
                       f"Min: {np.min(status_vals):.6f}, Max: {np.max(status_vals):.6f}\n")
                f.write(f"  误差   - MAE: {mae:.6f}, RMSE: {rmse:.6f}\n")
                f.write("\n")
        
        print(f"统计信息已保存: {stats_filename}")
    
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
        
        # 绘制所有维度的对比图
        self.plot_all_dimensions()
        
        # 创建汇总图
        self.create_summary_plot()
        
        # 生成统计信息
        self.generate_statistics()
        
        # 关闭HDF5文件
        if self.h5_file:
            self.h5_file.close()
        
        print("\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)
        print(f"输出目录: {self.output_dir}")
        print(f"  - 图像保存在: {self.pic_dir}")
        print(f"  - 运动对比图保存在: {self.motion_dir}")
        print(f"  - 统计信息: {self.output_dir / 'statistics_14d.txt'}")
        
        return True
    
    def __del__(self):
        """析构函数，确保HDF5文件被关闭"""
        if self.h5_file:
            self.h5_file.close()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python h5_read_14d_joint.py <h5_file_path> [dimensions]")
        print("示例:")
        print("  python h5_read_14d_joint.py processed_episode_0.hdf5")
        print("  python h5_read_14d_joint.py processed_episode_0.hdf5 0,1,2,3,4,5,6,7,8,9,10,11,12,13")
        print("  python h5_read_14d_joint.py processed_episode_0.hdf5 0,1,2,3,4,5,6")
        print("  python h5_read_14d_joint.py data/ros_synced_data_20241021_120000.h5")
        sys.exit(1)
    
    h5_file_path = sys.argv[1]
    
    # 解析特定维度参数
    specific_dimensions = None
    if len(sys.argv) > 2:
        dim_str = sys.argv[2]
        
        # 检查是否是预定义的维度组
        if dim_str.lower() == 'all_14d':
            # 所有14个维度
            specific_dimensions = list(range(14))
            print(f"所有14个维度: {specific_dimensions}")
        elif dim_str.lower() == 'first_7d':
            # 前7个维度
            specific_dimensions = list(range(7))
            print(f"前7个维度: {specific_dimensions}")
        elif dim_str.lower() == 'last_7d':
            # 后7个维度
            specific_dimensions = list(range(7, 14))
            print(f"后7个维度: {specific_dimensions}")
        else:
            try:
                specific_dimensions = [int(x.strip()) for x in dim_str.split(',')]
                print(f"指定维度: {specific_dimensions}")
            except ValueError:
                print("错误: 维度参数格式不正确，应为逗号分隔的整数，如: 0,1,2,3,4,5,6,7,8,9,10,11,12,13")
                print("或者使用预定义组: all_14d, first_7d, last_7d")
                sys.exit(1)
    
    # 检查文件是否存在
    if not os.path.exists(h5_file_path):
        print(f"错误: 文件不存在: {h5_file_path}")
        sys.exit(1)
    
    # 创建读取器并处理
    reader = H5DataReader14D(h5_file_path, specific_dimensions)
    success = reader.process()
    
    if success:
        print("\n数据处理成功完成！")
    else:
        print("\n数据处理失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()
