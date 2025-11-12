#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从h5文件读取指定帧的左右手四元数，并通过ROS2发送reset pose
使用方法:
python send_reset_pose.py ros_synced_data_20251023_214101.h5 --frame 10
"""

import h5py
import numpy as np
import argparse
import sys
import logging
from pathlib import Path

# ROS2导入（可选）
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, Pose
    from tf2_ros import TransformBroadcaster, TransformListener, Buffer
    from geometry_msgs.msg import TransformStamped
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
    import time
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    rclpy = None
    Node = None
    PoseStamped = None
    Pose = None
    TransformBroadcaster = None
    TransformListener = None
    Buffer = None
    TransformStamped = None
    R = None
    Slerp = None

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


if ROS2_AVAILABLE:
    class ResetPoseSenderNode(Node):
        """发送reset pose的ROS2节点"""
        
        def __init__(self, base_frame='base'):
            super().__init__('reset_pose_sender')
            
            self.base_frame = base_frame
            
            # 创建发布器
            self.endposetarget_L_pub = self.create_publisher(
                PoseStamped, 
                '/endposetarget_L', 
                10
            )
            
            self.endposetarget_R_pub = self.create_publisher(
                PoseStamped, 
                '/endposetarget_R', 
                10
            )
            
            # 创建TF broadcaster和listener
            self.tf_broadcaster = TransformBroadcaster(self)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            
            self.get_logger().info("ResetPoseSender节点已初始化")
            self.get_logger().info("  - 发布话题: /endposetarget_L")
            self.get_logger().info("  - 发布话题: /endposetarget_R")
            self.get_logger().info(f"  - Base frame: {self.base_frame}")
            self.get_logger().info("  - TF监听器已启动")
else:
    class ResetPoseSenderNode:
        """Dummy class when ROS2 is not available"""
        def __init__(self):
            raise ImportError("ROS2不可用，无法创建ResetPoseSenderNode")
        
class ResetPoseSender:
    """发送reset pose的包装类"""
    
    def __init__(self, base_frame='base'):
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2不可用，无法创建ResetPoseSender")
        self.node = ResetPoseSenderNode(base_frame=base_frame)
        
        # 等待TF数据
        time.sleep(1.0)
        self.node.get_logger().info("TF监听器准备就绪")
        
        # 尝试检测可用的TF frames
        self._detect_available_frames()
    
    def send_reset_pose(self, left_quat, right_quat, left_pos, right_pos):
        """
        发送reset pose
        
        Args:
            left_quat: 左手四元数 [x, y, z, w]
            right_quat: 右手四元数 [x, y, z, w]
            left_pos: 左手位置 [x, y, z]
            right_pos: 右手位置 [x, y, z]
        """
        time_now = self.node.get_clock().now()
        
        # 发送左手pose
        if left_quat is not None and left_pos is not None:
            left_pose = Pose()
            left_pose.position.x = float(left_pos[0])
            left_pose.position.y = float(left_pos[1])
            left_pose.position.z = float(left_pos[2])
            left_pose.orientation.x = float(left_quat[0])
            left_pose.orientation.y = float(left_quat[1])
            left_pose.orientation.z = float(left_quat[2])
            left_pose.orientation.w = float(left_quat[3])
            
            left_ps = PoseStamped()
            left_ps.header.stamp = time_now.to_msg()
            left_ps.header.frame_id = 'endposetarget_L'
            left_ps.pose = left_pose
            self.node.endposetarget_L_pub.publish(left_ps)
            
            # 发送TF
            try:
                transform = TransformStamped()
                transform.header.frame_id = 'base_footprint'
                transform.header.stamp = time_now.to_msg()
                transform.child_frame_id = 'left_target'
                transform.transform.translation.x = float(left_pose.position.x)
                transform.transform.translation.y = float(left_pose.position.y)
                transform.transform.translation.z = float(left_pose.position.z)
                transform.transform.rotation.x = float(left_pose.orientation.x)
                transform.transform.rotation.y = float(left_pose.orientation.y)
                transform.transform.rotation.z = float(left_pose.orientation.z)
                transform.transform.rotation.w = float(left_pose.orientation.w)
                self.node.tf_broadcaster.sendTransform(transform)
                
                self.node.get_logger().info("✓ 左手pose已发送")
                self.node.get_logger().info(f"  位置: [{left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f}]")
                self.node.get_logger().info(f"  四元数: [{left_quat[0]:.4f}, {left_quat[1]:.4f}, {left_quat[2]:.4f}, {left_quat[3]:.4f}]")
            except Exception as e:
                self.node.get_logger().error(f"❌ Error sending left_target TF: {e}")
        
        # 发送右手pose
        if right_quat is not None and right_pos is not None:
            right_pose = Pose()
            right_pose.position.x = float(right_pos[0])
            right_pose.position.y = float(right_pos[1])
            right_pose.position.z = float(right_pos[2])
            right_pose.orientation.x = float(right_quat[0])
            right_pose.orientation.y = float(right_quat[1])
            right_pose.orientation.z = float(right_quat[2])
            right_pose.orientation.w = float(right_quat[3])
            
            right_ps = PoseStamped()
            right_ps.header.stamp = time_now.to_msg()
            right_ps.header.frame_id = 'endposetarget_R'
            right_ps.pose = right_pose
            self.node.endposetarget_R_pub.publish(right_ps)
            
            # 发送TF
            try:
                time_now = self.node.get_clock().now()
                transform = TransformStamped()
                transform.header.frame_id = 'base_footprint'
                transform.header.stamp = time_now.to_msg()
                transform.child_frame_id = 'right_target'
                transform.transform.translation.x = float(right_pose.position.x)
                transform.transform.translation.y = float(right_pose.position.y)
                transform.transform.translation.z = float(right_pose.position.z)
                transform.transform.rotation.x = float(right_pose.orientation.x)
                transform.transform.rotation.y = float(right_pose.orientation.y)
                transform.transform.rotation.z = float(right_pose.orientation.z)
                transform.transform.rotation.w = float(right_pose.orientation.w)
                self.node.tf_broadcaster.sendTransform(transform)
                
                self.node.get_logger().info("✓ 右手pose已发送")
                self.node.get_logger().info(f"  位置: [{right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f}]")
                self.node.get_logger().info(f"  四元数: [{right_quat[0]:.4f}, {right_quat[1]:.4f}, {right_quat[2]:.4f}, {right_quat[3]:.4f}]")
            except Exception as e:
                self.node.get_logger().error(f"❌ Error sending right_target TF: {e}")
    
    def _detect_available_frames(self):
        """检测可用的TF frames"""
        try:
            # 等待一下确保TF数据已经接收
            time.sleep(0.5)
            
            # 尝试查找可用的frames
            self.node.get_logger().info("检测可用的TF frames...")
            
            # 常见的base frame名称
            possible_base_frames = ['base', 'base_link', 'base_footprint', 'world', 'odom']
            
            # 尝试查找左手腕
            for base in possible_base_frames:
                try:
                    transform = self.node.tf_buffer.lookup_transform(
                        base,
                        'wrist_roll_l_link',
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )
                    self.node.get_logger().info(f"✓ 找到可用的base frame: {base}")
                    self.node.base_frame = base
                    return
                except:
                    continue
            
            self.node.get_logger().warn(f"⚠️ 无法找到可用的base frame，将使用默认值: {self.node.base_frame}")
            
        except Exception as e:
            self.node.get_logger().warn(f"TF frame检测失败: {e}")
    
    def get_tf_transform(self, target_frame, source_frame, timeout=0.5):
        """
        获取TF变换（参考inference_6d_ros.py）
        
        Args:
            target_frame: 目标frame (例如 'wrist_roll_l_link')
            source_frame: 源frame (例如 'base')
            timeout: 超时时间（秒）
        
        Returns:
            dict: {'translation': [x,y,z], 'rotation': [x,y,z,w]} 或 None
        """
        try:
            # 获取最新的变换
            transform = self.node.tf_buffer.lookup_transform(
                source_frame,  # target frame (base)
                target_frame,  # source frame (wrist)
                rclpy.time.Time(),  # 使用最新可用的变换
                timeout=rclpy.duration.Duration(seconds=timeout)
            )
            
            # 提取位置和旋转
            trans = transform.transform.translation
            rot = transform.transform.rotation
            
            return {
                'translation': np.array([trans.x, trans.y, trans.z]),  # [x, y, z]
                'rotation': np.array([rot.x, rot.y, rot.z, rot.w])    # [qx, qy, qz, qw]
            }
            
        except Exception as e:
            # 只在第一次失败时打印警告，避免刷屏
            if not hasattr(self, '_tf_error_shown'):
                self.node.get_logger().warn(f"TF查找失败 {source_frame} → {target_frame}: {str(e)[:100]}")
                self._tf_error_shown = True
            return None
    
    def get_current_pose_from_tf(self, target_frame, base_frame=None):
        """
        从TF获取当前位姿
        
        Args:
            target_frame: 目标frame (例如 'wrist_roll_l_link' 或 'wrist_roll_r_link')
            base_frame: 基准frame (如果为None，使用节点的base_frame)
        
        Returns:
            tuple: (position [x,y,z], quaternion [x,y,z,w]) 或 None
        """
        if base_frame is None:
            base_frame = self.node.base_frame
        
        result = self.get_tf_transform(target_frame, base_frame)
        
        if result is None:
            return None
        
        return result['translation'], result['rotation']
    
    def interpolate_pose(self, start_pos, start_quat, end_pos, end_quat, alpha):
        """
        在两个位姿之间插值
        
        Args:
            start_pos: 起始位置 [x, y, z]
            start_quat: 起始四元数 [x, y, z, w]
            end_pos: 目标位置 [x, y, z]
            end_quat: 目标四元数 [x, y, z, w]
            alpha: 插值参数 (0.0 到 1.0)
        
        Returns:
            tuple: (插值后的位置, 插值后的四元数)
        """
        # 位置线性插值
        interp_pos = start_pos * (1.0 - alpha) + end_pos * alpha
        
        # 四元数球面线性插值 (SLERP)
        key_rots = R.from_quat([start_quat, end_quat])
        slerp = Slerp([0.0, 1.0], key_rots)
        interp_rot = slerp([alpha])[0]
        interp_quat = interp_rot.as_quat()
        
        return interp_pos, interp_quat
    
    def send_reset_pose_with_interpolation(self, target_left_pos, target_left_quat, 
                                          target_right_pos, target_right_quat,
                                          duration=3.0, hz=50):
        """
        使用插值平滑发送reset pose
        
        Args:
            target_left_pos: 目标左手位置 [x, y, z]
            target_left_quat: 目标左手四元数 [x, y, z, w]
            target_right_pos: 目标右手位置 [x, y, z]
            target_right_quat: 目标右手四元数 [x, y, z, w]
            duration: 插值总时长（秒）
            hz: 发送频率（Hz）
        """
        self.node.get_logger().info(f"开始插值运动 (时长: {duration}s, 频率: {hz}Hz)")
        
        # 获取当前位姿
        self.node.get_logger().info(f"正在从TF获取当前位姿 (base_frame: {self.node.base_frame})...")
        
        current_left = self.get_current_pose_from_tf('wrist_roll_l_link')
        current_right = self.get_current_pose_from_tf('wrist_roll_r_link')
        
        if current_left is None or current_right is None:
            self.node.get_logger().warn("⚠️ 无法获取当前TF，将直接发送目标位姿")
            # 直接发送目标位姿
            self.send_reset_pose(target_left_quat, target_right_quat, 
                               target_left_pos, target_right_pos)
            return
        
        start_left_pos, start_left_quat = current_left
        start_right_pos, start_right_quat = current_right
        
        self.node.get_logger().info(f"✓ 当前左手位置: [{start_left_pos[0]:.3f}, {start_left_pos[1]:.3f}, {start_left_pos[2]:.3f}]")
        self.node.get_logger().info(f"✓ 目标左手位置: [{target_left_pos[0]:.3f}, {target_left_pos[1]:.3f}, {target_left_pos[2]:.3f}]")
        self.node.get_logger().info(f"✓ 当前右手位置: [{start_right_pos[0]:.3f}, {start_right_pos[1]:.3f}, {start_right_pos[2]:.3f}]")
        self.node.get_logger().info(f"✓ 目标右手位置: [{target_right_pos[0]:.3f}, {target_right_pos[1]:.3f}, {target_right_pos[2]:.3f}]")
        
        # 计算插值步数
        dt = 1.0 / hz
        num_steps = int(duration / dt)
        
        self.node.get_logger().info(f"开始发送插值轨迹 ({num_steps} 步)...")
        
        start_time = time.time()
        
        for i in range(num_steps + 1):
            # 计算插值参数 (0.0 到 1.0)
            alpha = i / num_steps
            
            # 使用平滑的插值曲线 (S曲线)
            # alpha_smooth = 3*alpha^2 - 2*alpha^3
            alpha_smooth = 3 * alpha**2 - 2 * alpha**3
            
            # 插值左手位姿
            interp_left_pos, interp_left_quat = self.interpolate_pose(
                start_left_pos, start_left_quat,
                target_left_pos, target_left_quat,
                alpha_smooth
            )
            
            # 插值右手位姿
            interp_right_pos, interp_right_quat = self.interpolate_pose(
                start_right_pos, start_right_quat,
                target_right_pos, target_right_quat,
                alpha_smooth
            )
            
            # 发送插值后的位姿
            self.send_reset_pose(
                interp_left_quat, interp_right_quat,
                interp_left_pos, interp_right_pos
            )
            
            # 控制发送频率
            if i < num_steps:
                elapsed = time.time() - start_time
                target_time = (i + 1) * dt
                sleep_time = target_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 每隔一段时间打印进度
                if i % (hz // 2) == 0:  # 每0.5秒打印一次
                    progress = (i / num_steps) * 100
                    self.node.get_logger().info(f"进度: {progress:.1f}% (步骤 {i}/{num_steps})")
            
            # 处理ROS事件
            rclpy.spin_once(self.node, timeout_sec=0.001)
        
        total_time = time.time() - start_time
        self.node.get_logger().info(f"✓ 插值运动完成 (实际用时: {total_time:.2f}s)")
        self.node.get_logger().info("✓ 已到达目标位姿")


def extract_quaternion_from_h5(h5_file_path, frame_idx=10):
    """
    从h5文件中提取指定帧的左右手四元数和位置
    
    Args:
        h5_file_path: h5文件路径
        frame_idx: 帧索引（从0开始）
    
    Returns:
        dict: {
            'left_quat': [x, y, z, w],
            'right_quat': [x, y, z, w],
            'left_pos': [x, y, z],
            'right_pos': [x, y, z]
        }
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            logging.info(f"成功打开h5文件: {h5_file_path}")
            
            # 检查文件格式
            keys = list(f.keys())
            logging.info(f"h5文件包含的键: {keys}")
            
            # 尝试ros_data_sync格式
            if 'action' in keys and 'status' in keys:
                logging.info("检测到ros_data_sync格式")
                
                # 读取action数据
                action_data = f['action']['data'][:]
                status_data = f['status']['data'][:]
                
                logging.info(f"Action数据形状: {action_data.shape}")
                logging.info(f"Status数据形状: {status_data.shape}")
                
                # 检查帧索引是否有效
                if frame_idx >= len(action_data):
                    logging.error(f"帧索引{frame_idx}超出范围，最大索引为{len(action_data)-1}")
                    return None
                
                # 读取action列名（如果有）
                if 'columns' in f['action'].attrs:
                    action_columns = f['action'].attrs['columns']
                    logging.info(f"Action列名: {action_columns}")
                
                if 'columns' in f['status'].attrs:
                    status_columns = f['status'].attrs['columns']
                    logging.info(f"Status列名: {status_columns}")
                
                # 获取第frame_idx帧的数据
                # 使用status数据（实际测量值）
                frame_data = status_data[frame_idx]
                
                logging.info(f"\n第{frame_idx}帧数据:")
                logging.info(f"完整数据维度: {len(frame_data)}")
                logging.info(f"数据: {frame_data}")
                
                # 数据格式: [left_data(0-6), left_gripper(7), right_data(8-14), right_gripper(15)]
                # 其中 left_data 和 right_data 是 xyz(3) + quat(4) = 7维
                # 总共16维
                
                if len(frame_data) >= 16:
                    # 提取左手数据（索引 0-6）
                    left_pos = frame_data[0:3]      # 左手位置 xyz
                    left_quat = frame_data[3:7]     # 左手四元数 xyzw
                    
                    # 提取右手数据（索引 8-14）
                    right_pos = frame_data[8:11]    # 右手位置 xyz
                    right_quat = frame_data[11:15]  # 右手四元数 xyzw
                    
                    result = {
                        'left_pos': left_pos,
                        'left_quat': left_quat,
                        'right_pos': right_pos,
                        'right_quat': right_quat
                    }
                    
                    logging.info(f"\n提取的第{frame_idx}帧数据:")
                    logging.info(f"左手位置: {left_pos}")
                    logging.info(f"左手四元数 [x,y,z,w]: {left_quat}")
                    logging.info(f"右手位置: {right_pos}")
                    logging.info(f"右手四元数 [x,y,z,w]: {right_quat}")
                    
                    return result
                else:
                    logging.error(f"数据维度不足16，实际为{len(frame_data)}")
                    return None
                    
            else:
                logging.error(f"未知的h5文件格式，键: {keys}")
                return None
                
    except Exception as e:
        logging.error(f"读取h5文件出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从h5文件读取指定帧的四元数并发送reset pose')
    parser.add_argument('h5_file', type=str, help='h5文件路径')
    parser.add_argument('--frame', type=int, default=10, help='要读取的帧索引（默认为10）')
    parser.add_argument('--loop', action='store_true', help='循环发送pose（每秒发送一次）')
    parser.add_argument('--interpolate', action='store_true', help='使用插值平滑发送（从当前TF位置）')
    parser.add_argument('--duration', type=float, default=3.0, help='插值运动时长（秒，默认3.0）')
    parser.add_argument('--hz', type=int, default=50, help='发送频率（Hz，默认50）')
    parser.add_argument('--base-frame', type=str, default='base', help='TF base frame名称（默认base）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        logging.error(f"文件不存在: {args.h5_file}")
        sys.exit(1)
    
    # 从h5文件提取数据
    logging.info(f"\n从h5文件读取第{args.frame}帧的四元数...")
    data = extract_quaternion_from_h5(args.h5_file, args.frame)
    
    if data is None:
        logging.error("提取数据失败")
        sys.exit(1)
    
    # 初始化ROS2
    rclpy.init()
    
    # 创建发送节点
    sender = ResetPoseSender(base_frame=args.base_frame)
    
    try:
        if args.loop:
            # 循环模式：每秒发送一次
            logging.info("\n循环发送模式（每秒发送一次），按Ctrl+C退出...")
            
            while rclpy.ok():
                sender.send_reset_pose(
                    data['left_quat'],
                    data['right_quat'],
                    data['left_pos'],
                    data['right_pos']
                )
                time.sleep(1.0)
                rclpy.spin_once(sender.node, timeout_sec=0.1)
        
        elif args.interpolate:
            # 插值模式：平滑运动到目标位姿
            logging.info("\n插值发送模式...")
            logging.info(f"  时长: {args.duration}s")
            logging.info(f"  频率: {args.hz}Hz")
            
            sender.send_reset_pose_with_interpolation(
                data['left_pos'],
                data['left_quat'],
                data['right_pos'],
                data['right_quat'],
                duration=args.duration,
                hz=args.hz
            )
            
            logging.info("\n✓ 插值运动完成")
        
        else:
            # 单次发送
            logging.info("\n发送reset pose（直接发送）...")
            sender.send_reset_pose(
                data['left_quat'],
                data['right_quat'],
                data['left_pos'],
                data['right_pos']
            )
            
            # 保持节点运行一段时间以确保消息被发送
            for _ in range(5):
                rclpy.spin_once(sender.node, timeout_sec=0.1)
                time.sleep(0.1)
            
            logging.info("\n✓ Reset pose发送完成")
    
    except KeyboardInterrupt:
        logging.info("\n用户中断")
    finally:
        sender.node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

