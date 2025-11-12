#!/usr/bin/env python3
"""
ROS2版本的inference_6d - 使用ROS话题数据替代H5文件
从 /camera/color/image_raw 获取图像
从 /tf 获取 wrist_roll_l_link 和 wrist_roll_r_link 相对base的pose
"""

import sys
import os
# 添加项目根目录到Python路径，以便导入hdt模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros
from tf2_ros import TransformException
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
import numpy as np
import argparse
import yaml
import cv2
import time
from threading import Lock

# torch和hdt只在推理模式需要，调试模式不需要
try:
    import torch
    import hdt.constants
    from hdt.modeling.utils import make_visual_encoder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    hdt = None


class ROSDataCollector(Node):
    """ROS2数据收集器 - 获取图像和TF数据"""
    
    def __init__(self, base_frame='base', debug=True):
        super().__init__('inference_6d_ros_node')
        
        self.base_frame = base_frame
        self.debug = debug
        
        # 数据存储
        self.latest_image = None
        self.latest_left_pose = None
        self.latest_right_pose = None
        self.latest_left_gripper = 1.0  # 左夹爪状态
        self.latest_right_gripper = 1.0  # 右夹爪状态
        self.data_lock = Lock()
        
        # 初始化
        self.bridge = CvBridge()
        
        # TF2 buffer, listener and broadcaster
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 匹配相机发布者的QoS配置
        # 相机使用: RELIABLE, KEEP_LAST, depth=1, VOLATILE
        from rclpy.qos import QoSDurabilityPolicy
        
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        self.get_logger().info("订阅图像话题 (RELIABLE QoS)...")
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            camera_qos
        )
        self.get_logger().info("✓ 订阅创建成功")
        
        # 订阅夹爪状态 (JointState消息)
        self.left_gripper_sub = self.create_subscription(
            JointState,
            '/gripper/left_status',
            self.left_gripper_callback,
            10
        )
        self.right_gripper_sub = self.create_subscription(
            JointState,
            '/gripper/right_status',
            self.right_gripper_callback,
            10
        )
        
        # 创建动作发布器
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
        
        # 创建夹爪控制发布器
        self.gripper_left_pub = self.create_publisher(
            JointState,
            '/gripper/left_command',
            10
        )
        self.gripper_right_pub = self.create_publisher(
            JointState,
            '/gripper/right_command',
            10
        )
        
        self.get_logger().info("ROSDataCollector initialized")
        self.get_logger().info(f"  - Subscribing to: /camera/color/image_raw")
        self.get_logger().info(f"  - Subscribing to: /gripper/left_status, /gripper/right_status")
        self.get_logger().info(f"  - Publishing to: /endposetarget_L, /endposetarget_R")
        self.get_logger().info(f"  - Publishing to: /gripper/left_command, /gripper/right_command")
        self.get_logger().info(f"  - Base frame: {base_frame}")
        self.get_logger().info(f"  - Looking for: wrist_roll_l_link, wrist_roll_r_link")
        
        # 等待TF数据准备好
        time.sleep(1.0)
        
    def left_gripper_callback(self, msg):
        """左夹爪状态回调函数
        从JointState消息中提取position[0]并乘以10
        """
        if len(msg.position) > 0:
            with self.data_lock:
                self.latest_left_gripper = msg.position[0] * 10.0
    
    def right_gripper_callback(self, msg):
        """右夹爪状态回调函数
        从JointState消息中提取position[0]并乘以10
        """
        if len(msg.position) > 0:
            with self.data_lock:
                self.latest_right_gripper = msg.position[0] * 10.0
    
    def image_callback(self, msg):
        """图像回调函数"""
        try:
            # 记录回调被调用
            if not hasattr(self, '_callback_count'):
                self._callback_count = 0
            self._callback_count += 1
            
            # 检测图像编码并转换为OpenCV BGR格式
            if msg.encoding == 'rgb8':
                # RGB转BGR
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif msg.encoding == 'bgr8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                # 尝试自动转换
                self.get_logger().warn(f"未知编码 {msg.encoding}，尝试转换为bgr8")
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 提取时间戳
            timestamp_sec = msg.header.stamp.sec
            timestamp_nanosec = msg.header.stamp.nanosec
            
            with self.data_lock:
                self.latest_image = cv_image
                self.latest_image_timestamp = (timestamp_sec, timestamp_nanosec)
                
            if self.debug:
                # ROS2没有throttle，这里简单限流
                if not hasattr(self, '_last_log_time'):
                    self._last_log_time = time.time()
                    self.get_logger().info(f"✓ 首次收到图像: {cv_image.shape}, 编码: {msg.encoding}")
                elif time.time() - self._last_log_time > 3.0:
                    self.get_logger().info(f"✓ 接收图像 #{self._callback_count}: {cv_image.shape}")
                    self._last_log_time = time.time()
                
        except Exception as e:
            self.get_logger().error(f"图像回调错误: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def get_tf_transform(self, target_frame, source_frame, timeout=0.5):
        """获取TF变换 (ROS2版本)"""
        try:
            # 获取最新的变换
            transform = self.tf_buffer.lookup_transform(
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
            
        except TransformException as e:
            if self.debug and not hasattr(self, '_tf_error_shown'):
                self.get_logger().warn(f"TF查找失败 {source_frame} → {target_frame}: {str(e)[:100]}")
                self._tf_error_shown = True
            return None
        except Exception as e:
            if self.debug:
                self.get_logger().error(f"TF获取异常: {e}")
            return None
    
    def get_current_data(self):
        """获取当前的图像、TF数据和夹爪状态"""
        # 获取左右手腕的TF
        left_pose = self.get_tf_transform('wrist_roll_l_link', self.base_frame)
        right_pose = self.get_tf_transform('wrist_roll_r_link', self.base_frame)
        
        # 获取图像、时间戳和夹爪状态
        with self.data_lock:
            image = self.latest_image.copy() if self.latest_image is not None else None
            img_timestamp = getattr(self, 'latest_image_timestamp', None)
            left_gripper = self.latest_left_gripper
            right_gripper = self.latest_right_gripper
        
        return {
            'image': image,
            'image_timestamp': img_timestamp,
            'left_pose': left_pose,
            'right_pose': right_pose,
            'left_gripper': left_gripper,
            'right_gripper': right_gripper,
            'timestamp': self.get_clock().now()
        }
    
    def publish_action(self, action):
        """
        发布动作到机器人（参考send_reset_pose.py的实现）
        action: 20维向量 [left_xyz(3), left_rot6d(6), left_gripper(1), 
                        right_xyz(3), right_rot6d(6), right_gripper(1)]
        """
        from scipy.spatial.transform import Rotation
        
        # 提取左臂动作
        left_xyz = action[0:3]
        left_rot6d = action[3:9]
        
        # 提取右臂动作
        right_xyz = action[10:13]
        right_rot6d = action[13:19]
        
        # 将6D旋转转换为四元数
        def rotation_6d_to_quaternion(rot6d):
            """
            将6D旋转转换为四元数（与训练代码的"错误"转换保持对称）
            
            训练时的转换：
              [qx,qy,qz,qw] -> [qw,qx,qy,qz] -> scipy当作[x,y,z,w] -> 旋转矩阵 -> 6D
            
            推理时的逆转换：
              6D -> 旋转矩阵 -> scipy.as_quat()得到[x,y,z,w]
              -> 这个[x,y,z,w]对应训练时的[qw,qx,qy,qz]
              -> 需要转回[qx,qy,qz,qw] = [y,z,w,x]
            """
            # 6D rotation是旋转矩阵前两列的flatten
            # reshape回(3, 2)矩阵
            rot_6d_matrix = rot6d.reshape(3, 2)
            
            # 提取两列
            col1 = rot_6d_matrix[:, 0]  # 第一列
            col2 = rot_6d_matrix[:, 1]  # 第二列
            
            # 归一化第一列
            col1_normalized = col1 / (np.linalg.norm(col1) + 1e-8)
            
            # Gram-Schmidt正交化第二列
            col2_orthogonal = col2 - np.dot(col1_normalized, col2) * col1_normalized
            col2_normalized = col2_orthogonal / (np.linalg.norm(col2_orthogonal) + 1e-8)
            
            # 计算第三列（叉积）
            col3 = np.cross(col1_normalized, col2_normalized)
            
            # 构建旋转矩阵
            rot_matrix = np.stack([col1_normalized, col2_normalized, col3], axis=1)
            
            # 转换为四元数
            rot = Rotation.from_matrix(rot_matrix)
            quat_scipy = rot.as_quat()  # scipy返回[x,y,z,w]
            
            # 逆向转换：[x,y,z,w] (scipy) -> [qx,qy,qz,qw] (ROS)
            # 因为训练时做了 [qw,qx,qy,qz] 当作 [x,y,z,w]
            # 所以现在的 [x,y,z,w] 实际对应 [qw,qx,qy,qz]
            # 需要转回 [qx,qy,qz,qw] = [y,z,w,x]
            quat_ros = np.array([
                quat_scipy[1],  # qx = y
                quat_scipy[2],  # qy = z
                quat_scipy[3],  # qz = w
                quat_scipy[0]   # qw = x
            ])
            
            return quat_ros
        
        left_quat = rotation_6d_to_quaternion(left_rot6d)
        right_quat = rotation_6d_to_quaternion(right_rot6d)
        
        # 获取当前时间戳
        time_now = self.get_clock().now()
        
        # ===== 发送左臂 =====
        # 创建 PoseStamped 消息
        left_ps = PoseStamped()
        left_ps.header.stamp = time_now.to_msg()
        left_ps.header.frame_id = 'endposetarget_L'  # 与send_reset_pose.py一致
        left_ps.pose.position.x = float(left_xyz[0])
        left_ps.pose.position.y = float(left_xyz[1])
        left_ps.pose.position.z = float(left_xyz[2])
        left_ps.pose.orientation.x = float(left_quat[0])
        left_ps.pose.orientation.y = float(left_quat[1])
        left_ps.pose.orientation.z = float(left_quat[2])
        left_ps.pose.orientation.w = float(left_quat[3])
        
        # 发布到话题
        self.endposetarget_L_pub.publish(left_ps)
        
        # 发送TF（与send_reset_pose.py一致）
        try:
            transform = TransformStamped()
            transform.header.frame_id = 'base_footprint'
            transform.header.stamp = time_now.to_msg()
            transform.child_frame_id = 'left_target'
            transform.transform.translation.x = float(left_xyz[0])
            transform.transform.translation.y = float(left_xyz[1])
            transform.transform.translation.z = float(left_xyz[2])
            transform.transform.rotation.x = float(left_quat[0])
            transform.transform.rotation.y = float(left_quat[1])
            transform.transform.rotation.z = float(left_quat[2])
            transform.transform.rotation.w = float(left_quat[3])
            self.tf_broadcaster.sendTransform(transform)
        except Exception as e:
            self.get_logger().error(f"发送left_target TF失败: {e}")
        
        # ===== 发送右臂 =====
        # 创建 PoseStamped 消息
        right_ps = PoseStamped()
        right_ps.header.stamp = time_now.to_msg()
        right_ps.header.frame_id = 'endposetarget_R'  # 与send_reset_pose.py一致
        right_ps.pose.position.x = float(right_xyz[0])
        right_ps.pose.position.y = float(right_xyz[1])
        right_ps.pose.position.z = float(right_xyz[2])
        right_ps.pose.orientation.x = float(right_quat[0])
        right_ps.pose.orientation.y = float(right_quat[1])
        right_ps.pose.orientation.z = float(right_quat[2])
        right_ps.pose.orientation.w = float(right_quat[3])
        
        # 发布到话题
        self.endposetarget_R_pub.publish(right_ps)
        
        # 发送TF（与send_reset_pose.py一致）
        try:
            transform = TransformStamped()
            transform.header.frame_id = 'base_footprint'
            transform.header.stamp = time_now.to_msg()
            transform.child_frame_id = 'right_target'
            transform.transform.translation.x = float(right_xyz[0])
            transform.transform.translation.y = float(right_xyz[1])
            transform.transform.translation.z = float(right_xyz[2])
            transform.transform.rotation.x = float(right_quat[0])
            transform.transform.rotation.y = float(right_quat[1])
            transform.transform.rotation.z = float(right_quat[2])
            transform.transform.rotation.w = float(right_quat[3])
            self.tf_broadcaster.sendTransform(transform)
        except Exception as e:
            self.get_logger().error(f"发送right_target TF失败: {e}")
        
        # ===== 发送夹爪控制 =====
        # 提取夹爪值
        left_gripper_value = action[9]   # 左夹爪
        right_gripper_value = action[19]  # 右夹爪
        
        # 发布左夹爪控制
        left_gripper_msg = JointState()
        left_gripper_msg.header.stamp = time_now.to_msg()
        left_gripper_msg.header.frame_id = "left_gripper"
        left_gripper_msg.name.append("left_gripper")
        left_gripper_msg.position.append(float(left_gripper_value))
        left_gripper_msg.velocity.append(0.0)
        left_gripper_msg.effort.append(0.0)
        self.gripper_left_pub.publish(left_gripper_msg)
        
        # 发布右夹爪控制
        right_gripper_msg = JointState()
        right_gripper_msg.header.stamp = time_now.to_msg()
        right_gripper_msg.header.frame_id = "right_gripper"
        right_gripper_msg.name.append("right_gripper")
        right_gripper_msg.position.append(float(right_gripper_value))
        right_gripper_msg.velocity.append(0.0)
        right_gripper_msg.effort.append(0.0)
        self.gripper_right_pub.publish(right_gripper_msg)
    
    def print_debug_info(self, data, save_image=False):
        """打印调试信息"""
        print("\n" + "="*70)
        print("ROS数据调试信息")
        print("="*70)
        
        # 图像信息 - 增强版
        if data['image'] is not None:
            img = data['image']
            print(f"✓ 图像数据:")
            print(f"  - Shape: {img.shape} (H={img.shape[0]}, W={img.shape[1]}, C={img.shape[2]})")
            print(f"  - Dtype: {img.dtype}")
            print(f"  - Range: [{img.min()}, {img.max()}]")
            print(f"  - Mean: {img.mean():.2f}")
            print(f"  - Std: {img.std():.2f}")
            
            # 检查图像是否为空或全黑
            if img.max() == 0:
                print(f"  ⚠️  警告: 图像全黑！")
            elif img.max() < 10:
                print(f"  ⚠️  警告: 图像非常暗，可能有问题")
            
            # BGR通道分析
            b_mean, g_mean, r_mean = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
            print(f"  - 通道均值: B={b_mean:.1f}, G={g_mean:.1f}, R={r_mean:.1f}")
            
            # 保存示例图像
            if save_image:
                import os
                save_dir = "/home/q/code/human-policy/debug_images"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "camera_sample.jpg")
                cv2.imwrite(save_path, img)
                print(f"  - 图像已保存到: {save_path}")
        else:
            print("✗ 图像数据: None")
        
        # 左手腕pose
        if data['left_pose'] is not None:
            print(f"✓ 左手腕 (wrist_roll_l_link):")
            print(f"  - Translation: {data['left_pose']['translation']}")
            print(f"  - Rotation (quat): {data['left_pose']['rotation']}")
        else:
            print("✗ 左手腕数据: None")
        
        # 左夹爪
        print(f"✓ 左夹爪状态: {data.get('left_gripper', 0.0):.3f}")
        
        # 右手腕pose
        if data['right_pose'] is not None:
            print(f"✓ 右手腕 (wrist_roll_r_link):")
            print(f"  - Translation: {data['right_pose']['translation']}")
            print(f"  - Rotation (quat): {data['right_pose']['rotation']}")
        else:
            print("✗ 右手腕数据: None")
        
        # 右夹爪
        print(f"✓ 右夹爪状态: {data.get('right_gripper', 0.0):.3f}")
        
        print(f"Timestamp: {data['timestamp']}")
        print("="*70 + "\n")


def quaternion_to_rotation_6d(quat):
    """
    将四元数转换为6D旋转表示
    quat: [qx, qy, qz, qw] - ROS/TF格式
    return: 6D rotation [r1, r2, r3, r4, r5, r6] (rotation matrix的前两列展平)
    
    注意：为了与训练数据的convert_robot_data_20d.py保持一致，
    需要将四元数转换为[qw, qx, qy, qz]格式给scipy
    """
    from scipy.spatial.transform import Rotation as R
    
    # 归一化四元数
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-8:
        # 使用单位四元数
        quat_normalized = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        quat_normalized = quat / quat_norm
    
    # 转换格式: [qx, qy, qz, qw] -> [qw, qx, qy, qz] 
    # （与convert_robot_data_20d.py保持一致）
    quat_scipy = np.array([
        quat_normalized[3],  # qw
        quat_normalized[0],  # qx
        quat_normalized[1],  # qy
        quat_normalized[2]   # qz
    ])
    
    try:
        # 转换为旋转矩阵
        rotation = R.from_quat(quat_scipy)
        rot_matrix = rotation.as_matrix()  # 3x3
        
        # 取前两列，展平为6D
        rot_6d = rot_matrix[:, :2].flatten()
    except Exception as e:
        print(f"警告：四元数转换失败，使用单位旋转: {e}")
        # 使用单位矩阵的前两列
        rot_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    return rot_6d


def pose_to_state_vector(left_pose, right_pose, left_gripper=0.0, right_gripper=0.0):
    """
    将左右手腕的pose和夹爪状态转换为20维状态向量
    格式: [left_xyz(3), left_rot6d(6), left_gripper(1), 
           right_xyz(3), right_rot6d(6), right_gripper(1)]
    
    Args:
        left_pose: 左手腕pose {'translation': [x,y,z], 'rotation': [qx,qy,qz,qw]}
        right_pose: 右手腕pose {'translation': [x,y,z], 'rotation': [qx,qy,qz,qw]}
        left_gripper: 左夹爪状态 (float)
        right_gripper: 右夹爪状态 (float)
    
    Returns:
        state: 20维numpy数组
    """
    state = np.zeros(20, dtype=np.float32)
    
    if left_pose is not None:
        # 左臂位置 (0-2)
        state[0:3] = left_pose['translation']
        # 左臂旋转6D (3-8)
        state[3:9] = quaternion_to_rotation_6d(left_pose['rotation'])
        # 左夹爪 (9)
        state[9] = left_gripper
    
    if right_pose is not None:
        # 右臂位置 (10-12)
        state[10:13] = right_pose['translation']
        # 右臂旋转6D (13-18)
        state[13:19] = quaternion_to_rotation_6d(right_pose['rotation'])
        # 右夹爪 (19)
        state[19] = right_gripper
    
    return state


def debug_mode(base_frame='base', max_frames=100):
    """调试模式 - 持续打印ROS数据并保存图像
    
    Args:
        base_frame: TF base frame名称
        max_frames: 最大保存帧数
    """
    rclpy.init()
    
    print("\n" + "="*70)
    print("ROS2数据调试模式 - 每帧保存图像")
    print("="*70)
    print(f"最大保存帧数: {max_frames}")
    print("保存位置: debug_images/")
    print("按 Ctrl+C 退出")
    print("="*70 + "\n")
    
    # 初始化数据收集器
    collector = ROSDataCollector(base_frame=base_frame, debug=True)
    
    # 等待数据并显示状态
    collector.get_logger().info("等待数据...")
    collector.get_logger().info("正在等待图像回调触发...")
    
    # 等待一段时间让数据流入
    for i in range(5):
        time.sleep(1.0)
        rclpy.spin_once(collector, timeout_sec=0.1)
        
        # 检查是否收到图像
        if hasattr(collector, '_callback_count'):
            collector.get_logger().info(f"✓ 已收到 {collector._callback_count} 帧图像")
            break
        else:
            collector.get_logger().warn(f"等待中... ({i+1}/5秒) - 尚未收到图像")
    
    if not hasattr(collector, '_callback_count'):
        collector.get_logger().error("⚠️  警告: 5秒后仍未收到图像数据！")
        collector.get_logger().error("可能的原因:")
        collector.get_logger().error("  1. QoS策略不匹配")
        collector.get_logger().error("  2. 相机节点未正常发布")
        collector.get_logger().error("  3. 话题名称错误")
        collector.get_logger().error("请检查: ros2 topic info /camera/color/image_raw")
    
    # 准备保存目录
    import os
    save_dir = "/home/q/code/human-policy/debug_images"
    os.makedirs(save_dir, exist_ok=True)
    print(f"图像保存目录: {save_dir}\n")
    
    # 计数器
    frame_count = 0
    saved_count = 0
    
    # 循环控制
    loop_rate = 0.5  # 每次循环等待0.5秒 (2Hz)
    
    try:
        # 首次TF检查
        collector.get_logger().info("检查TF数据...")
        test_left = collector.get_tf_transform('wrist_roll_l_link', collector.base_frame, timeout=0.5)
        test_right = collector.get_tf_transform('wrist_roll_r_link', collector.base_frame, timeout=0.5)
        
        if test_left is None or test_right is None:
            collector.get_logger().warn("⚠️  TF数据不可用，但程序会继续并保存图像")
            collector.get_logger().warn("检查: ros2 run tf2_ros tf2_echo base wrist_roll_l_link")
        else:
            collector.get_logger().info("✓ TF数据正常")
        
        print("开始主循环...\n")
        
        while rclpy.ok() and saved_count < max_frames:
            # Spin once to process callbacks
            rclpy.spin_once(collector, timeout_sec=0.1)
            
            # 获取当前数据
            data = collector.get_current_data()
            frame_count += 1
            
            # 每帧都保存图像
            if data['image'] is not None:
                # 获取时间戳
                if data['image_timestamp'] is not None:
                    ts_sec, ts_nanosec = data['image_timestamp']
                    timestamp_str = f"{ts_sec}.{ts_nanosec:09d}"
                    timestamp_float = ts_sec + ts_nanosec / 1e9
                else:
                    timestamp_str = "N/A"
                    timestamp_float = 0.0
                
                # 如果有TF数据，转换为状态向量并添加标注
                if data['left_pose'] is not None and data['right_pose'] is not None:
                    state = pose_to_state_vector(
                        data['left_pose'], 
                        data['right_pose'], 
                        data['left_gripper'], 
                        data['right_gripper']
                    )
                    
                    # 在图像上添加标注
                    img_vis = data['image'].copy()
                    h, w = img_vis.shape[:2]
                    
                    # 添加半透明背景
                    overlay = img_vis.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
                    img_vis = cv2.addWeighted(overlay, 0.3, img_vis, 0.7, 0)
                    
                    # 添加文字信息
                    y_pos = 25
                    cv2.putText(img_vis, f"Frame: {frame_count}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_pos += 25
                    cv2.putText(img_vis, f"Timestamp: {timestamp_str}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_pos += 30
                    cv2.putText(img_vis, f"Left: {data['left_pose']['translation'].round(3)}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    y_pos += 25
                    cv2.putText(img_vis, f"Right: {data['right_pose']['translation'].round(3)}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # 保存图像
                    save_path = os.path.join(save_dir, f"frame_{saved_count:04d}.jpg")
                    save_path_annotated = os.path.join(save_dir, f"frame_{saved_count:04d}_annotated.jpg")
                    cv2.imwrite(save_path, data['image'])
                    cv2.imwrite(save_path_annotated, img_vis)
                    
                    saved_count += 1
                    print(f"[{saved_count}/{max_frames}] frame_{saved_count-1:04d}.jpg | TS: {timestamp_str} | L: {data['left_pose']['translation'].round(2)} | R: {data['right_pose']['translation'].round(2)}")
                    
                else:
                    # 没有TF数据，保存带时间戳的原始图像
                    img_vis = data['image'].copy()
                    h, w = img_vis.shape[:2]
                    
                    # 添加半透明背景
                    overlay = img_vis.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
                    img_vis = cv2.addWeighted(overlay, 0.3, img_vis, 0.7, 0)
                    
                    # 添加时间戳
                    y_pos = 25
                    cv2.putText(img_vis, f"Frame: {frame_count}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_pos += 25
                    cv2.putText(img_vis, f"Timestamp: {timestamp_str}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_pos += 25
                    cv2.putText(img_vis, "No TF data", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    save_path = os.path.join(save_dir, f"frame_{saved_count:04d}_no_tf.jpg")
                    cv2.imwrite(save_path, data['image'])
                    save_path_annotated = os.path.join(save_dir, f"frame_{saved_count:04d}_no_tf_annotated.jpg")
                    cv2.imwrite(save_path_annotated, img_vis)
                    
                    saved_count += 1
                    print(f"[{saved_count}/{max_frames}] frame_{saved_count-1:04d}_no_tf.jpg | TS: {timestamp_str} | 无TF数据")
            else:
                # 没有图像数据
                if frame_count % 10 == 0:
                    print(f"[警告] 帧 {frame_count}: 无图像数据")
            
            # Sleep to maintain rate
            time.sleep(loop_rate)
            
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        collector.destroy_node()
        rclpy.shutdown()
        
        print("\n" + "="*70)
        print("采集完成")
        print("="*70)
        print(f"总帧数: {frame_count}")
        print(f"保存图像: {saved_count}")
        if saved_count > 0:
            print(f"保存位置: {save_dir}")
        print("="*70)


def load_policy(policy_path, policy_config_path, device):
    """加载策略模型 (从原inference_6d.py复制)"""
    if not TORCH_AVAILABLE:
        raise ImportError("推理模式需要安装torch。请运行: pip install torch")
    
    with open(policy_config_path, "r") as fp:
        policy_config = yaml.safe_load(fp)
    policy_type = policy_config["common"]["policy_class"]

    if policy_type == "ACT":
        try:
            policy = torch.jit.load(policy_path, map_location=device).eval().to(device)
            print("加载JIT traced模型成功")
            is_jit = True
        except (RuntimeError, Exception) as e:
            print(f"JIT加载失败: {e}")
            print("尝试加载普通checkpoint...")
            from hdt.policy import ACTPolicy
            
            act_config = {
                'lr': 1e-5,
                'num_queries': policy_config['common'].get('action_chunk_size', 100),
                'kl_weight': policy_config['model'].get('kl_weight', 10),
                'hidden_dim': policy_config['model'].get('hidden_dim', 512),
                'chunk_size': policy_config['common'].get('action_chunk_size', 100),
                'dim_feedforward': policy_config['model'].get('dim_feedforward', 3200),
                'lr_backbone': policy_config['model'].get('lr_backbone', 1e-5),
                'backbone': policy_config['model'].get('backbone', 'resnet18'),
                'enc_layers': policy_config['model'].get('enc_layers', 4),
                'dec_layers': policy_config['model'].get('dec_layers', 7),
                'nheads': policy_config['model'].get('nheads', 8),
                'camera_names': policy_config['common'].get('camera_names', ['left']),
                'state_dim': policy_config['common'].get('state_dim', 20),
                'action_dim': policy_config['common'].get('action_dim', 20),
                'image_feature_strategy': policy_config['model'].get('image_feature_strategy', 'linear'),
                'use_language_conditioning': policy_config['model'].get('use_language_conditioning', False),
            }
            
            policy = ACTPolicy(act_config)
            checkpoint = torch.load(policy_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            policy.model.load_state_dict(new_state_dict, strict=False)
            print(f"成功加载 {len(new_state_dict)} 个权重参数")
            
            policy = policy.to(device).eval()
            is_jit = False
            print("加载普通checkpoint成功")

        class PolicyWrapper(torch.nn.Module):
            def __init__(self, policy, is_jit=False):
                super().__init__()
                self.policy = policy
                self.is_jit = is_jit

            @torch.no_grad()
            def forward(self, image, qpos):
                if self.is_jit:
                    return self.policy(image, qpos)
                else:
                    empty_lang_embed = torch.load('hdt/empty_lang_embed.pt').float().to(qpos.device)
                    empty_lang_embed = empty_lang_embed.unsqueeze(0)
                    lang_mask = torch.ones(empty_lang_embed.shape[:2], dtype=torch.bool, device=qpos.device)
                    conditioning_dict = {
                        'language_embeddings': empty_lang_embed,
                        'language_embeddings_mask': lang_mask
                    }
                    return self.policy(image, qpos, conditioning_dict=conditioning_dict)
            
        my_policy_wrapper = PolicyWrapper(policy, is_jit=is_jit)
        my_policy_wrapper.eval().to(device)

        visual_encoder, visual_preprocessor = make_visual_encoder("ACT", policy_config)
        return my_policy_wrapper, visual_preprocessor
    else:
        raise ValueError(f"Policy type {policy_type} not yet supported in ROS version")


def get_norm_stats(data_path, embodiment_name="h1_inspire"):
    """加载归一化统计数据 (从原inference_6d.py复制)"""
    import pickle
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    
    with open(data_path, "rb") as f:
        try:
            norm_stats = pickle.load(f)
        except ModuleNotFoundError as e:
            print(f"警告: 遇到兼容性问题 ({e})")
            print("尝试使用兼容模式加载...")
            f.seek(0)
            import sys
            import numpy
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
            norm_stats = pickle.load(f)
    
    np.load = np_load_old
    
    available_embodiments = list(norm_stats.keys())
    print(f"可用的embodiment: {available_embodiments}")
    
    if embodiment_name in norm_stats:
        print(f"使用指定的embodiment: {embodiment_name}")
        norm_stats = norm_stats[embodiment_name]
    elif len(available_embodiments) == 1:
        embodiment_name = available_embodiments[0]
        print(f"自动使用唯一的embodiment: {embodiment_name}")
        norm_stats = norm_stats[embodiment_name]
    else:
        embodiment_name = available_embodiments[0]
        print(f"使用第一个embodiment: {embodiment_name}")
        norm_stats = norm_stats[embodiment_name]
    
    return norm_stats


def normalize_input(state, image, norm_stats, visual_preprocessor):
    """
    归一化输入数据
    Args:
        - state: np.array of shape (20,) - 20维状态向量
        - image: np.array of shape (H, W, 3) in BGR uint8 [0, 255]
        - norm_stats: dict with keys "qpos_mean", "qpos_std", "action_mean", "action_std"
        - visual_preprocessor: function that takes in BCHW UINT8 image and returns processed BCHW image
    """
    if not TORCH_AVAILABLE:
        raise ImportError("归一化输入需要安装torch。请运行: pip install torch")
    # 处理图像 - 转换为CHW格式
    image_chw = image.transpose((2, 0, 1))  # HWC -> CHW
    # 使用单相机，复制为双相机输入
    image_data = np.stack([image_chw, image_chw], axis=0)  # (2, C, H, W)
    
    image_data = visual_preprocessor(image_data).float()
    B, C, H, W = image_data.shape
    image_data = image_data.view((1, B, C, H, W)).to(device='cuda')

    # 处理状态数据
    qpos_data = torch.from_numpy(state).float().to(device='cuda')
    qpos_data = (qpos_data - norm_stats["qpos_mean"]) / (norm_stats["qpos_std"] + 1e-6)
    qpos_data = qpos_data.view((1, -1))

    return (qpos_data, image_data)


def inference_mode(model_path, model_cfg_path, norm_stats_path, base_frame='base', chunk_size=15, rate=10.0):
    """实时推理模式
    
    Args:
        model_path: 模型权重路径
        model_cfg_path: 模型配置路径
        norm_stats_path: 归一化统计路径
        base_frame: TF base frame名称
        chunk_size: 动作chunk大小
        rate: 推理频率 (Hz)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("推理模式需要安装PyTorch")
    
    # 初始化ROS2
    rclpy.init()
    
    print("\n" + "="*70)
    print("ROS2实时推理系统")
    print("="*70)
    
    # 加载模型
    print("加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    policy, visual_preprocessor = load_policy(model_path, model_cfg_path, device)
    print("✓ 模型加载成功")
    
    # 加载归一化统计
    print("加载归一化统计...")
    norm_stats = get_norm_stats(norm_stats_path)
    
    # 转换为torch tensor
    for key in ["qpos_mean", "qpos_std", "action_mean", "action_std"]:
        if key in norm_stats and not isinstance(norm_stats[key], torch.Tensor):
            norm_stats[key] = torch.from_numpy(norm_stats[key]).float().to(device)
    
    print("✓ 归一化统计加载成功")
    
    # 创建数据收集器
    print("创建ROS2节点...")
    collector = ROSDataCollector(base_frame=base_frame, debug=True)
    print("✓ ROS2节点创建成功")
    
    # 等待数据流入
    print("\n等待数据...")
    print("正在等待图像回调触发...")
    for i in range(5):
        time.sleep(1.0)
        rclpy.spin_once(collector, timeout_sec=0.1)
        
        # 检查是否收到图像
        if hasattr(collector, '_callback_count'):
            print(f"✓ 已收到 {collector._callback_count} 帧图像")
            break
        else:
            print(f"等待中... ({i+1}/5秒) - 尚未收到图像")
    
    if not hasattr(collector, '_callback_count'):
        print("⚠️  警告: 5秒后仍未收到图像数据！")
        print("可能的原因:")
        print("  1. 相机节点未正常发布")
        print("  2. QoS策略不匹配")
        print("请检查: ros2 topic hz /camera/color/image_raw")
        print("\n继续运行，等待数据...")
    
    # 检查TF
    print("检查TF数据...")
    test_left = collector.get_tf_transform('wrist_roll_l_link', collector.base_frame, timeout=0.5)
    test_right = collector.get_tf_transform('wrist_roll_r_link', collector.base_frame, timeout=0.5)
    
    if test_left is None or test_right is None:
        print("⚠️  TF数据不可用")
        print("检查: ros2 run tf2_ros tf2_echo base wrist_roll_l_link")
        print("\n继续运行...")
    else:
        print("✓ TF数据正常")
    
    print("\n开始推理循环...")
    print(f"推理频率: {rate} Hz")
    print(f"Chunk size: {chunk_size}")
    print("按 Ctrl+C 退出")
    print("="*70 + "\n")
    
    # 推理状态
    output = None
    act_index = 0
    count = 0
    last_inference_time = time.time()
    inference_interval = 1.0 / rate
    
    try:
        while rclpy.ok():
            # 处理ROS回调
            rclpy.spin_once(collector, timeout_sec=0.001)
            
            # 检查是否到了推理时间
            current_time = time.time()
            if current_time - last_inference_time >= inference_interval:
                # 获取当前数据
                data = collector.get_current_data()
                
                # 检查数据完整性
                if data['image'] is None:
                    if count % 100 == 0:
                        print(f"[警告 {count}] 无图像数据")
                    time.sleep(0.001)
                    continue
                
                if data['left_pose'] is None or data['right_pose'] is None:
                    if count % 100 == 0:
                        print(f"[警告 {count}] TF数据不完整")
                    time.sleep(0.001)
                    continue
                
                # 转换为状态向量（包含夹爪状态）
                state = pose_to_state_vector(
                    data['left_pose'], 
                    data['right_pose'], 
                    data['left_gripper'], 
                    data['right_gripper']
                )
                
                # 归一化
                qpos_data, image_data = normalize_input(
                    state, data['image'], norm_stats, visual_preprocessor
                )
                
                # 推理
                # 检查是否需要重新推理（每chunk_size-5帧）
                if output is None or act_index >= chunk_size - 5:
                    with torch.no_grad():
                        output = policy(image_data, qpos_data)[0].detach().cpu().numpy()
                        # 反归一化
                        output = output * norm_stats["action_std"].cpu().numpy() + norm_stats["action_mean"].cpu().numpy()
                        act_index = 0
                
                # 获取当前动作
                action = output[act_index]
                act_index += 1
                count += 1
                last_inference_time = current_time
                
                # 发布动作到机器人
                collector.publish_action(action)
                
                # 打印结果（完整的20维）
                if count % 10 == 0:
                    if data['image_timestamp'] is not None:
                        ts_sec, ts_nanosec = data['image_timestamp']
                        print(f"\n[推理 {count}] TS: {ts_sec}.{ts_nanosec:09d}")
                    else:
                        print(f"\n[推理 {count}]")
                    
                    # 打印完整的20维状态向量
                    print(f"  State (20D):")
                    print(f"    Left  - xyz: {state[0:3].round(3)}, rot6d: {state[3:9].round(3)}, gripper: {state[9]:.3f}")
                    print(f"    Right - xyz: {state[10:13].round(3)}, rot6d: {state[13:19].round(3)}, gripper: {state[19]:.3f}")
                    
                    # 打印完整的20维动作
                    print(f"  Action (20D):")
                    print(f"    Left  - xyz: {action[0:3].round(3)}, rot6d: {action[3:9].round(3)}, gripper: {action[9]:.3f}")
                    print(f"    Right - xyz: {action[10:13].round(3)}, rot6d: {action[13:19].round(3)}, gripper: {action[19]:.3f}")
                    
                    print(f"  ✓ 动作已发布到 /endposetarget_L 和 /endposetarget_R")
            else:
                # 短暂sleep避免CPU占用过高
                time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    
    finally:
        collector.destroy_node()
        rclpy.shutdown()
        
        print(f"\n推理完成. 总推理次数: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ROS2版本的inference_6d - 调试和实时推理', add_help=True)
    parser.add_argument('--debug', action='store_true', help='调试模式：每帧保存图像')
    parser.add_argument('--base_frame', type=str, default='base', help='base frame名称 (默认: base)')
    parser.add_argument('--max-frames', type=int, default=100, help='最大保存帧数 (默认: 100)')
    parser.add_argument('--norm_stats_path', type=str, help='归一化统计路径 (推理模式必需)')
    parser.add_argument('--model_path', type=str, help='模型权重路径 (推理模式必需)')
    parser.add_argument('--model_cfg_path', type=str, help='模型配置路径 (推理模式必需)')
    parser.add_argument('--chunk_size', type=int, help='chunk size', default=15)
    parser.add_argument('--rate', type=float, default=10.0, help='推理频率 (Hz, 默认: 10.0)')

    args = parser.parse_args()

    # 调试模式
    if args.debug:
        debug_mode(
            base_frame=args.base_frame, 
            max_frames=args.max_frames
        )
    else:
        # 推理模式需要torch
        if not TORCH_AVAILABLE:
            print("错误: 推理模式需要安装PyTorch")
            print("安装方法:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("\n如果只想调试数据获取，请使用: python3 cet/inference_6d_ros_realtime.py --debug")
            exit(1)
        
        # 检查必需参数
        if not args.model_path or not args.model_cfg_path or not args.norm_stats_path:
            print("错误: 推理模式需要提供以下参数:")
            print("  --model_path: 模型权重路径")
            print("  --model_cfg_path: 模型配置路径")
            print("  --norm_stats_path: 归一化统计路径")
            print("\n示例:")
            print("python3 cet/inference_6d_ros_realtime.py \\")
            print("  --model_path h_p_cpkt/ros_robot20_training_fixdim_b512_ckpt/policy_iter_120000_seed_0/pytorch_model.bin \\")
            print("  --model_cfg_path hdt/configs/models/act_resnet_vr_robot20.yaml \\")
            print("  --norm_stats_path h_p_cpkt/ros_robot20_training_fixdim_b512_ckpt/dataset_stats.pkl \\")
            print("  --chunk_size 15 \\")
            print("  --rate 10.0")
            exit(1)
        
        # 运行推理模式
        inference_mode(
            model_path=args.model_path,
            model_cfg_path=args.model_cfg_path,
            norm_stats_path=args.norm_stats_path,
            base_frame=args.base_frame,
            chunk_size=args.chunk_size,
            rate=args.rate
        )

