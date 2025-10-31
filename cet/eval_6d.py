import h5py
import pickle
import torch
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import yaml
import hdt.constant_20d as hdt_constants  # 直接使用20维constants
from hdt.modeling.utils import make_visual_encoder

def get_norm_stats(data_path, embodiment_name="h1_inspire"):
    # 使用numpy的兼容模式加载pickle文件
    import numpy as np
    np_load_old = np.load
    # 修改默认参数以兼容旧版本
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    
    with open(data_path, "rb") as f:
        try:
            norm_stats = pickle.load(f)
        except ModuleNotFoundError as e:
            print(f"警告: 遇到兼容性问题 ({e})")
            print("尝试使用兼容模式加载...")
            f.seek(0)
            # 使用pickle5或修复numpy命名空间
            import sys
            import numpy
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
            norm_stats = pickle.load(f)
    
    # 恢复原始设置
    np.load = np_load_old
    
    # 自动检测可用的embodiment
    available_embodiments = list(norm_stats.keys())
    print(f"可用的embodiment: {available_embodiments}")
    
    if embodiment_name in norm_stats:
        print(f"使用指定的embodiment: {embodiment_name}")
        norm_stats = norm_stats[embodiment_name]
    elif len(available_embodiments) == 1:
        # 如果只有一个embodiment，自动使用它
        embodiment_name = available_embodiments[0]
        print(f"自动使用唯一的embodiment: {embodiment_name}")
        norm_stats = norm_stats[embodiment_name]
    else:
        # 尝试使用第一个包含'human'或'robot'的embodiment
        for emb in available_embodiments:
            if 'human' in emb.lower() or 'robot' in emb.lower():
                print(f"自动选择embodiment: {emb}")
                norm_stats = norm_stats[emb]
                break
        else:
            # 使用第一个
            embodiment_name = available_embodiments[0]
            print(f"使用第一个embodiment: {embodiment_name}")
            norm_stats = norm_stats[embodiment_name]
    
    return norm_stats

def load_policy(policy_path, policy_config_path, device):
    with open(policy_config_path, "r") as fp:
        policy_config = yaml.safe_load(fp)
    policy_type = policy_config["common"]["policy_class"]
    
    # 使用20维constants
    state_dim = policy_config["common"].get("state_dim", 20)
    print(f"✓ 使用 constant_20d (state_dim={state_dim})")

    if policy_type == "ACT":
        # 尝试加载JIT traced模型或普通checkpoint
        try:
            # 首先尝试JIT格式
            policy = torch.jit.load(policy_path, map_location=device).eval().to(device)
            print("加载JIT traced模型成功")
            is_jit = True
        except (RuntimeError, pickle.UnpicklingError) as e:
            print(f"JIT加载失败: {e}")
            print("尝试加载普通checkpoint...")
            # 加载普通checkpoint格式的ACT模型
            from hdt.policy import ACTPolicy
            
            # 将字典配置转换为ACTPolicy需要的格式
            # policy_config是嵌套字典，需要展平为ACTPolicy可以使用的格式
            act_config = {
                'lr': 1e-5,  # 推理时学习率不重要
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
            
            # 创建ACT策略
            policy = ACTPolicy(act_config)
            
            # 加载权重
            checkpoint = torch.load(policy_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # 直接是state dict
                state_dict = checkpoint
            
            # 去掉"model."前缀（如果存在）
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v  # 去掉"model."前缀
                else:
                    new_state_dict[k] = v
            
            policy.model.load_state_dict(new_state_dict, strict=False)
            print(f"成功加载 {len(new_state_dict)} 个权重参数")
            
            policy = policy.to(device).eval()
            is_jit = False
            print("加载普通checkpoint成功")

        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy, is_jit=False):
                super().__init__()
                self.policy = policy
                self.is_jit = is_jit

            @torch.no_grad()
            def forward(self, image, qpos):
                if self.is_jit:
                    return self.policy(image, qpos)
                else:
                    # 对于ACTPolicy，需要传入conditioning_dict
                    empty_lang_embed = torch.load('hdt/empty_lang_embed.pt').float().to(qpos.device)
                    empty_lang_embed = empty_lang_embed.unsqueeze(0)  # (1, seq_len, embed_dim)
                    
                    # 创建language mask (全True表示有效)
                    lang_mask = torch.ones(empty_lang_embed.shape[:2], dtype=torch.bool, device=qpos.device)
                    
                    conditioning_dict = {
                        'language_embeddings': empty_lang_embed,
                        'language_embeddings_mask': lang_mask
                    }
                    return self.policy(image, qpos, conditioning_dict=conditioning_dict)
            
        my_policy_wrapper = polciy_wrapper(policy, is_jit=is_jit)
        my_policy_wrapper.eval().to(device)

        visual_encoder, visual_preprocessor = make_visual_encoder("ACT", policy_config)
        return my_policy_wrapper, visual_preprocessor
    elif policy_type == "RDT":
        visual_encoder, visual_preprocessor = make_visual_encoder("RDT", {"visual_backbone": "MASKCLIP"})

        from hdt.modeling.modeling_hdt import HumanDiffusionTransformer
        policy = HumanDiffusionTransformer(
            action_dim=policy_config["common"]["state_dim"],
            pred_horizon=policy_config["common"]["action_chunk_size"],
            config=policy_config,
            lang_token_dim=policy_config["model"]["lang_token_dim"],
            img_token_dim=policy_config["model"]["img_token_dim"],
            state_token_dim=policy_config["model"]["state_token_dim"],
            max_lang_cond_len=policy_config["dataset"]["tokenizer_max_length"],
            visual_encoder=visual_encoder,
            lang_pos_embed_config=[
                # Similarly, no initial pos embed for language
                ("lang", -policy_config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=torch.float32,
        )
        checkpoint = torch.load(policy_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        policy.load_state_dict(new_state_dict, strict=True)
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos):
                return self.policy(image, qpos, conditioning_dict={})
    
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().to(device)

        return my_policy_wrapper, visual_preprocessor
    elif policy_type == "DP":
        visual_encoder, visual_preprocessor = make_visual_encoder("DP", {"visual_backbone": "MASKCLIP"})
        
        from hdt.modeling.modeling_vanilla_dp import DiffusionPolicy
        policy = DiffusionPolicy(action_dim=128,
            chunk_size=64,
            img_token_dim=visual_encoder.hidden_size,
            state_token_dim=128,
            num_inference_timesteps=20,
            visual_encoder=visual_encoder)
        checkpoint = torch.load(policy_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        policy.load_state_dict(new_state_dict, strict=True)
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos, cond_dict):
                return self.policy(image, qpos, conditioning_dict=cond_dict)
    
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().cuda()

        return my_policy_wrapper, visual_preprocessor
    else:
        raise ValueError("Invalid policy type: {}".format(policy_type))

def normalize_input(state, left_img_chw, right_img_chw, norm_stats, visual_preprocessor, match_human=False):
    """
    Args
        - state: np.array of shape (N,) e.g., 20 for 20D data, 26 for H1 data, etc.
        - left_img_chw: np.array of shape (3, H, W) in uint8 [0, 255]
        - right_img_chw: np.array of shape (3, H, W) in uint8 [0, 255] or None
        - norm_stats: dict with keys "qpos_mean", "qpos_std", "action_mean", "action_std"
        - visual_preprocessor: function that takes in BCHW UINT8 image and returns processed BCHW image
        - match_human: if True, use state directly; if False, pad with zeros to match expected size
    """
    # 处理图像数据
    if right_img_chw is None:
        # 单相机情况：复制左相机图像作为右相机（或根据需要处理）
        image_data = np.stack([left_img_chw, left_img_chw], axis=0)
    else:
        image_data = np.stack([left_img_chw, right_img_chw], axis=0)
    
    image_data = visual_preprocessor(image_data).float()
    B, C, H, W = image_data.shape
    image_data = image_data.view((1, B, C, H, W)).to(device='cuda')

    # 处理状态数据 - 自动适配维度
    state_dim = state.shape[0]
    expected_dim = len(norm_stats["qpos_mean"])
    
    if match_human or state_dim == expected_dim:
        # 直接使用状态数据（20维或其他匹配维度）
        qpos_data = torch.from_numpy(state).float()
    else:
        # 需要填充到预期维度
        qpos_data = torch.zeros(expected_dim, dtype=torch.float32)
        if hasattr(hdt_constants, 'QPOS_INDICES') and len(hdt_constants.QPOS_INDICES) > 0:
            qpos_data[hdt_constants.QPOS_INDICES] = torch.from_numpy(state).float()
        else:
            # 如果没有QPOS_INDICES，直接复制前N维
            qpos_data[:state_dim] = torch.from_numpy(state).float()
    
    qpos_data = (qpos_data - norm_stats["qpos_mean"]) / (norm_stats["qpos_std"] + 1e-6)
    qpos_data = qpos_data.view((1, -1)).to(device='cuda')

    return (qpos_data, image_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval HDT policy on figures w/ optional sim visualization', add_help=False)
    parser.add_argument('--hdf_file_path', type=str, help='hdf file path', required=True)
    parser.add_argument('--norm_stats_path', type=str, help='norm stats path', required=True)
    parser.add_argument('--model_path', type=str, help='model path', required=True)
    parser.add_argument('--lang_embeddings_path', type=str, help='lang embeddings path', required=True)
    parser.add_argument('--chunk_size', type=int, help='chunk size', default=64)
    parser.add_argument('--model_cfg_path', type=str, help='path to model cfg yaml', required=True)
    parser.add_argument('--plot', action='store_true')

    args = vars(parser.parse_args())

    chunk_size = args['chunk_size']
    device = "cuda"

    with h5py.File(args['hdf_file_path'], 'r') as data:
        actions = np.array(data['action'])
        left_imgs = np.array(data['observation.image.left'])
        states = np.array(data['observation.state'])
        
        # 检查是否有右相机数据
        has_right_camera = 'observation.image.right' in data
        if has_right_camera:
            right_imgs = np.array(data['observation.image.right'])
            print("检测到双相机数据")
        else:
            right_imgs = None
            print("检测到单相机数据（仅左相机）")
        
        print(f"数据形状:")
        print(f"  - actions: {actions.shape}")
        print(f"  - states: {states.shape}")
        print(f"  - left_imgs: {left_imgs.shape}")
        if has_right_camera:
            print(f"  - right_imgs: {right_imgs.shape}")

        # 处理压缩图像
        if len(left_imgs.shape) == 2:
            # compressed images
            print("解压缩JPEG图像数据...")
            left_img_list = []
            right_img_list = []
            
            for i in range(left_imgs.shape[0]):
                # 解码左相机图像
                left_compressed = left_imgs[i]
                # 找到实际数据长度
                nonzero_indices = np.nonzero(left_compressed)[0]
                if len(nonzero_indices) > 0:
                    actual_length = nonzero_indices[-1] + 1
                    left_img = cv2.imdecode(left_compressed[:actual_length], cv2.IMREAD_COLOR)
                else:
                    left_img = None
                
                if left_img is None:
                    # 创建黑色图像
                    left_img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                left_img_list.append(left_img.transpose((2, 0, 1)))
                
                # 处理右相机
                if has_right_camera:
                    right_compressed = right_imgs[i]
                    nonzero_indices = np.nonzero(right_compressed)[0]
                    if len(nonzero_indices) > 0:
                        actual_length = nonzero_indices[-1] + 1
                        right_img = cv2.imdecode(right_compressed[:actual_length], cv2.IMREAD_COLOR)
                    else:
                        right_img = None
                    
                    if right_img is None:
                        right_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    right_img_list.append(right_img.transpose((2, 0, 1)))
                else:
                    # 复制左相机作为右相机
                    right_img_list.append(left_img_list[-1].copy())
                
                if (i + 1) % 100 == 0:
                    print(f"已解压 {i+1}/{left_imgs.shape[0]} 帧")
            
            # BCHW format
            left_imgs = np.stack(left_img_list, axis=0)
            right_imgs = np.stack(right_img_list, axis=0)
            print(f"解压后图像形状: left={left_imgs.shape}, right={right_imgs.shape}")

        init_action = actions[0]
        init_left_img = left_imgs[0]
        init_right_img = right_imgs[0] if has_right_camera else left_imgs[0]

    norm_stats = get_norm_stats(args['norm_stats_path'])

    policy, visual_preprocessor = load_policy(args['model_path'], args['model_cfg_path'], device)

    # Reset robot and the environment
    output = None
    act = None
    act_index = 0

    if args['plot']:
        predicted_list = []
        gt_list = []
        record_list = []

    for t in tqdm(range(states.shape[0])):
            print("step", t)
            t_start = t

            # Select offseted episodes
            cur_action = actions[t_start]
            cur_left_img = left_imgs[t_start]
            cur_right_img = right_imgs[t_start]
            # cur_state = states[t_start][hdt.constants.QPOS_INDICES]

            cur_state = states[t_start]

            qpos_data, image_data = normalize_input(cur_state, cur_left_img, cur_right_img, norm_stats, visual_preprocessor, match_human = True)

            #! here to mask data (optional - comment out for 20D data)
            # 对于20维数据，通常不需要mask
            # 如果是128维数据，可以取消下面的注释来mask某些维度
            # if hasattr(hdt.constants, 'OUTPUT_LEFT_KEYPOINTS'):
            #     qpos_data[:, hdt.constants.OUTPUT_LEFT_KEYPOINTS] = 0
            # if hasattr(hdt.constants, 'OUTPUT_RIGHT_KEYPOINTS'):
            #     qpos_data[:, hdt.constants.OUTPUT_RIGHT_KEYPOINTS] = 0
            # if hasattr(hdt.constants, 'OUTPUT_HEAD_EEF'):
            #     qpos_data[:, hdt.constants.OUTPUT_HEAD_EEF] = 0

            if output is None or act_index == chunk_size - 5:
                output = policy(image_data, qpos_data)[0].detach().cpu().numpy() # (chuck_size,action_dim)
                output = output * norm_stats["action_std"] + norm_stats["action_mean"]
                act_index = 0
            act = output[act_index]
            act_index += 1

            if args['plot']:
                # 保存所有维度的数据
                predicted_list.append(act)
                gt_list.append(cur_action)

            # 可视化图像（注释掉以加快推理速度）
            # if cur_right_img is not None and not np.array_equal(cur_left_img, cur_right_img):
            #     # 双相机：并排显示
            #     img = np.concatenate((cur_left_img.transpose((1, 2, 0)), cur_right_img.transpose((1, 2, 0))), axis=1)
            #     title = 'Left & Right Camera View'
            # else:
            #     # 单相机：只显示左相机
            #     img = cur_left_img.transpose((1, 2, 0))
            #     title = 'Left Camera View'
            # 
            # plt.cla()
            # plt.title(title)
            # plt.imshow(img, aspect='equal')
            # plt.pause(0.001)

            act = act.astype(np.float32)

    if args['plot']:
        print("绘制所有维度的对比图...")
        
        # 转换为numpy数组
        predicted_array = np.array(predicted_list)  # (T, action_dim)
        gt_array = np.array(gt_list)  # (T, action_dim)
        
        num_dims = predicted_array.shape[1]
        print(f"总共 {num_dims} 个维度")
        
        # 计算每个维度的误差
        errors = np.abs(predicted_array - gt_array)
        mae_per_dim = np.mean(errors, axis=0)
        
        # 20维数据的维度标签
        dim_labels = []
        if num_dims == 20:
            dim_labels = [
                'Left X', 'Left Y', 'Left Z',  # 0-2: 左臂位置
                'Left R1', 'Left R2', 'Left R3', 'Left R4', 'Left R5', 'Left R6',  # 3-8: 左臂旋转6D
                'Left Gripper',  # 9: 左夹爪
                'Right X', 'Right Y', 'Right Z',  # 10-12: 右臂位置
                'Right R1', 'Right R2', 'Right R3', 'Right R4', 'Right R5', 'Right R6',  # 13-18: 右臂旋转6D
                'Right Gripper'  # 19: 右夹爪
            ]
        else:
            dim_labels = [f'Dim {i}' for i in range(num_dims)]
        
        # 创建一个大图，显示所有维度
        rows = (num_dims + 3) // 4  # 每行4个子图
        cols = min(4, num_dims)
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i in range(num_dims):
            ax = axes[i]
            ax.plot(predicted_array[:, i], label='Predicted', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.plot(gt_array[:, i], label='GT', linewidth=1.5, alpha=0.8)
            ax.set_title(f'{dim_labels[i]}\nMAE: {mae_per_dim[i]:.4f}', fontsize=10)
            ax.set_xlabel('Time Steps', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(num_dims, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'All {num_dims} Dimensions: Predicted vs Ground Truth', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('prediction_vs_gt_all_dims.png', dpi=150, bbox_inches='tight')
        print(f"✓ 所有维度对比图已保存: prediction_vs_gt_all_dims.png")
        
        # 另外创建一个汇总的MAE条形图
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(num_dims), mae_per_dim, color='steelblue', alpha=0.7)
        plt.xlabel('Dimension', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.title(f'MAE per Dimension (Average: {np.mean(mae_per_dim):.4f})', fontsize=14, fontweight='bold')
        plt.xticks(range(num_dims), dim_labels, rotation=45, ha='right', fontsize=9)
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(np.mean(mae_per_dim), color='red', linestyle='--', linewidth=2, label=f'Avg MAE: {np.mean(mae_per_dim):.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('mae_per_dimension.png', dpi=150, bbox_inches='tight')
        print(f"✓ MAE统计图已保存: mae_per_dimension.png")
        
        # 打印详细统计信息
        print("\n" + "="*60)
        print("推理结果统计")
        print("="*60)
        print(f"总时间步数: {len(predicted_list)}")
        print(f"动作维度: {num_dims}")
        print(f"平均MAE: {np.mean(mae_per_dim):.6f}")
        print(f"最大MAE: {np.max(mae_per_dim):.6f} (维度 {np.argmax(mae_per_dim)}: {dim_labels[np.argmax(mae_per_dim)]})")
        print(f"最小MAE: {np.min(mae_per_dim):.6f} (维度 {np.argmin(mae_per_dim)}: {dim_labels[np.argmin(mae_per_dim)]})")
        print("="*60)
        
        print("\n推理完成！结果已保存。")

    with open('record_list.pkl', 'wb') as f:  # Open file in binary write mode
        pickle.dump(record_list, f)
    