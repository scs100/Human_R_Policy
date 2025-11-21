import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import v2
import torch

from hdt.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

import hdt.constants
import hdt.constant_20d
import hdt  # 确保 hdt 模块在全局作用域可用
# 导入16维常量（与constant_20d同样的方式）
try:
    import hdt.constant_16d_joint
except ImportError as e:
    # 如果导入失败，记录错误但不在模块级别抛出
    # 将在运行时根据action_dim决定是否需要
    _constant_16d_joint_error = e
    hdt.constant_16d_joint = None

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        
        # 根据action_dim选择使用哪个常量模块
        self.action_dim = args_override['action_dim']
        if self.action_dim == 20:
            self.constants = hdt.constant_20d
        elif self.action_dim == 16:
            # 使用16维关节角常量
            # 使用全局的 hdt 模块引用，避免局部变量冲突
            import hdt as _hdt_module
            if _hdt_module.constant_16d_joint is None:
                # 如果导入失败，尝试重新导入
                import sys
                import os
                # 获取当前文件所在目录（hdt目录）
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # 获取项目根目录（hdt的父目录）
                parent_dir = os.path.dirname(current_dir)
                # 确保项目根目录在Python路径中
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                # 重新导入
                try:
                    import hdt.constant_16d_joint
                    _hdt_module.constant_16d_joint = hdt.constant_16d_joint
                except ImportError:
                    # 如果还是失败，尝试直接导入
                    import constant_16d_joint
                    _hdt_module.constant_16d_joint = constant_16d_joint
            self.constants = _hdt_module.constant_16d_joint
        else:
            self.constants = hdt.constants
        # patch_h = 24
        # patch_w = 42
        patch_h = 16
        patch_w = 22
        # FIXME(roger): RSS temp. EEF loss
        self.USE_EEF_LOSS = True
        self.transform = v2.Compose([
            v2.Resize((patch_h * 14, patch_w * 14)),
            # v2.CenterCrop((patch_h * 14, patch_w * 14)),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, image, qpos, actions=None, is_pad=None, conditioning_dict=None):
        env_state = None
            
        image = self.transform(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad, conditioning_dict)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            all_l1 = (all_l1 * ~is_pad.unsqueeze(-1))
            loss_dict['l1'] = all_l1.mean()
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            if self.USE_EEF_LOSS:
                # 检查是否有EEF索引（对于关节角数据可能没有）
                if len(self.constants.OUTPUT_LEFT_EEF) > 0 and len(self.constants.OUTPUT_RIGHT_EEF) > 0:
                    loss_dict['eef_loss'] = all_l1[:, :, self.constants.OUTPUT_LEFT_EEF].mean() + all_l1[:, :, self.constants.OUTPUT_RIGHT_EEF].mean()
                    loss_dict['loss'] = loss_dict['loss'] + loss_dict['eef_loss'] * 2
                # 对于关节角数据，可以使用关节角损失代替EEF损失
                elif hasattr(self.constants, 'OUTPUT_LEFT_ARM_JOINTS') and hasattr(self.constants, 'OUTPUT_RIGHT_ARM_JOINTS'):
                    if len(self.constants.OUTPUT_LEFT_ARM_JOINTS) > 0 and len(self.constants.OUTPUT_RIGHT_ARM_JOINTS) > 0:
                        loss_dict['arm_joint_loss'] = all_l1[:, :, self.constants.OUTPUT_LEFT_ARM_JOINTS].mean() + all_l1[:, :, self.constants.OUTPUT_RIGHT_ARM_JOINTS].mean()
                        loss_dict['loss'] = loss_dict['loss'] + loss_dict['arm_joint_loss'] * 2
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state, conditioning_dict=conditioning_dict) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, image, qpos, actions=None, is_pad=None, conditioning_dict=None):
        env_state = None # TODO
        normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
