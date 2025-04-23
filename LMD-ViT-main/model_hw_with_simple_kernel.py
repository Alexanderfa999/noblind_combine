import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_hw import LMD, Downsample, Upsample

# 模糊核预测网络
class KernelPredictionNetwork(nn.Module):
    """模糊核预测网络"""
    def __init__(self, K=9, kernel_size=33):
        super(KernelPredictionNetwork, self).__init__()
        
        self.kernel_size = kernel_size
        self.K = K
        
        # 简化版本的核预测网络
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        
        self.down1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.LeakyReLU(inplace=True)
        
        self.down2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.LeakyReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6 = nn.LeakyReLU(inplace=True)
        
        # 全局平均池化后的输出层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 512)
        self.fc_relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(512, K * kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 特征提取
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        
        x = self.down1(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        
        x = self.down2(x)
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        
        # 全局特征
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_relu(self.fc1(x))
        x = self.fc2(x)
        
        # 重塑为多个核
        k = x.view(x.size(0), self.K, self.kernel_size * self.kernel_size)
        k = self.softmax(k)
        k = k.view(x.size(0), self.K, self.kernel_size, self.kernel_size)
        
        return k

class SimpleLMDWithKernel(nn.Module):
    """简化版的集成模糊核预测的LMD-ViT模型"""
    def __init__(self, img_size=256, in_chans=3, 
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 shift_flag=True, modulator=False, cross_modulator=False,
                 kernel_pred_K=9, kernel_size=33, **kwargs):
        super().__init__()
        
        # 初始化LMD模型，禁用修剪功能
        self.lmd = LMD(img_size=img_size, in_chans=in_chans, dd_in=in_chans,
                    embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                    win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer, patch_norm=patch_norm,
                    use_checkpoint=use_checkpoint, token_projection=token_projection, token_mlp=token_mlp,
                    downsample=Downsample, upsample=Upsample, shift_flag=shift_flag, modulator=modulator,
                    cross_modulator=cross_modulator, prune_loc=[0, 0, 0, 0, 0, 0, 0, 0, 0])  # 禁用所有修剪
        
        # 初始化简化版的模糊核预测网络
        self.kernel_prediction = KernelPredictionNetwork(
            K=kernel_pred_K, 
            kernel_size=kernel_size
        )
        
    def forward(self, x, mask=None):
        # 执行去模糊操作，忽略修剪功能
        deblurred, gate_x, pred_score, decision = self.lmd(x, mask)
        
        # 预测模糊核
        kernels = self.kernel_prediction(x)
        
        # 返回结果
        return deblurred, gate_x, kernels, pred_score, decision
    
    def load_lmd_weights(self, checkpoint):
        """加载LMD模型的预训练权重"""
        # 处理权重名称不匹配的问题
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
            
        # 只加载匹配的权重
        missing, unexpected = self.lmd.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            print(f"缺少权重: {len(missing)} 项")
        if len(unexpected) > 0:
            print(f"多余权重: {len(unexpected)} 项")
        
        return self 