from copy import deepcopy
import pdb
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
from torchvision.utils import save_image

from mimo_modules.MIMOUNet import EBlock, DBlock

# 导入J-MKPD中的模糊核预测网络相关类
class Down(nn.Module):
    """double conv and then downscaling with maxpool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,  kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
           # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            #nn.BatchNorm2d(out_channels),
        )

        self.down_sampling = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.double_conv(x)
        down_sampled = self.down_sampling(feat)
        return feat, down_sampled


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, feat_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels,  in_channels, kernel_size=2, stride=2)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.feat = nn.Sequential(
            nn.Conv2d(feat_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        x1 = self.double_conv(x1)

        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        feat = self.feat(x)
        return feat

class PooledSkip(nn.Module):
    def __init__(self, output_spatial_size):
        super().__init__()

        self.output_spatial_size = output_spatial_size

    def forward(self, x):
        global_avg_pooling = x.mean((2,3), keepdim=True)
        return global_avg_pooling.repeat(1,1,self.output_spatial_size,self.output_spatial_size)


class KernelPredictionNetwork(nn.Module):
    def __init__(self, K=9, blur_kernel_size=33, bilinear=False, no_softmax=False):
        super(KernelPredictionNetwork, self).__init__()

        self.no_softmax = no_softmax
        if no_softmax:
            print('Softmax is not being used')

        self.inc_rgb = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.inc_gray = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.blur_kernel_size = blur_kernel_size
        self.K = K

        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.feat = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.feat5_gap = PooledSkip(2)
        self.feat4_gap = PooledSkip(4)  
        self.feat3_gap = PooledSkip(8)  
        self.feat2_gap = PooledSkip(16)  
        self.feat1_gap = PooledSkip(32) 

        self.kernel_up1 = Up(1024, 1024, 512, bilinear)
        self.kernel_up2 = Up(512, 512, 256, bilinear)
        self.kernel_up3 = Up(256, 256, 256, bilinear)
        self.kernel_up4 = Up(256, 128, 128, bilinear)
        self.kernel_up5 = Up(128, 64, 64, bilinear)
        if self.blur_kernel_size > 33:
            self.kernel_up6 = Up(64, 0, 64, bilinear)

        self.kernels_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, K, kernel_size=3, padding=1)
        )
        self.kernel_softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # Encoder
        if x.shape[1] == 3:
            x1 = self.inc_rgb(x)
        else:
            x1 = self.inc_gray(x)
        x1_feat, x2 = self.down1(x1)
        x2_feat, x3 = self.down2(x2)
        x3_feat, x4 = self.down3(x3)
        x4_feat, x5 = self.down4(x4)
        x5_feat, x6 = self.down5(x5)
        x6_feat = self.feat(x6)

        feat6_gap = x6_feat.mean((2,3), keepdim=True)
        feat5_gap = self.feat5_gap(x5_feat)
        feat4_gap = self.feat4_gap(x4_feat)
        feat3_gap = self.feat3_gap(x3_feat)
        feat2_gap = self.feat2_gap(x2_feat)
        feat1_gap = self.feat1_gap(x1_feat)
        
        k1 = self.kernel_up1(feat6_gap, feat5_gap)
        k2 = self.kernel_up2(k1, feat4_gap)
        k3 = self.kernel_up3(k2, feat3_gap)
        k4 = self.kernel_up4(k3, feat2_gap)
        k5 = self.kernel_up5(k4, feat1_gap)

        if self.blur_kernel_size == 65:
            k6 = self.kernel_up6(k5)
            k = self.kernels_end(k6)
        else:
            k = self.kernels_end(k5)
        N, F, H, W = k.shape  # H and W should be one
        k = k.view(N, self.K, self.blur_kernel_size * self.blur_kernel_size)

        if self.no_softmax:
            k = F.leaky_relu(k)
        else:
            k = self.kernel_softmax(k)

        k = k.view(N, self.K, self.blur_kernel_size, self.blur_kernel_size)

        return k


# 导入LMD-ViT的原始代码
class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
        )

    def forward(self, x):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, :C // 2]
        global_x = (x[:, :, C // 2:]).mean(dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C // 2)], dim=-1)
        return self.out_conv(x)


class FastLeFF(nn.Module):

    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()

        from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(DepthwiseConv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                                    act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x h x w x c
        x = self.linear1(x).permute(0, 3, 1, 2)

        # spatial restore
        x = self.dwconv(x).permute(0, 2, 3, 1)

        # flaten
        x = self.linear2(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        return flops


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# 此处省略LMD-ViT-main中的其他模块代码...
# 需要将model_hw.py中的其他代码复制到这里

# 修改LMD类来集成模糊核预测网络
class LMDWithKernel(nn.Module):
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 dowsample=None, upsample=None, shift_flag=True, modulator=False,
                 cross_modulator=False, prune_loc=[0, 0, 1, 1, 1, 1, 1, 0, 0],
                 kernel_pred_K=9, kernel_size=33, **kwargs):
        super().__init__()
        
        # 初始化LMD中的部分
        from model_hw import Downsample, Upsample, LMD
        
        if dowsample is None:
            dowsample = Downsample
        if upsample is None:
            upsample = Upsample
            
        # 创建基础LMD网络
        self.base_lmd = LMD(
            img_size=img_size, in_chans=in_chans, dd_in=dd_in,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, patch_norm=patch_norm,
            use_checkpoint=use_checkpoint, token_projection=token_projection, token_mlp=token_mlp,
            dowsample=dowsample, upsample=upsample, shift_flag=shift_flag, modulator=modulator,
            cross_modulator=cross_modulator, prune_loc=prune_loc, **kwargs
        )
        
        # 初始化模糊核预测网络
        self.kernel_prediction = KernelPredictionNetwork(
            K=kernel_pred_K, 
            blur_kernel_size=kernel_size, 
            bilinear=True
        )
        
        # 添加融合层，用于将预测的核与特征融合
        self.kernel_fusion = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        
    def forward(self, x, mask=None):
        # 预测模糊核
        kernels = self.kernel_prediction(x)
        
        # 使用LMD网络处理图像
        deblurred, gate_x, pred_score_lists, decision_lists = self.base_lmd(x, mask)
        
        # 返回结果中包含预测的模糊核
        return deblurred, gate_x, kernels, pred_score_lists, decision_lists
    
    def forward_with_kernels(self, x, kernels=None, mask=None):
        """使用预先计算好的模糊核进行图像去模糊"""
        if kernels is None:
            # 如果没有提供模糊核，则预测模糊核
            kernels = self.kernel_prediction(x)
            
        # 使用LMD网络处理图像
        deblurred, gate_x, pred_score_lists, decision_lists = self.base_lmd(x, mask)
        
        # 返回结果
        return deblurred, gate_x, kernels, pred_score_lists, decision_lists 