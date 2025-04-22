import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

# 模糊核预测网络相关类
class Down(nn.Module):
    """下采样和双卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.down_sampling = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.double_conv(x)
        down_sampled = self.down_sampling(feat)
        return feat, down_sampled


class Up(nn.Module):
    """上采样和双卷积"""
    def __init__(self, in_channels, feat_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
    """池化并重复特征"""
    def __init__(self, output_spatial_size):
        super().__init__()
        self.output_spatial_size = output_spatial_size

    def forward(self, x):
        global_avg_pooling = x.mean((2,3), keepdim=True)
        return global_avg_pooling.repeat(1,1,self.output_spatial_size,self.output_spatial_size)


class KernelPredictionNetwork(nn.Module):
    """模糊核预测网络"""
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


class LMDWithKernel(nn.Module):
    """集成模糊核预测的去模糊模型"""
    def __init__(self, img_size=256, in_chans=3, 
                 kernel_pred_K=9, kernel_size=33, **kwargs):
        super().__init__()
        
        # 初始化模糊核预测网络
        self.kernel_prediction = KernelPredictionNetwork(
            K=kernel_pred_K, 
            blur_kernel_size=kernel_size, 
            bilinear=True
        )
        
        # 简单的去模糊网络
        self.deblur_network = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, in_chans, kernel_size=3, padding=1)
        )
        
    def forward(self, x, mask=None):
        # 预测模糊核
        kernels = self.kernel_prediction(x)
        
        # 去模糊处理
        deblurred = self.deblur_network(x)
        
        # 模拟其他输出
        gate_x = torch.ones_like(x[:, :1])
        pred_score_lists = []
        decision_lists = []
        
        return deblurred, gate_x, kernels, pred_score_lists, decision_lists 