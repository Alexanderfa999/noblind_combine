import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(卷积 => 批归一化 => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样和双卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样和连接跳跃连接"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 如果使用双线性插值，不需要转置卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理输入尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 连接
        x = torch.cat([x2, x1], dim=1)
        
        # 双卷积
        x = self.conv(x)
        
        # 释放不再需要的变量
        del x1, x2
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        return x

class LowMemKernelPredictionNetwork(nn.Module):
    """
    低内存版本的核预测网络
    采用U-Net架构进行模糊核预测，针对内存优化进行修改
    K：预测的核数量
    kernel_size：核大小
    """
    def __init__(self, K=41, kernel_size=21, bilinear=True):
        super(LowMemKernelPredictionNetwork, self).__init__()
        self.K = K  # 预测核数量
        self.kernel_size = kernel_size  # 核大小
        self.bilinear = bilinear  # 是否使用双线性插值上采样
        
        # 减少通道数量以节省内存
        factor = 2 if bilinear else 1
        
        # 编码器
        self.inc = DoubleConv(3, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512 // factor)
        
        # 解码器
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        
        # 输出核预测头
        # K*kernel_size*kernel_size的输出用于预测K个模糊核
        self.outc = nn.Conv2d(32, K * kernel_size * kernel_size, kernel_size=1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播，预测每个像素的模糊核
        输入：x，形状为[B, C, H, W]
        输出：kernels，形状为[B, K, kernel_size, kernel_size, H_out, W_out]
        """
        # 保存原始大小
        B, C, H, W = x.shape
        
        # 确保输入尺寸是8的倍数
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # 使用半精度计算减少内存占用
        with torch.cuda.amp.autocast():
            # 编码路径
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            # 解码路径
            x = self.up1(x5, x4)
            del x4, x5  # 释放内存
            x = self.up2(x, x3)
            del x3  # 释放内存
            x = self.up3(x, x2)
            del x2  # 释放内存
            x = self.up4(x, x1)
            del x1  # 释放内存
            
            # 输出层
            x = self.outc(x)
            
            # 将输出重塑为核形状
            _, _, H_out, W_out = x.shape
            
            # 将输出重新整形为[B, K*kernel_size*kernel_size, H_out, W_out]
            x = x.view(B, self.K, self.kernel_size * self.kernel_size, H_out, W_out)
            
            # 对每个核应用softmax，确保核的和为1
            x = F.softmax(x, dim=2)
            
            # 重塑为[B, K, kernel_size, kernel_size, H_out, W_out]
            x = x.view(B, self.K, self.kernel_size, self.kernel_size, H_out, W_out)
            
            # 清理内存
            torch.cuda.empty_cache()
            
            return x 