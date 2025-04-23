import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from kernel_model_low_mem import LowMemKernelPredictionNetwork


def window_partition(x, window_size):
    """将特征图划分为非重叠窗口
    
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
        
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    # 确保特征图尺寸是窗口大小的整数倍
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, H, W, _ = x.shape
    
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """从非重叠窗口重构特征图
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 高度
        W (int): 宽度
        
    Returns:
        x: (B, H, W, C)
    """
    # 计算批次大小
    B = int(windows.shape[0] / (math.ceil(H / window_size) * math.ceil(W / window_size)))
    
    # 重构特征图
    x = windows.view(B, math.ceil(H / window_size), math.ceil(W / window_size), 
                    window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, math.ceil(H / window_size) * window_size, 
                                                    math.ceil(W / window_size) * window_size, -1)
    
    # 如果有填充，去除填充部分
    if x.shape[1] > H or x.shape[2] > W:
        x = x[:, :H, :W, :].contiguous()
        
    return x


class Mlp(nn.Module):
    """多层感知机，简化版"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 2  # 减少中间层维度
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """简化的窗口多头自注意力模块"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 定义相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # 获取窗口内每个token的相对位置索引
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # 简化的QKV投影，减少参数
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 提前计算相对位置偏置
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """前向传播
        Args:
            x: 输入特征图, [num_windows*B, N, C]
            mask: (0/-inf) 掩码，[num_windows, Wh*Ww, Wh*Ww] 或 None
        """
        B_, N, C = x.shape
        
        # 使用梯度检查点以减少内存使用
        with torch.cuda.amp.autocast(enabled=True):
            qkv = self.qkv(x)
            qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # 计算注意力图
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            
            # 添加相对位置编码
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
            
            # 应用注意力掩码（如果有）
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)
            
            # 归一化并应用dropout
            attn = self.attn_drop(attn)
            
            # 计算加权和
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        
        # 清理缓存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        return x


class SwinTransformerBlock(nn.Module):
    """简化的Swin Transformer块"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        
        # 确保移位大小小于窗口大小
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
            
        self.norm1 = norm_layer(dim)
        
        # 窗口注意力模块
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # 减小MLP比例以节省内存
        mlp_hidden_dim = int(dim * mlp_ratio / 2)  # 减少隐藏层维度
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 如果是移位窗口，计算注意力掩码
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征尺寸与分辨率不匹配"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分区窗口
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 应用窗口注意力
        if self.use_checkpoint and self.training:
            attn_windows = checkpoint.checkpoint(self.attn, x_windows, self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        if self.use_checkpoint and self.training:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(checkpoint.checkpoint(self.mlp, self.norm2(x)))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """简化的补丁合并层"""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征尺寸与输入分辨率不匹配"

        x = x.view(B, H, W, C)

        # 填充如果需要
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        # 分组计算
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """Swin Transformer的一个阶段"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            for i in range(depth)])

        # 下采样层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # 逐块处理以减少内存
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
                
            # 清理 CUDA 缓存
            if i < len(self.blocks) - 1 and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """图像到补丁嵌入"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 处理非标准尺寸
        _, _, H, W = x.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)

        # 下采样嵌入
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class LowMemLMDViT(nn.Module):
    """低内存版本的LMD-ViT模型"""
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=16,
                 depths=[1, 1, 1, 1, 1, 1, 1, 1, 1], num_heads=[1, 1, 2, 4, 8, 4, 2, 1, 1],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=True, **kwargs):
        super().__init__()

        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # 拆分图像为非重叠补丁
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置嵌入
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer < self.num_layers // 2:
                # 编码阶段
                layer = BasicLayer(
                    dim=embed_dim * 2 ** min(i_layer, 3),  # 限制通道增长
                    input_resolution=(patches_resolution[0] // (2 ** min(i_layer, 3)),
                                    patches_resolution[1] // (2 ** min(i_layer, 3))),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers // 2 - 1) else None,
                    use_checkpoint=use_checkpoint)
            else:
                # 解码阶段
                layer = BasicLayer(
                    dim=embed_dim * 2 ** max(0, 3 - (i_layer - self.num_layers // 2)),
                    input_resolution=(patches_resolution[0] // (2 ** max(0, 3 - (i_layer - self.num_layers // 2))),
                                    patches_resolution[1] // (2 ** max(0, 3 - (i_layer - self.num_layers // 2)))),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # 输出头
        self.norm = norm_layer(self.num_features)
        self.head = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

        # 核预测网络
        self.kernel_predictor = LowMemKernelPredictionNetwork(K=41, kernel_size=21)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # 图像嵌入和位置编码
        x = self.patch_embed(x)
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 编码器部分
        for i in range(self.num_layers // 2):
            x = self.layers[i](x)
            # 清理 CUDA 缓存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        # 记录中间特征
        bottleneck = x

        # 解码器部分
        for i in range(self.num_layers // 2, self.num_layers):
            x = self.layers[i](x)
            # 清理 CUDA 缓存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        # 标准化输出特征
        x = self.norm(x)
        
        return x, bottleneck

    def forward(self, x, return_kernels=False):
        # 使用半精度计算
        with torch.cuda.amp.autocast():
            # 获取特征
            B, C, H, W = x.shape
            
            # 确保尺寸是8的倍数
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))
                
            features, bottleneck = self.forward_features(x)
            
            # 将特征重塑回空间形式
            patches_resolution = self.patch_embed.patches_resolution
            features = features.view(B, patches_resolution[0], patches_resolution[1], -1).permute(0, 3, 1, 2)
            
            # 重建输出
            x_recon = self.head(features)
            
            # 插值到原始尺寸
            if pad_h > 0 or pad_w > 0:
                x_recon = x_recon[:, :, :H, :W]
            
            # 重塑并预测核
            feat = x_recon.transpose(1, 2).reshape(B, -1, H, W)
            
            # 预测模糊核
            kernels = self.kernel_predictor(x)
            
            # 执行去模糊（可选）
            if return_kernels:
                return kernels
                
            # 否则，我们返回输入图像
            # 在真实应用中，这里应该实现去模糊操作
            return x_recon, kernels 