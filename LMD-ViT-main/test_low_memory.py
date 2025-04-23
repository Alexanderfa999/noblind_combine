import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from torchvision.utils import make_grid, save_image
from kernel_model_low_mem import LowMemKernelPredictionNetwork
from model_hw import LMD, Downsample, Upsample

class LowMemLMDWithKernel(torch.nn.Module):
    """低内存版LMD-ViT模型与核预测网络集成版本"""
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 embed_dim=16, # 降低基础通道数
                 depths=[1, 1, 1, 1, 1, 1, 1, 1, 1], # 降低每层深度
                 num_heads=[1, 1, 2, 4, 8, 4, 2, 1, 1], # 降低注意力头数
                 win_size=8, mlp_ratio=2., # 降低MLP比率
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=torch.nn.LayerNorm, patch_norm=True,
                 use_checkpoint=True, # 启用梯度检查点
                 token_projection='linear', token_mlp='leff',
                 kernel_pred_K=5, # 降低核预测数量 
                 kernel_size=17, # 降低核大小
                 **kwargs):
        super().__init__()
        
        # 初始化LMD模型，使用简化版配置
        self.lmd = LMD(img_size=img_size, in_chans=in_chans, dd_in=dd_in,
                      embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                      win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                      norm_layer=norm_layer, patch_norm=patch_norm,
                      use_checkpoint=use_checkpoint, token_projection=token_projection, token_mlp=token_mlp,
                      dowsample=Downsample, upsample=Upsample,
                      prune_loc=[0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # 初始化简化版核预测网络
        self.kernel_prediction = LowMemKernelPredictionNetwork(
            K=kernel_pred_K, 
            kernel_size=kernel_size, 
            bilinear=True
        )
        
    def forward(self, x, mask=None):
        # 去模糊处理
        deblurred, gate_x, pred_score_lists, decision_lists = self.lmd(x, mask)
        
        # 分别进行核预测，而不是一次处理整个批次
        batch_size = x.shape[0]
        kernels_list = []
        
        # 逐个处理每个样本以节省内存
        for i in range(batch_size):
            kernel = self.kernel_prediction(x[i:i+1])
            kernels_list.append(kernel)
        
        # 合并结果
        if batch_size > 1:
            kernels = torch.cat(kernels_list, dim=0)
        else:
            kernels = kernels_list[0]
        
        return deblurred, gate_x, kernels, pred_score_lists, decision_lists
    
    def load_lmd_weights(self, checkpoint):
        """尝试加载权重到简化模型"""
        try:
            # 处理权重名称不匹配的问题
            new_state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
                
            # 尝试加载权重，非严格匹配
            missing, unexpected = self.lmd.load_state_dict(new_state_dict, strict=False)
            print(f"缺少权重: {len(missing)} 项")
            print(f"多余权重: {len(unexpected)} 项")
            
        except Exception as e:
            print(f"加载权重时出错: {e}")
            print("将使用随机初始化权重")
            
        return self

def save_kernels_grid(kernels, output_path):
    """将预测的模糊核保存为图像网格"""
    # 检查核张量的形状并打印
    print(f"核张量形状: {kernels.shape}")
    
    # 处理不同形状的核张量
    if len(kernels.shape) == 6:  # [B, K, kernel_size, kernel_size, H, W]
        # 核预测网络输出的核尺寸
        n_kernels = kernels.shape[1]
        kernel_size = kernels.shape[2]
        
        # 取第一个样本的所有核和第一个空间位置的核
        # 只使用中心点的核，因为显示所有位置的核会太多
        # 形状变为 [K, kernel_size, kernel_size]
        center_h, center_w = kernels.shape[4] // 2, kernels.shape[5] // 2
        kernels_display = kernels[0, :, :, :, center_h, center_w]
        
        # 添加通道维度 [K, 1, kernel_size, kernel_size]
        kernels_display = kernels_display[:, None, :, :]
    else:  # [B, K, H, W] 或其他形状
        # 可能是其他版本的模型输出，尝试兼容
        n_kernels = kernels.shape[1]
        kernels_display = kernels[0, :, :, :]  # [K, H, W]
        
        # 添加通道维度
        kernels_display = kernels_display[:, None, :, :] 
    
    # 创建核的网格图
    kernels_grid = make_grid(kernels_display, nrow=min(3, n_kernels), normalize=True, scale_each=True, pad_value=1)
    
    # 保存为图像
    save_image(kernels_grid, output_path)
    print(f'核已保存到 {output_path}')
    
    return kernels_grid

def load_image(image_path, max_size=256):  # 降低图像尺寸
    """加载并处理图像"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图像文件: {image_path}")
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 强制调整大小到较小尺寸
    h, w, _ = img.shape
    scale = max_size / max(h, w)
    h, w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (w, h))
    
    # 转换为张量 [B, C, H, W]
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
    
    return img, img_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../J-MKPD-main/testing_imgs/car2.jpg', help='输入图像路径')
    parser.add_argument('--output', type=str, default='results_low_mem', help='输出目录')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/GoPro.pth', help='LMD-ViT 权重路径')
    parser.add_argument('--max_size', type=int, default=256, help='图像最大尺寸')
    parser.add_argument('--kernel_K', type=int, default=5, help='预测模糊核的数量')
    parser.add_argument('--kernel_size', type=int, default=17, help='模糊核的大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 读取图像
    print(f"加载图像: {args.input}")
    img_raw, img_tensor = load_image(args.input, max_size=args.max_size)
    
    # 确保图像尺寸是8的倍数
    _, _, h, w = img_tensor.shape
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    
    print(f"原始图像尺寸: {h}x{w}")
    print(f"调整后尺寸: {new_h}x{new_w}")
    
    if h != new_h or w != new_w:
        # 使用插值调整图像大小
        img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        print(f"已调整图像尺寸为: {img_tensor.shape[2]}x{img_tensor.shape[3]}")
    
    # 设置设备
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    print("初始化低内存模型...")
    model = LowMemLMDWithKernel(
        kernel_pred_K=args.kernel_K,
        kernel_size=args.kernel_size,
        img_size=max(new_h, new_w)
    )
    
    # 加载预训练权重
    if os.path.exists(args.checkpoint):
        print(f"尝试加载权重: {args.checkpoint}")
        try:
            # 使用CPU加载权重
            model = model.load_lmd_weights(torch.load(args.checkpoint, map_location='cpu'))
            print("权重已加载")
        except Exception as e:
            print(f"加载权重失败: {e}")
    else:
        print(f"找不到权重文件: {args.checkpoint}")
        print("将使用随机初始化模型...")
    
    # 转移模型到设备上
    model.to(device)
    model.eval()
    
    # 记录原始图像大小，用于恢复
    orig_h, orig_w = h, w
    
    # 推理
    print("执行模型推理...")
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        
        # 执行推理
        deblurred, gate_x, kernels, _, _ = model(img_tensor)
        
        # 如果需要，恢复到原始大小
        if h != new_h or w != new_w:
            deblurred = F.interpolate(deblurred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    
    # 保存去模糊后的图像
    deblurred_np = deblurred[0].permute(1, 2, 0).numpy()
    deblurred_np = np.clip(deblurred_np * 255.0, 0, 255).astype(np.uint8)
    deblurred_np = cv2.cvtColor(deblurred_np, cv2.COLOR_RGB2BGR)
    
    # 保存输入的模糊图像
    input_np = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
    
    # 获取输入文件名（不含路径和扩展名）
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # 保存结果
    input_path = os.path.join(args.output, f"{input_name}_input.jpg")
    deblurred_path = os.path.join(args.output, f"{input_name}_deblurred.jpg")
    kernels_path = os.path.join(args.output, f"{input_name}_kernels.png")
    
    cv2.imwrite(input_path, input_np)
    cv2.imwrite(deblurred_path, deblurred_np)
    save_kernels_grid(kernels, kernels_path)
    
    print(f"处理完成，结果保存在 {args.output} 目录中")
    print(f"- 输入图像: {input_path}")
    print(f"- 去模糊图像: {deblurred_path}")
    print(f"- 预测的模糊核: {kernels_path}")

if __name__ == "__main__":
    # 设置PyTorch性能相关参数
    torch.backends.cudnn.benchmark = False
    
    main() 