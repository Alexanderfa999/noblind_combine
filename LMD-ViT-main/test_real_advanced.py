import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from torchvision.utils import make_grid, save_image
from kernel_model import KernelPredictionNetwork
from model_hw import LMD, Downsample, Upsample

class AdvancedLMDWithKernel(torch.nn.Module):
    """集成模糊核预测的完整LMD-ViT模型 - 禁用修剪功能"""
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=torch.nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 kernel_pred_K=9, kernel_size=33, **kwargs):
        super().__init__()
        
        # 初始化完整的LMD模型，禁用修剪功能
        self.lmd = LMD(img_size=img_size, in_chans=in_chans, dd_in=dd_in,
                      embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                      win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                      norm_layer=norm_layer, patch_norm=patch_norm,
                      use_checkpoint=use_checkpoint, token_projection=token_projection, token_mlp=token_mlp,
                      dowsample=Downsample, upsample=Upsample,
                      prune_loc=[0, 0, 0, 0, 0, 0, 0, 0, 0])  # 禁用所有修剪
        
        # 初始化模糊核预测网络
        self.kernel_prediction = KernelPredictionNetwork(
            K=kernel_pred_K, 
            blur_kernel_size=kernel_size, 
            bilinear=True
        )
        
    def forward(self, x, mask=None):
        # 使用LMD-ViT进行去模糊
        deblurred, gate_x, pred_score_lists, decision_lists = self.lmd(x, mask)
        
        # 预测模糊核
        kernels = self.kernel_prediction(x)
        
        # 返回结果，包含预测的模糊核
        return deblurred, gate_x, kernels, pred_score_lists, decision_lists
    
    def load_lmd_weights(self, checkpoint):
        """加载LMD模型的预训练权重"""
        # 处理权重名称不匹配的问题
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
            
        # 加载权重
        missing, unexpected = self.lmd.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            print(f"缺少权重: {len(missing)} 项")
        if len(unexpected) > 0:
            print(f"多余权重: {len(unexpected)} 项")
            
        return self

def save_kernels_grid(kernels, output_path):
    """将预测的模糊核保存为图像网格"""
    n_kernels = kernels.shape[1]
    kernels_ext = kernels[0, :, :, :]  # 选择第一个样本的所有核
    kernels_ext = kernels_ext[:, None, :, :]  # 添加通道维度
    
    # 创建核的网格图
    kernels_grid = make_grid(kernels_ext, nrow=3, normalize=True, scale_each=True, pad_value=1)
    
    # 保存为图像
    save_image(kernels_grid, output_path)
    print(f'核已保存到 {output_path}')
    
    return kernels_grid

def load_image(image_path):
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
    
    # 调整大小，如果需要
    h, w, _ = img.shape
    max_size = 1024
    if h > max_size or w > max_size:
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
    parser.add_argument('--output', type=str, default='results_advanced', help='输出目录')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/GoPro.pth', help='LMD-ViT 权重路径')
    parser.add_argument('--kernel_K', type=int, default=9, help='预测模糊核的数量')
    parser.add_argument('--kernel_size', type=int, default=33, help='模糊核的大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 读取图像
    print(f"加载图像: {args.input}")
    img_raw, img_tensor = load_image(args.input)
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    print("初始化模型...")
    model = AdvancedLMDWithKernel(
        kernel_pred_K=args.kernel_K,
        kernel_size=args.kernel_size,
        img_size=max(new_h, new_w)
    )
    
    # 加载预训练权重
    if os.path.exists(args.checkpoint):
        print(f"加载权重: {args.checkpoint}")
        model = model.load_lmd_weights(torch.load(args.checkpoint, map_location='cpu'))
    else:
        print(f"找不到权重: {args.checkpoint}")
        print("将使用随机初始化模型...")
    
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
    deblurred_np = deblurred[0].permute(1, 2, 0).cpu().numpy()
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
    save_kernels_grid(kernels.cpu(), kernels_path)
    
    print(f"处理完成，结果保存在 {args.output} 目录中")
    print(f"- 输入图像: {input_path}")
    print(f"- 去模糊图像: {deblurred_path}")
    print(f"- 预测的模糊核: {kernels_path}")

if __name__ == "__main__":
    main() 