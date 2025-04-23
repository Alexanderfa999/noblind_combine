import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from torchvision.utils import make_grid, save_image
from model_hw_with_simple_kernel import KernelPredictionNetwork

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

def load_image(file_path):
    """加载图像文件"""
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像文件: {file_path}")
    
    # 转换为RGB（PyTorch期望的格式）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整大小如果图像太大
    h, w, _ = img.shape
    max_size = 1024
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        h, w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (w, h))
    
    # 转换为张量 [1, 3, H, W]
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    return img, img_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../J-MKPD-main/testing_imgs/car2.jpg', help='输入图像文件路径')
    parser.add_argument('--output', type=str, default='results_real_image', help='输出目录')
    parser.add_argument('--kernel_K', type=int, default=9, help='预测模糊核的数量')
    parser.add_argument('--kernel_size', type=int, default=33, help='模糊核的大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"找不到输入文件: {args.input}")
        return
    
    # 获取输入文件名（不含路径和扩展名）
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # 加载图像
    print(f"加载图像文件: {args.input}")
    img_raw, img_tensor = load_image(args.input)
    print(f"原始图像尺寸: {img_tensor.shape[2]}x{img_tensor.shape[3]}")
    
    # 调整图像大小为固定尺寸
    img_tensor_resized = F.interpolate(img_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    print(f"已调整图像尺寸为: {img_tensor_resized.shape[2]}x{img_tensor_resized.shape[3]}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化仅核预测模型
    print("初始化核预测网络...")
    kernel_model = KernelPredictionNetwork(
        K=args.kernel_K, 
        kernel_size=args.kernel_size
    )
    kernel_model.to(device)
    kernel_model.eval()
    
    # 推理
    print("执行核预测...")
    with torch.no_grad():
        img_tensor_resized = img_tensor_resized.to(device)
        kernels = kernel_model(img_tensor_resized)
    
    # 保存输入的模糊图像
    input_img_path = os.path.join(args.output, f"{input_name}_input.jpg")
    blur_np = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
    cv2.imwrite(input_img_path, blur_np)
    print(f"输入图像已保存到: {input_img_path}")
    
    # 保存预测的模糊核
    kernels_path = os.path.join(args.output, f"{input_name}_kernels.png")
    print("保存预测的模糊核...")
    save_kernels_grid(kernels.cpu(), kernels_path)
    
    print(f"处理完成，结果保存在 {args.output} 目录中")

if __name__ == "__main__":
    main() 