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

def load_npy_file(file_path):
    """加载.npy文件并提取图像数据"""
    try:
        # 加载NPY文件，允许加载Python对象
        data = np.load(file_path, allow_pickle=True)
        print(f"NPY文件加载成功，类型: {type(data)}, 形状: {data.shape if hasattr(data, 'shape') else '不适用'}")
        
        # 处理标量数组包含字典的情况
        if data.shape == ():
            data_dict = data.item()
            if isinstance(data_dict, dict):
                print(f"检测到字典数据，键: {list(data_dict.keys())}")
                
                # 获取模糊图像
                if 'img_blur' in data_dict:
                    blur_img = data_dict['img_blur']
                    print(f"使用img_blur，形状: {blur_img.shape}")
                    
                    # 将图像转换为张量
                    if len(blur_img.shape) == 3:  # [H, W, C]
                        # 确保是RGB图像
                        if blur_img.shape[2] == 1:
                            blur_img_rgb = np.repeat(blur_img, 3, axis=2)
                        else:
                            blur_img_rgb = blur_img
                            
                        # 将图像转换为Tensor [1, C, H, W]
                        img_tensor = torch.from_numpy(blur_img.transpose(2, 0, 1)).float()
                        if img_tensor.max() > 1.0:
                            img_tensor = img_tensor / 255.0
                        img_tensor = img_tensor.unsqueeze(0)
                        
                        # 转换为可显示的图像
                        display_img = blur_img_rgb
                        if display_img.max() <= 1.0:
                            display_img = (display_img * 255).astype(np.uint8)
                        else:
                            display_img = display_img.astype(np.uint8)
                            
                        return display_img, img_tensor
                    else:
                        raise ValueError(f"不支持的图像形状: {blur_img.shape}")
                else:
                    raise ValueError("在字典中找不到'img_blur'键")
            else:
                raise ValueError(f"数据不是字典类型: {type(data_dict)}")
        else:
            raise ValueError(f"不支持的NPY数据形状: {data.shape}")
    except Exception as e:
        print(f"处理NPY文件时出错: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='val_data/test_sample.npy', help='输入.npy文件路径')
    parser.add_argument('--output', type=str, default='results_kernel_only', help='输出目录')
    parser.add_argument('--kernel_K', type=int, default=9, help='预测模糊核的数量')
    parser.add_argument('--kernel_size', type=int, default=33, help='模糊核的大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"找不到输入文件: {args.input}")
        return
        
    # 加载图像
    print(f"加载NPY文件: {args.input}")
    img_raw, img_tensor = load_npy_file(args.input)
    
    # 调整图像大小为固定尺寸
    img_tensor = F.interpolate(img_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    print(f"已调整图像尺寸为: {img_tensor.shape[2]}x{img_tensor.shape[3]}")
        
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
        img_tensor = img_tensor.to(device)
        kernels = kernel_model(img_tensor)
    
    # 保存输入的模糊图像
    blur_np = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.output, 'blurred.jpg'), blur_np)
    
    # 保存预测的模糊核
    print("保存预测的模糊核...")
    save_kernels_grid(kernels.cpu(), os.path.join(args.output, 'kernels.png'))
    
    print(f"处理完成，结果保存在 {args.output} 目录中")

if __name__ == "__main__":
    main() 