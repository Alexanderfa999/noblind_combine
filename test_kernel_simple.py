import torch
import numpy as np
import cv2
import os
from torchvision.utils import make_grid, save_image
from kernel_model import LMDWithKernel

def save_kernels_grid(kernels, output_path):
    """将预测的模糊核保存为图像网格"""
    n_kernels = kernels.shape[1]
    kernels_ext = kernels[0, :, :, :]  # 选择第一个样本的所有核
    kernels_ext = kernels_ext[:, None, :, :]  # 添加通道维度
    
    # 创建核的网格图
    kernels_grid = make_grid(kernels_ext, nrow=int(np.sqrt(n_kernels)), normalize=True, scale_each=True, pad_value=1)
    
    # 保存为图像
    save_image(kernels_grid, output_path)
    print(f'Kernels saved to {output_path}')
    
    return kernels_grid

def main():
    # 创建输出目录
    os.makedirs('results_simple', exist_ok=True)
    
    # 获取测试图像
    image_path = 'J-MKPD-main/testing_imgs/car2.jpg'
    if not os.path.exists(image_path):
        print(f"测试图像不存在: {image_path}")
        return
    
    # 读取图像
    blur_img = cv2.imread(image_path)
    blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
    
    # 转换为PyTorch张量
    blur_tensor = torch.from_numpy(blur_img).permute(2, 0, 1).float() / 255.0
    blur_tensor = blur_tensor.unsqueeze(0)  # 添加批次维度
    
    # 创建并运行模型
    device = torch.device('cpu')
    model = LMDWithKernel(kernel_pred_K=9, kernel_size=33)
    model.to(device)
    model.eval()
    
    # 推理
    with torch.no_grad():
        deblurred, gate_x, kernels, _, _ = model(blur_tensor)
    
    # 保存结果
    deblurred = torch.clamp(deblurred, 0, 1)
    deblurred_np = deblurred[0].permute(1, 2, 0).numpy() * 255
    deblurred_np = deblurred_np.astype(np.uint8)
    deblurred_np = cv2.cvtColor(deblurred_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite('results_simple/deblurred.jpg', deblurred_np)
    
    # 保存预测的模糊核
    save_kernels_grid(kernels, 'results_simple/kernels.png')
    
    print("处理完成，结果保存在results_simple目录中")

if __name__ == "__main__":
    main() 