import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from torchvision.utils import make_grid, save_image
from model_hw_with_full_kernel import FullLMDWithKernel


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


def load_checkpoint(model, checkpoint_path):
    """加载模型权重"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        model.load_lmd_weights(checkpoint)
    else:
        print("权重格式不兼容，无法加载")
    
    return model


def load_img(path):
    """加载并预处理图像"""
    # 检查路径是否是文件还是目录
    if os.path.isdir(path):
        # 查找目录中的.npy文件
        npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]
        if npy_files:
            # 加载第一个.npy文件
            npy_path = os.path.join(path, npy_files[0])
            print(f"加载NPY文件: {npy_path}")
            try:
                data = np.load(npy_path, allow_pickle=True)
                print(f"NPY文件加载成功，类型: {type(data)}, 形状: {data.shape if hasattr(data, 'shape') else '不适用'}")
                
                # 如果是标量数组，可能包含一个字典
                if data.shape == ():
                    data_dict = data.item()
                    if isinstance(data_dict, dict):
                        print(f"检测到字典数据，键: {list(data_dict.keys())}")
                        # 优先使用模糊图像
                        if 'img_blur' in data_dict:
                            img_array = data_dict['img_blur']
                            print(f"使用img_blur，形状: {img_array.shape if hasattr(img_array, 'shape') else '不适用'}")
                            
                            if isinstance(img_array, np.ndarray):
                                # 根据维度处理
                                if len(img_array.shape) == 3:  # [H, W, C]
                                    img = img_array
                                    # 确保是RGB图像
                                    if img.shape[2] == 1:
                                        img = np.repeat(img, 3, axis=2)
                                    # 转为张量 [C, H, W]
                                    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
                                    if img_tensor.max() > 1.0:
                                        img_tensor = img_tensor / 255.0
                                    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
                                    
                                    # 确保图像是0-255范围
                                    if img.max() <= 1.0:
                                        img = (img * 255).astype(np.uint8)
                                    else:
                                        img = img.astype(np.uint8)
                                    
                                    return img, img_tensor
                                elif len(img_array.shape) == 4:  # [B, H, W, C] or [B, C, H, W]
                                    # 假设是 [B, H, W, C]
                                    if img_array.shape[3] in [1, 3]:
                                        img = img_array[0]  # 取第一个样本
                                        # 确保是RGB图像
                                        if img.shape[2] == 1:
                                            img = np.repeat(img, 3, axis=2)
                                        # 转为张量 [C, H, W]
                                        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
                                        if img_tensor.max() > 1.0:
                                            img_tensor = img_tensor / 255.0
                                        img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
                                    else:  # 假设是 [B, C, H, W]
                                        img_tensor = torch.from_numpy(img_array).float()
                                        if img_tensor.max() > 1.0:
                                            img_tensor = img_tensor / 255.0
                                        img = img_array[0].transpose(1, 2, 0)  # [H, W, C]
                                        # 确保是RGB图像
                                        if img.shape[2] == 1:
                                            img = np.repeat(img, 3, axis=2)
                                    
                                    # 确保图像是0-255范围
                                    if img.max() <= 1.0:
                                        img = (img * 255).astype(np.uint8)
                                    else:
                                        img = img.astype(np.uint8)
                                    
                                    return img, img_tensor
                                else:
                                    raise ValueError(f"不支持的图像形状: {img_array.shape}")
                            elif isinstance(img_array, torch.Tensor):
                                # 处理张量格式
                                img_tensor = img_array
                                if len(img_tensor.shape) == 3:  # [C, H, W]
                                    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
                                # 转为numpy数组用于显示
                                img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
                                # 确保是RGB图像
                                if img.shape[2] == 1:
                                    img = np.repeat(img, 3, axis=2)
                                # 确保图像是0-255范围
                                if img.max() <= 1.0:
                                    img = (img * 255).astype(np.uint8)
                                else:
                                    img = img.astype(np.uint8)
                                
                                return img, img_tensor
                            else:
                                raise ValueError(f"不支持的图像类型: {type(img_array)}")
                        else:
                            raise ValueError(f"在字典中找不到'img_blur'键")
                    else:
                        raise ValueError(f"数据不是字典类型: {type(data_dict)}")
                else:
                    # 处理普通的numpy数组
                    img_array = data
                    # 处理后续逻辑与之前相同...
                    raise ValueError(f"暂不支持非字典类型的NPY数据: {data.shape}")
            except Exception as e:
                print(f"处理NPY文件时出错: {e}")
                raise
        else:
            raise FileNotFoundError(f"在目录 {path} 中找不到.npy文件")
    else:
        # 当path是文件路径时的处理
        if path.endswith('.npy'):
            # 与上面目录处理逻辑相同，这里简化处理
            print(f"加载NPY文件: {path}")
            data = np.load(path, allow_pickle=True)
            
            # 如果是标量数组，可能包含一个字典
            if data.shape == ():
                data_dict = data.item()
                if isinstance(data_dict, dict):
                    print(f"检测到字典数据，键: {list(data_dict.keys())}")
                    # 优先使用模糊图像
                    if 'img_blur' in data_dict:
                        img_array = data_dict['img_blur']
                        print(f"使用img_blur，形状: {img_array.shape if hasattr(img_array, 'shape') else '不适用'}")
                        
                        if isinstance(img_array, np.ndarray):
                            # 根据维度处理
                            if len(img_array.shape) == 3:  # [H, W, C]
                                img = img_array
                                # 确保是RGB图像
                                if img.shape[2] == 1:
                                    img = np.repeat(img, 3, axis=2)
                                # 转为张量 [C, H, W]
                                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
                                if img_tensor.max() > 1.0:
                                    img_tensor = img_tensor / 255.0
                                img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
                                
                                # 确保图像是0-255范围
                                if img.max() <= 1.0:
                                    img = (img * 255).astype(np.uint8)
                                else:
                                    img = img.astype(np.uint8)
                                
                                return img, img_tensor
                            elif len(img_array.shape) == 4:  # [B, H, W, C] or [B, C, H, W]
                                # 假设是 [B, H, W, C]
                                if img_array.shape[3] in [1, 3]:
                                    img = img_array[0]  # 取第一个样本
                                    # 确保是RGB图像
                                    if img.shape[2] == 1:
                                        img = np.repeat(img, 3, axis=2)
                                    # 转为张量 [C, H, W]
                                    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
                                    if img_tensor.max() > 1.0:
                                        img_tensor = img_tensor / 255.0
                                    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
                                else:  # 假设是 [B, C, H, W]
                                    img_tensor = torch.from_numpy(img_array).float()
                                    if img_tensor.max() > 1.0:
                                        img_tensor = img_tensor / 255.0
                                    img = img_array[0].transpose(1, 2, 0)  # [H, W, C]
                                    # 确保是RGB图像
                                    if img.shape[2] == 1:
                                        img = np.repeat(img, 3, axis=2)
                                
                                # 确保图像是0-255范围
                                if img.max() <= 1.0:
                                    img = (img * 255).astype(np.uint8)
                                else:
                                    img = img.astype(np.uint8)
                                
                                return img, img_tensor
                            else:
                                raise ValueError(f"不支持的图像形状: {img_array.shape}")
                        elif isinstance(img_array, torch.Tensor):
                            # 处理张量格式
                            img_tensor = img_array
                            if len(img_tensor.shape) == 3:  # [C, H, W]
                                img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
                            # 转为numpy数组用于显示
                            img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
                            # 确保是RGB图像
                            if img.shape[2] == 1:
                                img = np.repeat(img, 3, axis=2)
                            # 确保图像是0-255范围
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                            
                            return img, img_tensor
                        else:
                            raise ValueError(f"不支持的图像类型: {type(img_array)}")
                    else:
                        raise ValueError(f"在字典中找不到'img_blur'键")
                else:
                    raise ValueError(f"数据不是字典类型: {type(data_dict)}")
            else:
                # 处理普通的numpy数组
                img_array = data
                # 处理后续逻辑与之前相同...
                raise ValueError(f"暂不支持非字典类型的NPY数据: {data.shape}")
        else:
            # 处理常规图像文件
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"无法读取图像文件: {path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 如果图像过大，先调整大小
            h, w, _ = img.shape
            max_size = 1024
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                h, w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (w, h))
            
            # 转为张量
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
            
            return img, img_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='val_data', help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='results_full', help='输出目录')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/GoPro.pth', help='LMD-ViT 权重路径')
    parser.add_argument('--kernel_K', type=int, default=9, help='预测模糊核的数量')
    parser.add_argument('--kernel_size', type=int, default=33, help='模糊核的大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 检查预训练模型是否存在
    if os.path.exists(args.checkpoint):
        print(f"使用权重: {args.checkpoint}")
    else:
        print(f"找不到权重: {args.checkpoint}")
        print("将使用随机初始化模型...")
    
    # 读取图像
    if not os.path.exists(args.input):
        print(f"找不到输入图像或目录: {args.input}")
        return
    
    img_raw, img_tensor = load_img(args.input)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 确保图像尺寸是8的倍数，以适应窗口大小要求
    _, _, h, w = img_tensor.shape
    new_h = ((h + 7) // 8) * 8  # 向上取整到8的倍数
    new_w = ((w + 7) // 8) * 8  # 向上取整到8的倍数
    
    print(f"原始图像尺寸: {h}x{w}")
    print(f"调整后尺寸: {new_h}x{new_w}")
    
    if h != new_h or w != new_w:
        # 使用插值调整图像大小
        img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        print(f"已调整图像尺寸为: {img_tensor.shape[2]}x{img_tensor.shape[3]}")
    
    model = FullLMDWithKernel(kernel_pred_K=args.kernel_K, kernel_size=args.kernel_size)
    
    # 加载预训练权重
    if os.path.exists(args.checkpoint):
        model = load_checkpoint(model, args.checkpoint)
    
    model.to(device)
    model.eval()
    
    # 记录原始图像尺寸以便恢复
    orig_h, orig_w = h, w
    
    # 图像推理
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        
        # 推理
        deblurred, gate_x, kernels, _, _ = model(img_tensor)
        
        # 如果进行了尺寸调整，恢复到原始尺寸
        if h != new_h or w != new_w:
            deblurred = F.interpolate(deblurred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    
    # 输出图像转为numpy并保存
    deblurred_np = deblurred[0].permute(1, 2, 0).cpu().numpy() * 255
    deblurred_np = np.clip(deblurred_np, 0, 255).astype(np.uint8)
    deblurred_np = cv2.cvtColor(deblurred_np, cv2.COLOR_RGB2BGR)
    
    # 保存结果
    cv2.imwrite(os.path.join(args.output, 'deblurred.jpg'), deblurred_np)
    
    # 保存输入的模糊图像
    blur_np = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.output, 'blurred.jpg'), blur_np)
    
    # 保存预测的模糊核
    save_kernels_grid(kernels.cpu(), os.path.join(args.output, 'kernels.png'))
    
    print(f"处理完成，结果保存在 {args.output} 目录中")


if __name__ == "__main__":
    main() 