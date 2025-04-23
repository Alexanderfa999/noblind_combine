import torch
from model_hw_with_kernel import LMDWithKernel
from test_flops import test_flops
import glob
import numpy as np
import torch.nn.functional as F
import pyiqa
import argparse
import os
import cv2
from skimage import img_as_ubyte
from tqdm import tqdm
import utils
from utils.timer import Timer
from utils import image_utils
from utils.image_utils import batch_weighted_PSNR, SSIM, cal_prec_accu_reca   
from PIL import Image
import math
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

def expand2rect(timg, blur_mask, factor=16.0):
    _, _, h, w = timg.size()

    Xh = int(math.ceil(h / float(factor)) * factor)
    Xw = int(math.ceil(w / float(factor)) * factor)

    img = torch.zeros(1, 3, Xh, Xw).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, Xh, Xw).type_as(timg)

    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((Xh - h) // 2):((Xh - h) // 2 + h), ((Xw - w) // 2):((Xw - w) // 2 + w)] = timg
    mask[:, :, ((Xh - h) // 2):((Xh - h) // 2 + h), ((Xw - w) // 2):((Xw - w) // 2 + w)].fill_(1)
    # mask[:, :, ((Xh - h) // 2):((Xh - h) // 2 + h), ((Xw - w) // 2):((Xw - w) // 2 + w)] = torch.unsqueeze(blur_mask[:,0,:,:],dim=1)

    return img, mask

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

def main(args):
    
    test_paths = sorted(glob.glob(args.input_dir+ '/*.npy'))
    print(f"Found {len(test_paths)} test samples")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 创建LMDWithKernel模型
    net = LMDWithKernel(
            img_size=512, embed_dim=32, win_size=8, token_projection='linear',
            token_mlp='lefflocal',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=3,
            drop_path_rate=0.1, prune_loc=[1, 1, 1, 1, 1, 1, 1, 1, 1],
            kernel_pred_K=args.kernel_num, kernel_size=args.kernel_size)
    
    if args.ckpt_path is not None:
        print(f"Loading checkpoint from {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path)
        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        checkpoint['state_dict'] = new_state_dict
        
        # 只加载LMD模型的权重，不加载模糊核预测网络的权重
        if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            missing_keys, unexpected_keys = net.module.base_lmd.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            missing_keys, unexpected_keys = net.base_lmd.load_state_dict(checkpoint["state_dict"], strict=False)
        
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
            
    # 如果提供了模糊核预测网络的权重，单独加载
    if args.kernel_ckpt_path is not None:
        print(f"Loading kernel prediction network weights from {args.kernel_ckpt_path}")
        kernel_checkpoint = torch.load(args.kernel_ckpt_path, map_location=device)
        
        # 仅加载KernelPredictionNetwork部分的权重
        # 需要处理权重名称不匹配的问题
        kernel_state_dict = {}
        for k, v in kernel_checkpoint.items():
            if k.startswith('module.'):
                k = k[7:]  # 移除'module.'前缀
            kernel_state_dict[k] = v
        
        if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            missing_keys, unexpected_keys = net.module.kernel_prediction.load_state_dict(kernel_state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = net.kernel_prediction.load_state_dict(kernel_state_dict, strict=False)
            
        print(f"Kernel network - Missing keys: {missing_keys}")
        print(f"Kernel network - Unexpected keys: {unexpected_keys}")
    
    net = net.to(device)
    
    # 创建输出目录
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'kernels'), exist_ok=True)

    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = SSIM(window_size=11, size_average=False)
    psnr_count, ssim_count, w_psnr_count, w_ssim_count, sample_count = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        net.eval()
        for idx, test_path in enumerate(tqdm(test_paths)):
            sample_count += 1
            split_path = test_path.split('/')
            sample_name = split_path[-1].split('.')[0]  # 获取样本名称

            sample = np.load(test_path, allow_pickle=True).item()
            blur = sample['img_blur']
            gt = sample['img_gt']
            mask = np.tile(np.uint8(sample['blur_mask'] * 255)[:, :, None], [1, 1, 3])
            
            blur = torch.from_numpy(blur).permute(2,0,1).float().div(255).unsqueeze(dim=0).to(device)
            gt = torch.from_numpy(gt).permute(2,0,1).float().div(255).unsqueeze(dim=0).to(device)
            mask = torch.from_numpy(mask).permute(2,0,1).float().div(255).unsqueeze(dim=0).to(device)
            mask_shortcut = mask # 3 channels
            
            blur, mask_pad = expand2rect(blur, mask, factor=128)
            _, _, H, W = blur.shape
            
            if sample_count == 1:
                deblur, _1, kernels, pred_score_list, decision_list = net(blur)
                mask_shortcut_ = torch.unsqueeze(mask_shortcut[:,0,:,:],dim=0)            

            with Timer(enable=True, name='test'):
                # 使用集成后的模型进行预测
                deblur, _1, kernels, pred_score_list, decision_list = net(blur)
                mask_shortcut_ = torch.unsqueeze(mask_shortcut[:,0,:,:], dim=0)
                
                # 保存预测的模糊核
                kernel_output_path = os.path.join(args.result_dir, 'kernels', f'{sample_name}_kernels.png')
                save_kernels_grid(kernels, kernel_output_path)
                    
            deblur = torch.masked_select(deblur, mask.bool()).reshape(1, 3, gt.shape[2], gt.shape[3])
            deblur = deblur[:, :, :H, :W]
            deblur_save = torch.clamp(deblur,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            
            # 保存去模糊结果
            result_path = os.path.join(args.result_dir, f'{sample_name}.png')
            cv2.imwrite(result_path, cv2.cvtColor(img_as_ubyte(deblur_save), cv2.COLOR_RGB2BGR))
            
            deblur = torch.clamp(deblur, 0, 1)
            gt = torch.clamp(gt, 0, 1)
            psnr_count += psnr_metric(deblur, gt).item()
            ssim_count += -ssim_metric(deblur, gt).sum().item()
            w_psnr_count += batch_weighted_PSNR(deblur, gt, mask_shortcut).sum().item()
            w_ssim_count += -ssim_metric(deblur, gt, mask_shortcut).sum().item()
        
        psnr_ave = psnr_count / sample_count
        ssim_ave = ssim_count / sample_count
        w_psnr_ave = w_psnr_count / sample_count
        w_ssim_ave = w_ssim_count / sample_count
        print('average psnr:', psnr_ave, 'average ssim:', ssim_ave, 'weighted psnr:', w_psnr_ave, 'weighted ssim:', w_ssim_ave)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single Image Local Motion Deblurring with Kernel Prediction')
    parser.add_argument('--input_dir', default='./val_data', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='./results_with_kernel', type=str, help='Directory for results')
    parser.add_argument('--ckpt_path', default='./ckpt/model_LMDVIT.pth', type=str, help='Path to LMD-ViT weights')
    parser.add_argument('--kernel_ckpt_path', default=None, type=str, help='Path to kernel prediction network weights')
    parser.add_argument('--kernel_num', type=int, default=9, help='Number of kernels to predict')
    parser.add_argument('--kernel_size', type=int, default=33, help='Size of the kernel')
    parser.add_argument('--dataset', default='ReLoBlur', type=str, help='Test Dataset')
    args = parser.parse_args()
    main(args) 