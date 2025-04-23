# 模型运行流程说明

## 项目概述

本项目包含多个模糊图像恢复模型的实现版本，主要基于 LMD-ViT 架构和核预测网络。根据不同的内存需求和使用场景，我们提供了以下几个版本：

- **原始版本**：完整的 LMD-ViT 模型，具有最佳性能但内存需求较高
- **仅核预测版本**：独立的核预测网络，用于模糊核估计
- **低内存版本**：优化后的低内存消耗版本，适用于资源有限的设备

## 1. 原始版本运行流程

### 环境准备

```bash
cd LMD-ViT-main
pip install -r requirements.txt
```

### 模型测试

```bash
python test.py --input_path <输入图像路径> --result_dir <结果保存目录> --weights <模型权重路径>
```

### 注意事项

- 原始版本需要较大内存（建议至少 8GB GPU 内存）
- 输入图像尺寸需要是 8 的倍数，否则会自动进行填充
- 默认使用 CUDA 进行加速，如无 GPU 可在脚本中修改 device 为 'cpu'

## 2. 仅核预测版本运行流程

仅核预测版本专注于模糊核的估计，计算量更小，适合快速测试或调试。

### 运行命令

```bash
python test_only_kernel.py --input <输入图像/npy文件路径> --output <输出目录> --kernel_count <核数量> --kernel_size <核大小>
```

### 示例

```bash
python test_only_kernel.py --input val_data/test_sample.npy --output results_kernel_only
```

### 注意事项

- 支持 .npy 格式的数据（包含 'img_blur', 'img_gt', 'blur_mask' 等键）
- 输出包括预测的模糊核可视化图像

## 3. 低内存版本运行流程

低内存版本针对资源受限设备进行了多项优化，包括减少通道数、使用半精度计算、梯度检查点等。

### 运行命令

```bash
python test_low_memory.py --input <输入图像路径> --output <输出目录> --checkpoint <权重文件> --max_size <最大尺寸> --kernel_K <核数量> --kernel_size <核大小>
```

### 示例

```bash
python test_low_memory.py --input ../J-MKPD-main/testing_imgs/car2.jpg --output results_low_mem
```

### 核心优化

- 减少模型深度和宽度
- 逐样本处理批次数据
- 及时释放不需要的中间变量
- 使用 torch.cuda.amp.autocast() 进行半精度计算
- 启用梯度检查点（用于训练时）

### 注意事项

- 确保图像尺寸是 8 的倍数，脚本会自动调整
- 对于较大图像，会自动缩放到指定的最大尺寸（默认为 256）
- 模型权重加载支持非严格匹配，允许部分权重缺失

## 4. 常见问题及解决方案

### 维度不匹配错误

如遇到 "size of tensor a (x) must match size of tensor b (y)" 等张量维度不匹配错误：

- 检查核张量形状是否正确，预期形状为 `[B, K, kernel_size, kernel_size, H, W]`
- 确认 `save_kernels_grid` 函数中的形状处理逻辑是否正确

### 内存不足

如遇到内存不足（CUDA OOM）错误：

- 减小输入图像尺寸（使用 `--max_size` 参数）
- 减少核预测数量（使用 `--kernel_K` 参数）
- 降低核大小（使用 `--kernel_size` 参数）
- 使用低内存版本的模型

### 显示警告但不影响运行的问题

以下警告可以忽略：
- torch.meshgrid 的 indexing 参数警告
- torch.utils.checkpoint 的 use_reentrant 参数警告
- CUDA 不可用时的 autocast 警告（会自动回退到 CPU 计算）

## 5. 调试与开发建议

### 核的可视化与检查

- 使用 `save_kernels_grid` 函数可视化预测的模糊核
- 打印核张量形状以验证是否符合预期

```python
print(f"核张量形状: {kernels.shape}")
```

### 内存优化技巧

- 使用 `del` 及时删除不再需要的大型变量
- 调用 `torch.cuda.empty_cache()` 释放 CUDA 缓存
- 对于大型数据集，考虑使用生成器而非一次性加载所有数据

### 模型修改建议

修改模型架构时，注意：
- 保持输出张量维度的一致性
- 对于核预测网络，确保输出经过 softmax 处理以使每个核的权重和为 1
- 修改通道数时需要相应调整相连层的参数 