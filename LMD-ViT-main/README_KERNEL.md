# LMD-ViT 与 J-MKPD 集成模型

这个项目将J-MKPD（基于核预测网络的盲去模糊）与LMD-ViT（自适应窗口剪枝的局部运动去模糊）集成在一起，使LMD-ViT能够显式地预测和利用模糊核进行去模糊处理。

## 功能和特点

- 在LMD-ViT基础上增加了模糊核预测功能
- 可以输出预测的模糊核，便于分析和可视化
- 保持了LMD-ViT原有的高质量去模糊效果
- 支持加载预训练的模型权重

## 环境配置

请确保您已安装以下依赖：

```bash
pip install -r requirements.txt
```

此外，还需要安装J-MKPD的依赖：

```bash
pip install torch torchvision
```

## 模型文件说明

- `model_hw_with_kernel.py`: 包含集成了模糊核预测网络的LMD-ViT模型
- `test_with_kernel.py`: 使用集成模型进行测试的脚本

## 使用方法

### 测试

使用以下命令运行测试脚本：

```bash
python test_with_kernel.py \
  --input_dir /path/to/test/data \
  --result_dir ./results_with_kernel \
  --ckpt_path ./ckpt/model_LMDVIT.pth \
  --kernel_ckpt_path /path/to/kernel/model.pth \
  --kernel_num 9 \
  --kernel_size 33
```

参数说明：
- `--input_dir`: 测试数据目录，包含.npy格式的模糊图像和真实图像
- `--result_dir`: 结果保存目录
- `--ckpt_path`: LMD-ViT模型权重路径
- `--kernel_ckpt_path`: 模糊核预测网络权重路径（可选）
- `--kernel_num`: 要预测的模糊核数量
- `--kernel_size`: 模糊核的大小

### 结果

测试结果将保存在指定的`result_dir`目录下，包括：
- 去模糊后的图像
- `kernels/`子目录：包含预测的模糊核可视化结果

## 模型架构

该集成模型包含两个主要部分：
1. **模糊核预测网络**：从J-MKPD中提取的模块，负责预测局部模糊核
2. **LMD-ViT网络**：基于Transformer的去模糊网络，处理图像的局部模糊

这两个网络协同工作，模糊核预测网络提供模糊核信息，LMD-ViT网络利用这些信息进行更精确的去模糊处理。

## 引用

如果您在研究中使用了本模型，请引用以下论文：

```
@inproceedings{wang2023lmd,
  title={Adaptive Window Pruning for Efficient Local Motion Deblurring},
  author={Wang, Hao and He, Yutong and Yu, Haofeng and others},
  booktitle={CVPR},
  year={2023}
}

@article{janini2022blind,
  title={Blind Motion Deblurring With Pixel-Wise Kernel Estimation via Kernel Prediction Networks},
  author={Janini, Pablo and Cavallaro, Andrea},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={6236--6249},
  year={2022}
}
``` 