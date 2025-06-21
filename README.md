# Feature Matching Network

基于深度学习的图像特征点匹配与跟踪系统。

## 项目结构

```
.
├── datasets.py         # 自定义数据集 PFPascalDataset
├── models.py           # FeatureMatchingNet 模型结构
├── loss.py             # MatchingLoss 损失函数
├── train.py            # 模型训练脚本
├── track.py            # 视频特征点匹配与轨迹可视化
└── best.pth           # 训练好的模型参数（训练后保存）
```

## 快速开始

### 1.安装依赖

```bash
pip install torch torchvision numpy pillow moviepy scikit-image matplotlib tqdm
```

### 2.模型训练

```bash
python train.py \
    --dataset_path ./data/PF-Pascal/images \
    --model_save_path best.pth 
```

训练完成后将自动保存模型参数为 `best.pth`，并生成损失曲线图 `loss_curve.png`。

### 3.视频测试

```bash
python track.py \
    --video_path test.mp4 \
    --model_path best.pth \
    --output_path result.mp4 
```
输出视频将展示每帧之间的关键点匹配连线。

## 模块说明

- **datasets.py**：构建图像对数据集，采用 Harris 角点检测 + 仿射扰动生成关键点匹配
- **models.py**：基于 ResNet-18 的轻量级特征提取网络
- **loss.py**：通过特征点采样 + 点积相似度构建 MatchingLoss 损失函数
- **train.py**：训练流程，包括损失曲线可视化与模型保存
- **track.py**：逐帧检测 + 特征提取 + 描述子匹配 + 可视化输出轨迹
