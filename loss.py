import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingLoss(nn.Module):
    def __init__(self):
        super(MatchingLoss, self).__init__()

    def forward(self, feat1, feat2, keypoints1, keypoints2):
        B, C, H, W = feat1.shape
        
        # 将关键点坐标归一化到 [-1, 1]
        kp1_norm = keypoints1.clone()
        kp1_norm[..., 0] = (kp1_norm[..., 0] / (W - 1)) * 2 - 1
        kp1_norm[..., 1] = (kp1_norm[..., 1] / (H - 1)) * 2 - 1

        kp2_norm = keypoints2.clone()
        kp2_norm[..., 0] = (kp2_norm[..., 0] / (W - 1)) * 2 - 1
        kp2_norm[..., 1] = (kp2_norm[..., 1] / (H - 1)) * 2 - 1

        grid1 = kp1_norm.unsqueeze(2)   # [B, N, 1, 2]
        grid2 = kp2_norm.unsqueeze(2)   # [B, N, 1, 2]

        # 从特征图中采样描述子
        desc1 = F.grid_sample(feat1, grid1, align_corners=True).squeeze(-1)  # [B, C, N]
        desc2 = F.grid_sample(feat2, grid2, align_corners=True).squeeze(-1)  # [B, C, N]

        # L2归一化描述子
        desc1 = F.normalize(desc1, p=2, dim=1)
        desc2 = F.normalize(desc2, p=2, dim=1)

        dot = torch.sum(desc1 * desc2, dim=1)  # [B, N]
        loss = torch.mean(1 - dot)
        return loss
