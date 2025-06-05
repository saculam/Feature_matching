import torch
import torch.nn as nn
import torch.nn.functional as F

class supernet(nn.Module):
    def __init__(self, grid_size=8):
        super(supernet, self).__init__()
        self.grid_size = grid_size
        
        # 特征提取器 (约15M参数)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 1/2 (H/2, W/2)
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 1/4 (H/4, W/4)
            
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 1/8 (H/8, W/8)
            
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        )
        
        # 关键点检测头 (约3M参数)
        self.detector = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, (grid_size**2)+1, 1)  # 65 channels for 8x8 grid
        )
        
        # 描述符头 (约2M参数)
        self.descriptor = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 256, 1)
        )
        
        # 参数初始化
        self._init_weights()
        
        # 打印参数数量
        total = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {total/1e6:.2f}M parameters")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        features = self.encoder(x)
    
        # 关键点检测
        detector_out = self.detector(features)  # 形状: [B, 65, H/8, W/8]
        
        scores = torch.softmax(detector_out, dim=1)[:, :-1]  # 去掉最后一维 [B, 64, H/8, W/8]
        
        # 调整形状为 [B, H, W]
        scores = scores.permute(0, 2, 3, 1)  # [B, H/8, W/8, 64]
        
        scores = scores.reshape(scores.size(0), -1, 8, 8)  # [B, (H/8)*(W/8), 8, 8]
        scores = nn.functional.pixel_shuffle(scores, 64)  # [B, 1, H, W]
        scores = scores.squeeze(1)  # 最终形状: [B, H, W]
        
        # 描述符计算
        descriptors = self.descriptor(features)  # 保持形状 [B, 256, H/8, W/8]
        descriptors = nn.functional.normalize(descriptors, p=2, dim=1)
        
        return scores, descriptors
