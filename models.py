import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FeatureMatchingNet(nn.Module):
    def __init__(self, output_dim=128):

        super(FeatureMatchingNet, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.conv_last = nn.Conv2d(512, output_dim, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)  # [B, 512, H', W']
        features = self.conv_last(features)  # [B, output_dim, H', W']
       
        features = nn.functional.interpolate(features, size=x.shape[2:], mode='bilinear', align_corners=True)
      
        features = nn.functional.normalize(features, p=2, dim=1)
        return features
