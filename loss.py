import torch
import torch.nn as nn
import torch.nn.functional as F
class SelfSupervisedLoss(nn.Module):
    def __init__(self, lambda_desc=250, lambda_dist=1.0):
        super().__init__()
        self.lambda_desc = lambda_desc
        self.lambda_dist = lambda_dist
    
    def forward(self, scores1, desc1, scores2, desc2):

        kp_loss = 1.0 - (torch.mean(scores1) + torch.mean(scores2))/2
 
        dist_loss = 1.0 / (self._spatial_variance(scores1) + self._spatial_variance(scores2) + 1e-6)

        desc_loss = 1.0 - F.cosine_similarity(desc1, desc2, dim=1).mean()
        
        total_loss = kp_loss + self.lambda_dist*dist_loss + self.lambda_desc*desc_loss
        return total_loss
    
    def _spatial_variance(self, scores):
        if scores.dim() == 4:
            scores = scores.mean(dim=1)
            
        batch_size, height, width = scores.shape
        grid_y, grid_x = torch.meshgrid(  
            torch.arange(height, device=scores.device).float(),
            torch.arange(width, device=scores.device).float(),
            indexing='ij'
        )
        
        prob = F.softmax(scores.view(batch_size, -1), dim=1).view_as(scores)
        mean_x = (grid_x * prob).sum(dim=(-2, -1))
        mean_y = (grid_y * prob).sum(dim=(-2, -1))
        var_x = ((grid_x - mean_x.view(-1, 1, 1))**2 * prob).sum(dim=(-2, -1))
        var_y = ((grid_y - mean_y.view(-1, 1, 1))**2 * prob).sum(dim=(-2, -1))
        
        return (var_x + var_y).mean()
