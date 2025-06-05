import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models import supernet
from dataset import UnsupervisedPFPascal
from loss import SelfSupervisedLoss
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据加载
    train_set = UnsupervisedPFPascal(args.dataset_path, 'train', args.img_size)
    val_set = UnsupervisedPFPascal(args.dataset_path, 'val', args.img_size)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # 模型初始化
    model = supernet().to(device)
    
    # 损失函数和优化器
    criterion = SelfSupervisedLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # 训练记录
    train_losses, val_losses = [], []
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        tqdm_desc = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for img1, img2 in tqdm_desc:
            img1, img2 = img1.to(device), img2.to(device)
            
            optimizer.zero_grad()
            scores1, desc1 = model(img1)
            scores2, desc2 = model(img2)
            
            loss = criterion(scores1, desc1, scores2, desc2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            tqdm_desc.set_postfix(loss=loss.item())
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2 in val_loader:
                img1, img2 = img1.to(device), img2.to(device)
                scores1, desc1 = model(img1)
                scores2, desc2 = model(img2)
                val_loss += criterion(scores1, desc1, scores2, desc2).item()
        # 记录损失
        train_loss = epoch_loss / len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
        
        # 保存检查点
        if (epoch+1) % args.save_interval == 0:
            torch.save(model.state_dict(), 
                      os.path.join(args.output_dir, f'epoch_{epoch+1}.pth'))
        
        # 打印信息
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # 保存最终模型和loss曲线
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_interval', type=int, default=10)
    args = parser.parse_args()
    
    train(args)
