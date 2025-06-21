import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

from datasets import PFPascalDataset
from models import FeatureMatchingNet
from loss import MatchingLoss

def train_model(dataset_path, model_save_path, epochs):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = PFPascalDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeatureMatchingNet(output_dim=128).to(device)
    criterion = MatchingLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    all_losses = []
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    line, = ax.plot([], [], 'b-')
    plt.show()

    best_loss = float('inf')
    iteration = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for batch in progress_bar:
            image1 = batch["image1"].to(device)
            image2 = batch["image2"].to(device)
            keypoints1 = batch["keypoints1"].to(device)  # [B, N, 2]
            keypoints2 = batch["keypoints2"].to(device)

            optimizer.zero_grad()
            feat1 = model(image1)
            feat2 = model(image2)
            loss = criterion(feat1, feat2, keypoints1, keypoints2)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            all_losses.append(loss_val)
            epoch_loss += loss_val
            iteration += 1

            progress_bar.set_postfix(loss=f"{loss_val:.4f}")
            line.set_data(range(len(all_losses)), all_losses)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished, Average Loss: {avg_epoch_loss:.4f}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model updated and saved with average loss {best_loss:.4f}!")

    plt.ioff()
    plt.savefig('loss_curve.png')
    print("Training complete, loss curve saved to loss_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train feature matching network")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='PF-Pascal 数据集图片目录')
    parser.add_argument('--model_save_path', type=str, required=True,
                        help='保存模型权重的路径，例如 model.pth')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数，默认为10')
    args = parser.parse_args()

    train_model(args.dataset_path, args.model_save_path, args.epochs)
