import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class UnsupervisedPFPascal(Dataset):
    def __init__(self, root_dir, split='train', img_size=512):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 加载图像对
        with open(os.path.join(root_dir, f'image_pairs/{split}_pairs.csv')) as f:
            self.pairs = [line.strip().split(',')[:2] for line in f.readlines()[1:]]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        img1 = Image.open(os.path.join( 'datasets', img1)).convert('L')
        img2 = Image.open(os.path.join( 'datasets', img2)).convert('L')
        return self.transform(img1), self.transform(img2)
