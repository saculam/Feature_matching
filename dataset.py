import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.feature import corner_harris, corner_peaks

class PFPascalDataset(Dataset):
    def __init__(self, image_dir, transform=None, num_keypoints=100):
        super(PFPascalDataset, self).__init__()
        self.image_dir = image_dir
        self.image_files = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.num_keypoints = num_keypoints
        self.transform = transform if transform is not None else transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_files)

    def detect_feature_points(self, image, num_keypoints, min_distance=5):
        
        img_gray = np.array(image.convert('L'))
        response = corner_harris(img_gray)
        coords = corner_peaks(response, min_distance=min_distance)
        keypoints = np.array([(c[1], c[0]) for c in coords])
        if keypoints.shape[0] > num_keypoints:
            responses = response[coords[:, 0], coords[:, 1]]
            order = np.argsort(responses)[::-1]  #
            keypoints = keypoints[order][:num_keypoints]
        return keypoints

    def get_random_homography(self, width, height):
        # 随机生成一个包含旋转、缩放和平移的仿射变换
        angle = np.deg2rad(np.random.uniform(-15, 15))
        scale = np.random.uniform(0.9, 1.1)
        tx = np.random.uniform(-0.1 * width, 0.1 * width)
        ty = np.random.uniform(-0.1 * height, 0.1 * height)
        H = np.array([
            [np.cos(angle) * scale, -np.sin(angle) * scale, tx],
            [np.sin(angle) * scale,  np.cos(angle) * scale, ty],
            [0, 0, 1]
        ])
        return H

    def transform_points(self, pts, H):
        num_pts = pts.shape[0]
        pts_aug = np.concatenate([pts, np.ones((num_pts, 1))], axis=1)  # [N,3]
        pts_transformed = (H @ pts_aug.T).T
        pts_transformed = pts_transformed[:, :2] / pts_transformed[:, 2:3]
        return pts_transformed

    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize((256, 256))
        keypoints1 = self.detect_feature_points(image_resized, self.num_keypoints)
        if keypoints1.size == 0:
            step = max(int(np.sqrt(256 * 256 / self.num_keypoints)), 1)
            xs = np.arange(0, 256, step)
            ys = np.arange(0, 256, step)
            grid = np.array([(x, y) for y in ys for x in xs])
            if grid.shape[0] > self.num_keypoints:
                indices = np.random.choice(grid.shape[0], self.num_keypoints, replace=False)
                keypoints1 = grid[indices]
            else:
                keypoints1 = grid

        width, height = 256, 256  
        H = self.get_random_homography(width, height)
        H_inv = np.linalg.inv(H)
        H_inv_flat = H_inv.flatten()[:8]
        image_transformed = image_resized.transform((width, height), Image.PERSPECTIVE, H_inv_flat, resample=Image.BICUBIC)

        image1_tensor = self.transform(image_resized)
        image2_tensor = self.transform(image_transformed)

        keypoints2 = self.transform_points(keypoints1, H)
        keypoints1 = torch.from_numpy(keypoints1).float()
        keypoints2 = torch.from_numpy(keypoints2).float()

        sample = {
            "image1": image1_tensor,
            "image2": image2_tensor,
            "keypoints1": keypoints1,
            "keypoints2": keypoints2
        }
        return sample
