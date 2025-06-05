import argparse
import cv2
import torch
import numpy as np
from models import supernet

class FeatureTracker:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = supernet().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 跟踪状态
        self.prev_kps = None
        self.prev_desc = None
        self.tracks = {}
        self.next_id = 0
        self.colors = np.random.randint(0, 255, (1000, 3))
    
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(gray).float().to(self.device) / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            scores, desc = self.model(tensor)
        
        # 获取关键点
        scores = scores.squeeze().cpu().numpy()
        kps = np.where(scores > scores.max() * 0.3)  # 阈值设为最大值的30%
        kps = np.stack(kps, axis=1)[:, ::-1]  # 转为(x,y)格式
        
        # 获取描述符
        desc = desc.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        # 匹配关键点
        if self.prev_kps is not None:
            # 简单的暴力匹配
            matches = []
            for i, (x1, y1) in enumerate(self.prev_kps):
                best_dist = float('inf')
                best_idx = -1
                for j, (x2, y2) in enumerate(kps):
                    dist = np.linalg.norm(desc[int(y1), int(x1)] - desc[int(y2), int(x2)])
                    if dist < best_dist and dist < 0.7:  # 匹配阈值
                        best_dist = dist
                        best_idx = j
                if best_idx != -1:
                    matches.append((i, best_idx))
            
            # 更新跟踪点
            updated_tracks = {}
            for old_idx, new_idx in matches:
                if old_idx in self.tracks:
                    updated_tracks[new_idx] = self.tracks[old_idx]
            
            # 添加新点
            for j in range(len(kps)):
                if j not in updated_tracks:
                    updated_tracks[j] = self.next_id
                    self.next_id += 1
            
            self.tracks = updated_tracks
        else:
            self.tracks = {i:i for i in range(len(kps))}
            self.next_id = len(kps)
        
        # 保存状态
        self.prev_kps = kps
        self.prev_desc = desc
        
        # 绘制结果
        display = frame.copy()
        for j, (x, y) in enumerate(kps):
            if j in self.tracks:
                color = self.colors[self.tracks[j] % len(self.colors)].tolist()
                cv2.circle(display, (int(x), int(y)), 5, color, -1)
        
        return display

def process_video(args):
    tracker = FeatureTracker(args.model_path)
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error opening video")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = tracker.process_frame(frame)
        cv2.imshow('Tracking', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--video_path', required=True)
    args = parser.parse_args()
    process_video(args)
