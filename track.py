import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw
from moviepy.editor import VideoFileClip, ImageSequenceClip
import torch
import torchvision.transforms as transforms
from collections import deque
from models import FeatureMatchingNet

def detect_keypoints(pil_img, max_kps=200, min_distance=10):
    gray = np.array(pil_img.convert("L"))
    orb = cv2.ORB_create(nfeatures=max_kps)
    kps = orb.detect(gray, None)
    selected_coords = []
    for kp in sorted(kps, key=lambda k: -k.response):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if all(np.hypot(x - px, y - py) > min_distance for px, py in selected_coords):
            selected_coords.append((x, y))
        if len(selected_coords) >= max_kps:
            break
    return selected_coords

def sample_descriptors(feature_map, keypoints):
    C, H, W = feature_map.shape
    keypoints = np.array(keypoints)
    if len(keypoints) == 0:
        return torch.empty(0, C)
    xs = keypoints[:, 0] / (W - 1) * 2 - 1
    ys = keypoints[:, 1] / (H - 1) * 2 - 1
    grid = torch.from_numpy(np.stack([xs, ys], axis=1)).float().unsqueeze(0).unsqueeze(2)
    descriptors = torch.nn.functional.grid_sample(feature_map.unsqueeze(0), grid, align_corners=True)
    return descriptors.squeeze(0).squeeze(-1).T

def draw_full_trajectories(frame_np, tracked_points, max_length=20):
    img = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(img)
    for pt in tracked_points:
        if len(pt.positions) < 2:
            continue
        pts = pt.positions[-max_length:]
        for i in range(len(pts) - 1):
            alpha = (i + 1) / len(pts)
            color = (int(255 * alpha), 0, 0)
            draw.line((pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]), fill=color, width=8)
        x2, y2 = exponential_smoothing(pt.positions)
        draw.ellipse((x2 - 6, y2 - 6, x2 + 6, y2 + 6), fill=(0, 255, 0))
    return np.array(img)

class TrackedPoint:
    def __init__(self, init_pos, init_desc, point_id):
        self.positions = [init_pos]
        self.descriptors = [init_desc]
        self.id = point_id
        self.lost_frames = 0

    def update(self, new_pos, new_desc):
        self.positions.append(new_pos)
        self.descriptors.append(new_desc)
        self.lost_frames = 0

    def mark_lost(self):
        self.lost_frames += 1

    def latest_position(self):
        return self.positions[-1]

    def smoothed_position(self, window=3):
        pts = self.positions[-window:]
        xs, ys = zip(*pts)
        return (np.mean(xs), np.mean(ys))
        
def exponential_smoothing(track, alpha=0.3):
    if len(track) < 2:
        return track[-1]
    x_prev, y_prev = track[-2]
    x_curr, y_curr = track[-1]
    x_smooth = alpha * x_curr + (1 - alpha) * x_prev
    y_smooth = alpha * y_curr + (1 - alpha) * y_prev
    return (x_smooth, y_smooth)

def track_video(video_path, model_path, output_path,
                num_keypoints, match_threshold,
                trail_len, min_movement, max_movement):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureMatchingNet(output_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    clip = VideoFileClip(video_path)
    fps = clip.fps
    W, H = clip.size
    n_frames = int(fps * clip.duration)

    annotated_frames = []
    match_history = deque(maxlen=trail_len)
    tracked_points = []
    next_id = 0
    MAX_LOST = 5

    f0_np = clip.get_frame(0)
    f0_pil = Image.fromarray(f0_np).resize((256, 256))
    kps0 = detect_keypoints(f0_pil, num_keypoints)
    feat0 = model(transform(f0_pil).unsqueeze(0).to(device))[0].cpu()
    desc0 = sample_descriptors(feat0, kps0)
    kps0_rescaled = [(x * W / 256, y * H / 256) for x, y in kps0]

    # 初始化轨迹
    for i, (pt, desc) in enumerate(zip(kps0_rescaled, desc0)):
        tracked_points.append(TrackedPoint(pt, desc, next_id))
        next_id += 1

    for i in range(1, n_frames):
        f1_np = clip.get_frame(i / fps)
        f1_pil = Image.fromarray(f1_np).resize((256, 256))
        kps1 = detect_keypoints(f1_pil, num_keypoints)
        with torch.no_grad():
            feat1 = model(transform(f1_pil).unsqueeze(0).to(device))[0].cpu()
        desc1 = sample_descriptors(feat1, kps1)
        kps1_rescaled = [(x * W / 256, y * H / 256) for x, y in kps1]

        used_indices = set()
        new_tracked = []

        for pt in tracked_points:
            last_pos = pt.latest_position()
            last_desc = pt.descriptors[-1]
            if len(desc1) == 0:
                pt.mark_lost()
                continue

            sims = torch.nn.functional.cosine_similarity(last_desc.unsqueeze(0), desc1)
            idx = torch.argmax(sims).item()
            match_score = sims[idx].item()

            if idx in used_indices or match_score < match_threshold:
                pt.mark_lost()
                continue

            new_pos = kps1_rescaled[idx]
            dx, dy = new_pos[0] - last_pos[0], new_pos[1] - last_pos[1]
            disp = np.hypot(dx, dy)

            if disp < 3:
                new_pos = last_pos  
            if disp > max_movement:
                pt.mark_lost()
                continue

            if len(pt.positions) >= 3:
                vx = pt.positions[-1][0] - pt.positions[-2][0]
                vy = pt.positions[-1][1] - pt.positions[-2][1]
                pred_x = pt.positions[-1][0] + 0.5 * vx
                pred_y = pt.positions[-1][1] + 0.5 * vy
                pred_dist = np.hypot(pred_x - new_pos[0], pred_y - new_pos[1])
                if pred_dist > 15:  
                    pt.mark_lost()
                    continue

            used_indices.add(idx)
            pt.update(new_pos, desc1[idx])
            new_tracked.append(pt)

        for idx, (pt, desc) in enumerate(zip(kps1_rescaled, desc1)):
            if idx not in used_indices:
                new_tracked.append(TrackedPoint(pt, desc, next_id))
                next_id += 1

        tracked_points = [pt for pt in new_tracked if pt.lost_frames <= MAX_LOST]

        # 可视化匹配
        pt_pairs = []
        for pt in tracked_points:
            if len(pt.positions) < 2:
                continue
            smoothed = exponential_smoothing(pt.positions)
            pt_pairs.append((pt.positions[-2], smoothed))

        match_history.append(pt_pairs)
        annotated = draw_full_trajectories(f1_np, tracked_points, max_length=20)
        annotated_frames.append(annotated)

    out_clip = ImageSequenceClip(annotated_frames, fps=fps)
    out_clip.write_videofile(output_path, codec="libx264", audio=False)
    print(f"已生成：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_keypoints", type=int, default=400)
    parser.add_argument("--match_threshold", type=float, default=0.65)
    parser.add_argument("--trail_len", type=int, default=15)
    parser.add_argument("--min_movement", type=float, default=15.0)
    parser.add_argument("--max_movement", type=float, default=70.0)
    args = parser.parse_args()

    track_video(
        video_path=args.video_path,
        model_path=args.model_path,
        output_path=args.output_path,
        num_keypoints=args.num_keypoints,
        match_threshold=args.match_threshold,
        trail_len=args.trail_len,
        min_movement=args.min_movement,
        max_movement=args.max_movement
    )
