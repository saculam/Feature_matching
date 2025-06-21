import argparse
import numpy as np
from PIL import Image, ImageDraw
from moviepy.editor import VideoFileClip, ImageSequenceClip
from skimage.feature import corner_harris, corner_peaks
import torch
import torchvision.transforms as transforms
from models import FeatureMatchingNet


def detect_keypoints(image_pil, max_points=200, min_distance=5):
    gray = np.array(image_pil.convert("L"))
    response = corner_harris(gray)
    coords = corner_peaks(response, min_distance=min_distance)
    if len(coords) > max_points:
        scores = response[coords[:, 0], coords[:, 1]]
        idx = np.argsort(scores)[::-1][:max_points]
        coords = coords[idx]
    return coords  # shape [N, 2], (y,x)


def sample_descriptors(feature_map, keypoints):
    C, H, W = feature_map.shape
    keypoints = np.array(keypoints)
    xs = keypoints[:, 1] / (W - 1) * 2 - 1
    ys = keypoints[:, 0] / (H - 1) * 2 - 1
    grid = torch.from_numpy(np.stack([xs, ys], axis=1)).float().unsqueeze(0).unsqueeze(2)  # [1,N,1,2]
    descriptors = torch.nn.functional.grid_sample(feature_map.unsqueeze(0), grid, align_corners=True)
    return descriptors.squeeze(0).squeeze(-1).T  # [N,C]


def match_descriptors(desc1, desc2, threshold=0.8):
    desc1 = torch.nn.functional.normalize(desc1, p=2, dim=1)
    desc2 = torch.nn.functional.normalize(desc2, p=2, dim=1)
    sim = torch.matmul(desc1, desc2.T)  # [N1,N2]
    top1 = torch.argmax(sim, dim=1)     # desc1 → desc2
    reverse_sim = torch.matmul(desc2, desc1.T)
    top2 = torch.argmax(reverse_sim, dim=1)  # desc2 → desc1

    mutual = []
    for i, j in enumerate(top1):
        if top2[j] == i and sim[i, j] > threshold:
            mutual.append((i, j.item()))
    return mutual


def rescale_points(keypoints, src_size, dst_size):
    scale_x = dst_size[0] / src_size[0]
    scale_y = dst_size[1] / src_size[1]
    return [(x * scale_x, y * scale_y) for y, x in keypoints]


def draw_all_matches(image_np, match_history):
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    for (x1, y1), (x2, y2) in match_history:
        draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=1)
        draw.ellipse((x2 - 2, y2 - 2, x2 + 2, y2 + 2), fill=(0, 255, 0))
    return np.array(img)


def track_video(video_path, model_path, output_path, num_keypoints=200,
                match_threshold=0.8, min_movement=3, max_movement=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureMatchingNet(output_dim=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    clip = VideoFileClip(video_path)
    fps = clip.fps
    n_frames = int(fps * clip.duration)
    W_orig, H_orig = clip.size

    annotated_frames = []
    match_history = []

    frame0_np = clip.get_frame(0)
    frame0_pil = Image.fromarray(frame0_np)
    frame0_resized = frame0_pil.resize((256, 256))
    kps_prev = detect_keypoints(frame0_resized, max_points=num_keypoints)
    input0 = transform(frame0_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_prev = model(input0)[0].cpu()
    desc_prev = sample_descriptors(feat_prev, kps_prev)
    kps_prev_orig = rescale_points(kps_prev, (256, 256), (W_orig, H_orig))

    for i in range(1, n_frames):
        t = i / fps
        frame_np = clip.get_frame(t)
        frame_pil = Image.fromarray(frame_np)
        frame_resized = frame_pil.resize((256, 256))

        kps_curr = detect_keypoints(frame_resized, max_points=num_keypoints)
        if len(kps_prev) == 0 or len(kps_curr) == 0:
            annotated_frames.append(frame_np)
            kps_prev = kps_curr
            kps_prev_orig = rescale_points(kps_prev, (256, 256), (W_orig, H_orig))
            continue

        input1 = transform(frame_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_curr = model(input1)[0].cpu()
        desc_curr = sample_descriptors(feat_curr, kps_curr)

        mutual_matches = match_descriptors(desc_prev, desc_curr, threshold=match_threshold)
        kps_curr_orig = rescale_points(kps_curr, (256, 256), (W_orig, H_orig))

        valid_lines = []
        for idx1, idx2 in mutual_matches:
            x1, y1 = kps_prev_orig[idx1]
            x2, y2 = kps_curr_orig[idx2]
            dist = np.hypot(x2 - x1, y2 - y1)
            if min_movement < dist < max_movement:
                match_history.append(((x1, y1), (x2, y2)))
                valid_lines.append(((x1, y1), (x2, y2)))

        annotated = draw_all_matches(frame_np, match_history)
        annotated_frames.append(annotated)

        kps_prev = kps_curr
        desc_prev = desc_curr
        kps_prev_orig = kps_curr_orig

    out_clip = ImageSequenceClip(annotated_frames, fps=fps)
    out_clip.write_videofile(output_path, codec="libx264", audio=False)
    print(f"视频已保存至：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="逐帧特征点匹配，可视化连线并保存所有轨迹")
    parser.add_argument("--video_path", required=True, help="输入视频路径")
    parser.add_argument("--model_path", required=True, help="训练好的模型 .pth 文件")
    parser.add_argument("--output_path", required=True, help="输出标注视频路径")
    parser.add_argument("--num_keypoints", type=int, default=100, help="每帧检测的最大特征点数量")
    parser.add_argument("--match_threshold", type=float, default=0.8, help="余弦相似度匹配阈值")
    parser.add_argument("--min_movement", type=float, default=3.0, help="最小移动距离才连线")
    parser.add_argument("--max_movement", type=float, default=50.0, help="最大允许连线的距离")
    args = parser.parse_args()

    track_video(
        args.video_path,
        args.model_path,
        args.output_path,
        num_keypoints=args.num_keypoints,
        match_threshold=args.match_threshold,
        min_movement=args.min_movement,
        max_movement=args.max_movement,
    )
