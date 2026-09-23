# preprocess_raw_data.py
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from decord import VideoReader, cpu
from transformers import ClapFeatureExtractor
import pandas as pd
from tqdm import tqdm

def preprocess_video(video_path, num_frames=30):
    """复制 _load_video_frames 逻辑"""
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise IOError(f"视频文件为空: {video_path}")

    # 均匀采样
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
        pad_len = num_frames - total_frames
        indices = np.concatenate([indices, np.full(pad_len, total_frames - 1)])

    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
    frames_float = frames.astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_float).permute(0, 3, 1, 2)  # (T, 3, H, W)
    frames_normalized = (frames_tensor - clip_mean) / clip_std
    frames_normalized = F.interpolate(frames_normalized, size=(224, 224),
                                      mode='bilinear', align_corners=False)
    return frames_normalized.numpy()  # (T, 3, 224, 224)

def preprocess_audio(audio_path, extractor, target_sr=48000):
    """复制 _load_audio_spec 逻辑"""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    inputs = extractor(
        [audio],
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )
    spec = inputs["input_features"]  # (1, freq, time)
    spec = F.interpolate(spec, size=(64, 64), mode='bilinear', align_corners=False)
    spec = spec.squeeze(0)  # (freq, time)
    return spec.numpy()

def main(Dataset_name):
    parser = argparse.ArgumentParser(description="离线预处理视频帧和音频频谱图")
    parser.add_argument("--video_dir", default=f"/home/data/tkx/Datasets/{Dataset_name}/video", help="视频文件夹")
    parser.add_argument("--audio_dir", default=f"/home/data/tkx/Datasets/{Dataset_name}/audio", help="音频文件夹")
    parser.add_argument("--output_dir", default=f"/home/data/tkx/Datasets/{Dataset_name}/preprocess", help="预处理结果保存文件夹")
    parser.add_argument("--csv_path", default=f"/home/data/tkx/Datasets/{Dataset_name}/label.csv", help="CSV文件，第一列为视频文件名（含扩展名）")
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--target_sr", type=int, default=48000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取视频列表
    df = pd.read_csv(args.csv_path, header=None)
    video_names = df[0].tolist()

    # 初始化音频特征提取器（仅需一次）
    extractor = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-fused")

    for vname in tqdm(video_names):
    #     basename = vname.rsplit('.', 1)[0]
    #     parts = basename.split('-')
    #     if len(parts) == 3:
    #         A, B, C = parts
    #         # 判断B是否以S结尾
    #         if B.endswith('S'):
    #             vbasename = f"{A}_{B[:-1]}_S"
    #         else:
    #             vbasename = f"{A}_{B}"
    #         abasename = f"{A}_{C}"
    #     else:
    #         raise ValueError(f"视频文件名格式不正确: {basename}")
        vbasename = vname.rsplit('.', 1)[0]
        abasename = vname.rsplit('.', 1)[0]
 
        video_path = os.path.join(args.video_dir, f"{vbasename}.mp4")
        audio_path = os.path.join(args.audio_dir, f"{abasename}.wav")

        # 处理视频帧
        frames = preprocess_video(video_path, args.num_frames)
        np.save(os.path.join(args.output_dir, f"{vbasename}_video_frames.npy"), frames)
        print(f"视频帧处理完成: {video_path}")

        # 处理音频频谱图
        spec = preprocess_audio(audio_path, extractor, args.target_sr)
        np.save(os.path.join(args.output_dir, f"{abasename}_audio_spec.npy"), spec)
        print(f"音频频谱图处理完成: {audio_path}")
        
    print("预处理完成！")

if __name__ == "__main__":
    main(Dataset_name="UnB-AVQ")