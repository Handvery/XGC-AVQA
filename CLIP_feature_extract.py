from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
import os
import pandas as pd
import math
from decord import VideoReader, cpu
from einops import rearrange
from mmedit.models.backbones.sr_backbones.coopclipiqa import load_clip_to_cpu


class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, frame_length, video_format='RGB'):
        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.format = video_format
        self.frame_length = frame_length

        # CLIP 预处理参数
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_path = os.path.join(self.videos_dir, video_name)
        
        # 使用 decord 加载视频
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            raise IOError(f"视频文件为空: {video_path}")

        # 均匀采样 self.frame_length 帧
        if total_frames >= self.frame_length:
            indices = np.linspace(0, total_frames - 1, self.frame_length, dtype=int)
        else:
            indices = np.arange(total_frames)
            pad_len = self.frame_length - total_frames
            indices = np.concatenate([indices, np.full(pad_len, total_frames - 1)])
        
        frames = vr.get_batch(indices).asnumpy()  # [frame_length, H, W, 3]

        # CLIP 标准化预处理
        frames_float = frames.astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_float).permute(0, 3, 1, 2)  # [frame_length, 3, H, W]
        frames_normalized = (frames_tensor - self.clip_mean) / self.clip_std

        sample = {'video': frames_normalized}
        return sample


class BaseMultiScaleExtractor(nn.Module):
    """多尺度特征提取器基类，负责钩子管理"""
    def __init__(self, clip_model_name="RN50", freeze_backbone=True):
        super().__init__()
        self.clip_model = load_clip_to_cpu(clip_model_name)
        self.clip_visual = self.clip_model.visual
        self.dtype = self.clip_model.dtype

        if freeze_backbone:
            for param in self.clip_visual.parameters():
                param.requires_grad = False

        self.intermediate_features = {}
        self.hooks = []

    def _register_hooks(self, layer_names):
        """根据给定的层名注册前向钩子"""
        for name in layer_names:
            if name == 'attnpool':
                continue  # attnpool 在 forward 中手动获取
            layer = self.clip_visual
            for part in name.split('.'):
                layer = getattr(layer, part)
            
            def hook_fn(module, input, output, name=name):
                self.intermediate_features[name] = output
            self.hooks.append(layer.register_forward_hook(hook_fn))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def forward(self, x):
        x = x.type(self.dtype)
        self.intermediate_features = {}
        with torch.set_grad_enabled(not self.clip_visual.training):
            final_feat = self.clip_model.encode_image(x, pos_embedding=False)  # [B, 1024]
            self.intermediate_features['attnpool'] = final_feat
        return self.intermediate_features


class ZeroParamMultiScaleExtractor(BaseMultiScaleExtractor):
    """
    无参多尺度特征融合提取器
    选定层次：layer2 (512) + layer3 (1024) + AttnPool (1024) = 2560 维
    """
    def __init__(self, clip_model_name="RN50", freeze_backbone=True):
        super().__init__(clip_model_name, freeze_backbone)
        # 修改为 layer2 和 layer3
        self.target_layers = ['layer2', 'layer3']
        self._register_hooks(self.target_layers)
        # 输出维度 = 512 + 1024 + 1024 = 2560
        self.output_dim = 512 + 1024 + 1024

    def forward(self, x):
        feat_dict = super().forward(x)
        
        f2 = feat_dict['layer2']        # [B, 512, 28, 28]
        f3 = feat_dict['layer3']        # [B, 1024, 14, 14]
        f_attn = feat_dict['attnpool']  # [B, 1024]
        
        # 全局平均池化
        f2_pooled = f2.mean(dim=[2, 3])   # [B, 512]
        f3_pooled = f3.mean(dim=[2, 3])   # [B, 1024]
        
        # 直接拼接
        fused = torch.cat([f2_pooled, f3_pooled, f_attn], dim=-1)  # [B, 2560]
        
        return fused


class CLIPSemanticExtractor(nn.Module):
    """原始 CLIP 顶层特征提取器（仅输出 1024 维）"""
    def __init__(self, clip_model_name="RN50", freeze_backbone=True):
        super().__init__()
        self.clip_model = load_clip_to_cpu(clip_model_name)
        self.clip_visual = self.clip_model.visual
        self.dtype = self.clip_model.dtype

        if freeze_backbone:
            for param in self.clip_visual.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x.type(self.dtype)
        with torch.set_grad_enabled(not self.clip_visual.training):
            final_global_feature = self.clip_model.encode_image(x, pos_embedding=False)  # [B, 1024]
        return final_global_feature


def get_features(video_data, frame_batch_size=64, extractor=None, device='cuda'):
    """特征提取，适配任意输出维度"""
    video_length = video_data.shape[0]
    output = torch.Tensor().to(device)
    extractor.eval()
    
    with torch.no_grad():
        for start in range(0, video_length, frame_batch_size):
            end = min(start + frame_batch_size, video_length)
            batch = video_data[start:end].to(device)
            features = extractor(batch)
            output = torch.cat((output, features), 0)
    
    return output.squeeze()


def main(DATASET_NAME):
    parser = ArgumentParser(description='Extracting Video Features with CLIP-IQA RN50')
    parser.add_argument("--seed", type=int, default=19901116)
    parser.add_argument('--model', default='CLIP-IQA-RN50', type=str,
                        help='which pre-trained model used (default: CLIP-IQA-RN50)')
    parser.add_argument('--feature_type', type=str, default='single', choices=['single', 'multiscale'],
                        help='single: original 1024-d; multiscale: fused 4096-d (no extra params)')
    parser.add_argument('--frame_batch_size', type=int, default=10,
                        help='frame batch size for feature extraction')
    parser.add_argument('--frame_length', type=int, default=30,
                        help='number of frames to extract feature')
    parser.add_argument('--features_dir', type=str, default=f'/home/data/tkx/Datasets/{DATASET_NAME}/features/CLIP',
                        help='path to save video spatial features')
    parser.add_argument('--videos_dir', type=str, default=f'/home/data/tkx/Datasets/{DATASET_NAME}/video',
                        help='directory containing video files')
    parser.add_argument('--csv_path', type=str, default=f'/home/data/tkx/Datasets/{DATASET_NAME}/label.csv',
                        help='path to label.csv file (first column: video names)')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument("--ith", type=int, default=0, help='start video id (for resuming)')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 读取 CSV
    try:
        df = pd.read_csv(args.csv_path, header=None)
        video_names = df[0].tolist()
        video_names = [name.replace('.yuv', '.mp4') for name in video_names]
        total_files = len(video_names)
        print(f"[Info] 成功读取 {args.csv_path}，共 {total_files} 个视频文件")
    except Exception as e:
        print(f"[Error] 无法读取 CSV 文件: {e}")
        exit(1)

    # 创建特征保存目录
    if not os.path.exists(args.features_dir):
        os.makedirs(args.features_dir)
        print(f"[Info] 创建特征保存目录: {args.features_dir}")

    # 设置设备
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    # 初始化特征提取器
    print(f"[Info] 初始化 CLIP-IQA RN50 模型，特征类型: {args.feature_type}")
    if args.feature_type == 'multiscale':
        extractor = ZeroParamMultiScaleExtractor(clip_model_name="RN50", freeze_backbone=True).to(device)
        suffix = "_CLIP_multiscale.npy"
        print(f"[Info] 多尺度融合特征提取器已初始化，输出维度: {extractor.output_dim}")
    else:
        extractor = CLIPSemanticExtractor(clip_model_name="RN50", freeze_backbone=True).to(device)
        suffix = "_CLIP.npy"
        print("[Info] 单一顶层特征提取器已初始化，输出维度: 1024")

    # 创建数据集
    dataset = VideoDataset(args.videos_dir, video_names, args.frame_length, video_format='RGB')

    # 统计
    processed_count = 0
    skip_count = 0
    error_count = 0

    # 遍历处理
    for i in range(args.ith, len(dataset)):
        v_name = video_names[i]
        base_name = os.path.splitext(v_name)[0]
        npy_path = os.path.join(args.features_dir, f"{base_name}{suffix}")

        # 断点续传
        if os.path.exists(npy_path):
            print(f"[{i+1}/{total_files}] 跳过已处理: {v_name}")
            skip_count += 1
            continue

        # 检查视频文件是否存在
        video_path = os.path.join(args.videos_dir, v_name)
        if not os.path.exists(video_path):
            print(f"[Warning] 视频文件不存在，跳过: {video_path}")
            error_count += 1
            continue

        try:
            current_data = dataset[i]
            print(f"[{i+1}/{total_files}] Video {v_name}: length {current_data['video'].shape[0]}")

            # 提取特征
            features = get_features(current_data['video'], args.frame_batch_size, extractor, device)
            print(f"  特征尺寸: {features.shape}")

            # 保存特征
            np.save(npy_path, features.to('cpu').numpy())
            processed_count += 1
            print(f"  已保存至: {npy_path}")
        except Exception as e:
            print(f"[Error] 处理视频 {v_name} 失败: {e}")
            error_count += 1

    # 完成统计
    print("\n" + "="*40)
    print(f"处理完成统计:")
    print(f"  - 总文件数: {total_files}")
    print(f"  - 新处理成功: {processed_count}")
    print(f"  - 跳过（已存在）: {skip_count}")
    print(f"  - 失败/缺失: {error_count}")
    print(f"特征保存路径: {args.features_dir}")
    print("="*40)

if __name__ == "__main__":
    main(DATASET_NAME="UnB-AVQ")