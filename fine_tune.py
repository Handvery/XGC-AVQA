from argparse import ArgumentParser
import os
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import random
from tensorboardX import SummaryWriter
import datetime
import librosa
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
import scipy.io
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

class VQADataset(Dataset):
    """
    同时返回：
      - 原始视频帧张量 (T, 3, 224, 224) – 从离线文件读取
      - 音频频谱图张量 (64, 64) – 从离线文件读取
      - CLIP 特征 (T, 1024)
      - CLAP 特征 (768,)
    """
    def __init__(self,
                 v_feature_dir,        # 预提取 CLIP 特征目录
                 a_feature_dir,        # 预提取 CLAP 特征目录
                 raw_data_dir,
                 video_names,
                 scores,
                 scale=1,
                 video_frames=30):
        super().__init__()
        self.v_feature_dir = v_feature_dir
        self.a_feature_dir = a_feature_dir
        self.video_frames = video_frames
        self.raw_data_dir = raw_data_dir


        # 预加载 CLIP 和 CLAP 特征
        self.v_clip  = np.zeros((len(video_names), video_frames, 1024))
        self.a_clap  = np.zeros((len(video_names), 768))
        self.mos    = np.zeros((len(video_names), 1))
        self.video_raw_paths = []
        self.audio_raw_paths = []

        for i, vname in enumerate(video_names):
            basename = vname.split('.')[0]

            # parts = basename.split('-')
            # if len(parts) == 3:
            #     A, B, C = parts
            #     # 判断B是否以S结尾
            #     if B.endswith('S'):
            #         vbasename = f"{A}_{B[:-1]}_S"
            #     else:
            #         vbasename = f"{A}_{B}"
            #     abasename = f"{A}_{C}"
            # else:
            #     raise ValueError(f"视频文件名格式不正确: {basename}")

            vbasename = basename
            abasename = basename
            clip_path = os.path.join(v_feature_dir, f"{vbasename}_CLIP.npy")
            clap_path = os.path.join(a_feature_dir, f"{abasename}_CLAP.npy")
            video_raw_path = os.path.join(raw_data_dir, f"{basename}_video_frames.npy")
            audio_raw_path = os.path.join(raw_data_dir, f"{basename}_audio_spec.npy")

            self.video_raw_paths.append(video_raw_path)
            self.audio_raw_paths.append(audio_raw_path)

            # 加载预提取特征
            v_feat = np.load(clip_path)
            a_feat = np.load(clap_path)
            if np.isnan(v_feat).any():
                v_feat = np.nan_to_num(v_feat, nan=0.0)
            if np.isnan(a_feat).any():
                a_feat = np.nan_to_num(a_feat, nan=0.0)

            # 截断/填充到固定帧数
            if v_feat.shape[0] < video_frames:
                pad = np.zeros((video_frames - v_feat.shape[0], v_feat.shape[1]))
                v_feat = np.vstack([v_feat, pad])
            else:
                v_feat = v_feat[:video_frames, :1024]
            self.v_clip[i] = v_feat
            self.a_clap[i] = a_feat

            self.mos[i] = scores[i]

        self.scale = scale
        self.label = self.mos / self.scale

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        v_clip = torch.from_numpy(self.v_clip[idx])
        a_clap = torch.from_numpy(self.a_clap[idx])
        label = torch.from_numpy(self.label[idx])
        v_raw = torch.from_numpy(np.load(self.video_raw_paths[idx]))
        a_spec = torch.from_numpy(np.load(self.audio_raw_paths[idx]))
        return v_raw, a_spec, v_clip, a_clap, label

# ---------- 模型部分：forward 签名接收四种数据，当前仅用预提取特征 ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x)

def norm2d(channels):
    groups = max(1, channels // 8)
    return nn.GroupNorm(groups, channels)

class InvertedResBlock(nn.Module):
    """MobileNetV2 倒残差块 + SE"""
    def __init__(self, inp, oup, stride=1, expand_ratio=2, reduction=4):
        super().__init__()
        hidden_dim = round(inp * expand_ratio)
        self.use_res = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Conv2d(inp, hidden_dim, 1, bias=False)
            )
            layers.append(
                norm2d(hidden_dim)
            )
            layers.append(
                nn.ReLU6(inplace=True)
            )
        layers.extend([
            conv3x3(hidden_dim, hidden_dim, stride, groups=hidden_dim),  # depthwise
            norm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # SE
            SEBlock(hidden_dim, reduction=reduction),
            # pointwise
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            norm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        y = x
        # forward through layer by layer
        y = self.conv(y)
        if self.use_res:
            y = x + y
        return y

class ResidualAdapter(nn.Module):
    def __init__(self, video_out_dim=1024, audio_out_dim=768):
        super().__init__()

        # 输入投影层（因为视频3通道，音频1通道）
        self.v_input_proj = nn.Conv2d(3, 16, 1, bias=False)
        self.a_input_proj = nn.Conv2d(4, 16, 1, bias=False)

        # 视觉流
        self.v_blocks = nn.Sequential(
            InvertedResBlock(16, 32, stride=2, expand_ratio=2),
            InvertedResBlock(32, 64, stride=2, expand_ratio=2),
            InvertedResBlock(64, 128, stride=2, expand_ratio=2),
            nn.AdaptiveAvgPool2d(1)
        )
        # 音频流
        self.a_blocks = nn.Sequential(
            InvertedResBlock(16, 32, stride=2, expand_ratio=2),
            InvertedResBlock(32, 64, stride=2, expand_ratio=2),
            InvertedResBlock(64, 128, stride=2, expand_ratio=2),
            nn.AdaptiveAvgPool2d(1)
        )

        # 特征投影到 128 维
        self.v_proj = nn.Linear(128, video_out_dim)
        self.a_proj = nn.Linear(128, audio_out_dim)

    def forward(self, v_raw, a_spec):
        B, T = v_raw.shape[0], v_raw.shape[1]
        # 视觉分支
        v_in = v_raw.view(B*T, *v_raw.shape[2:])           # (B*T, 3, H, W)

        v_in = self.v_input_proj(v_in)
        v_feat = self.v_blocks(v_in).flatten(1)          # (B*T, 128)
        v_res = self.v_proj(v_feat).view(B, T, -1)               # (B, T, 1024)

        # 音频分支不变
        a_in = self.a_input_proj(a_spec)
        a_feat = self.a_blocks(a_in).flatten(1)
        a_res = self.a_proj(a_feat)

        return v_res, a_res

class VSFA(torch.nn.Module):
    """
    训练时使用的模型：视觉/听觉物理‑语义融合 + 交叉注意力融合，输出质量分数
    """
    def __init__(self,
                 d_model=256,
                 nhead=4,
                 num_layers=2,
                 dropout=0.4,
                 video_dim=1024,
                 audio_dim=768):
        super().__init__()
        self.d_model = d_model
        self.adapter = ResidualAdapter()

        # 视觉分支
        self.v_proj = torch.nn.Linear(video_dim, d_model)
        self.v_pos_enc   = PositionalEncoding(d_model, max_len=500)
        self.v_transformer = torch.nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward=d_model*1, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.v_out_ln = torch.nn.LayerNorm(d_model)

        # 音频分支
        self.a_proj = torch.nn.Linear(audio_dim, d_model)
        self.a_out_ln = torch.nn.LayerNorm(d_model)

        # 回归头
        self.reg_fc1 = torch.nn.Linear(2 * d_model, 2 * d_model)
        self.reg_drop = torch.nn.Dropout(dropout)
        self.reg_relu = torch.nn.ReLU()
        self.reg_fc2 = torch.nn.Linear(2 * d_model, d_model)
        self.reg_fc3 = torch.nn.Linear(d_model, 1)

    def forward(self, v_raw, a_spec, v_input,  a_input):
        v_res, a_res = self.adapter(v_raw, a_spec)
        v_input = v_input + v_res
        a_input = a_input + a_res

        # 视觉处理
        v_feat = self.v_proj(v_input)
        v_seq = self.v_pos_enc(v_feat)
        for layer in self.v_transformer:
            v_seq = layer(v_seq)

        v_fused = torch.mean(v_seq, dim=1)
            
        v_fused = self.v_out_ln(v_fused)

        # 音频处理
        a_fused = self.a_proj(a_input)
        a_fused = self.a_out_ln(a_fused)

        combined = torch.cat([v_fused, a_fused], dim=1)
        out = self.reg_fc1(combined)
        out = self.reg_drop(out)
        out = self.reg_relu(out)
        out = self.reg_fc2(out)
        out = self.reg_drop(out)
        out = self.reg_relu(out)
        out = self.reg_fc3(out)
        return out


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def compute_metrics(y_pred, y):
    '''
    compute metrics btw predictions & labels
    '''
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    # logistic regression btw y_pred & y
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)

    # compute  PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return [SRCC, KRCC, PLCC, RMSE]

def ranking_loss(preds, targets, margin=0.1, threshold=2.0):
    """
    向量化计算成对边际排序损失
    preds: (B, 1) 或 (B,)
    targets: (B, 1) 或 (B,)
    threshold: 当两样本 MOS 差的绝对值 > threshold 时才参与计算，避免标注噪声影响
    """
    # 展平为 1D
    preds = preds.view(-1)
    targets = targets.view(-1)
    B = preds.size(0)

    # 构造所有对的差异矩阵
    pred_diff = preds.unsqueeze(0) - preds.unsqueeze(1)   # (B, B)
    target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)  # (B, B)

    # 只保留一对样本中第一个索引小于第二个的对（避免重复）
    mask = torch.triu(torch.ones(B, B, device=preds.device), diagonal=1).bool()
    # 同时要求 MOS 之差超过阈值
    valid = torch.abs(target_diff) > threshold
    mask = mask & valid

    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)

    # 希望 pred 的顺序与 target 一致：即 target_diff > 0 时，pred_diff 也应 > 0
    # Margin Ranking Loss 形式：max(0, margin - sign(target_diff) * pred_diff)
    sign = torch.sign(target_diff)
    loss_mat = torch.clamp(margin - sign * pred_diff, min=0.0)
    loss = loss_mat[mask].mean()
    return loss

def main(DATASET, group_size=12):
    parser = ArgumentParser(description='"XGC-AVQA')
    parser.add_argument("--seed", type=int, default=19920517) 
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.000001)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--model', default='GeneralAVQA-', type=str,
                        help='model name (default: GeneralAVQA)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0,
                        help='test ratio (default: 0)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='val ratio (default: 0.2)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument('--raw_data_dir', type=str, default=f'/home/data/tkx/Datasets/{DATASET}/preprocess',
                        help='path save video frames')

    parser.add_argument('--v_feature_dir', type=str, default=f'/home/data/tkx/Datasets/{DATASET}/features/CLIP',
                        help='path save video semantic features')
    parser.add_argument('--a_feature_dir', type=str, default=f'/home/data/tkx/Datasets/{DATASET}/features/CLAP',
                        help='path save audio semantic features')

    parser.add_argument('--trained_model_path', type=str, default=f'/home/data/tkx/method/fine_tune_{DATASET}',
                        help='path to save model checkpoint')
    
    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="/home/data/tkx/method/logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    parser.add_argument('--csv_path', type=str, default=f'/home/data/tkx/Datasets/{DATASET}/label.csv',
                        help='path to label.csv (first column: video name, second column: MOS)')
    parser.add_argument('--pretrained_model_path', type=str, default=f'/home/data/tkx/method/weights/base_{DATASET}',
    # MSAV:3 0.9586 SJTU-UAV:8 LIVE-SJTU:5 0.9778
                    help='path to script1 saved base model for fine-tuning')
    args = parser.parse_args()


    trained_model_path = args.trained_model_path
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)


    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    v_feature_dir = args.v_feature_dir
    a_feature_dir = args.a_feature_dir
    raw_data_dir = args.raw_data_dir
    df = pd.read_csv(args.csv_path, header=None)          # 假设 CSV 第一列是视频名，第四列是 MOS
    video_names = df[0].tolist()
    Mos = df[3].tolist()                                  # 根据实际列位置调整

    print('EXP ID: {}'.format(args.exp_id))
    print(args.model)
    print(f'group_size: {group_size}')

    time_str = datetime.datetime.now().strftime("%I%M%B%d")

    if not args.disable_visualization:  # Tensorboard Visualization
        writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}'
                               .format(args.log_dir, args.exp_id, args.model,
                                       args.lr, args.batch_size, args.epochs,time_str))

    device = torch.device("cuda:7" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    all_SRCC = 0
    all_PLCC = 0
    for exepoch in range(10):
        trained_model_file = trained_model_path + '/' + str(exepoch)
        index = [i for i in range(len(video_names))]
        random.shuffle(index)
        train_index, val_index, test_index = [], [], []
        train_cost, val_cost, test_cost = [], [], []

        # 每12个视频为一个同源组，必须划分在同一集合
        num_videos = len(video_names)
        
        # 确保总视频数是12的倍数（可选，根据你的实际情况）
        if num_videos % group_size != 0:
            print(f"Warning: Total videos {num_videos} is not divisible by group size {group_size}.")

        num_groups = num_videos // group_size
        
        # 1. 生成组索引并打乱（注意：这里打乱的是组，而不是单个样本）
        group_indices = list(range(num_groups))
        random.shuffle(group_indices)
        
        # 2. 计算各集合需要的组数量
        test_group_len = round(args.test_ratio * num_groups)
        val_group_len = round(args.val_ratio * num_groups)
        print(f"val:{args.val_ratio} * {num_groups} = {val_group_len}")
        # 剩余的给训练集
        
        # 3. 切分组索引
        test_groups = group_indices[:test_group_len]
        val_groups = group_indices[test_group_len : test_group_len + val_group_len]
        train_groups = group_indices[test_group_len + val_group_len:]
        
        # 4. 定义辅助函数：将组索引展开为具体的视频样本索引
        def expand_group_indices(group_list):
            sample_indices = []
            for g in group_list:
                start_idx = g * group_size
                end_idx = start_idx + group_size
                # 防止越界（如果总数不是12的倍数）
                end_idx = min(end_idx, num_videos)
                sample_indices.extend(range(start_idx, end_idx))
            return sample_indices
        
        # 5. 获取最终的样本索引
        train_indices = expand_group_indices(train_groups)
        val_indices = expand_group_indices(val_groups)
        test_indices = expand_group_indices(test_groups)
        # 根据索引构建各集合的视频名与MOS列表
        train_index = [video_names[i] for i in train_indices]
        train_cost = [Mos[i] for i in train_indices]
        val_index = [video_names[i] for i in val_indices]
        val_cost = [Mos[i] for i in val_indices]
        test_index = [video_names[i] for i in test_indices]
        test_cost = [Mos[i] for i in test_indices]
        # 可选：打印各集合大小以验证
        print(f"Train size: {len(train_index)}, Val size: {len(val_index)}, Test size: {len(test_index)}")

        
        scale = max(train_cost) # label normalization factor
        print(f"Scale: {scale}")
        train_dataset = VQADataset(
            v_feature_dir=v_feature_dir,
            a_feature_dir=a_feature_dir,
            raw_data_dir=raw_data_dir,
            video_names=train_index,
            scores=train_cost,
            scale=scale,
            video_frames=30
        )
        val_dataset   = VQADataset(
            v_feature_dir=v_feature_dir,
            a_feature_dir=a_feature_dir,
            raw_data_dir=raw_data_dir,
            video_names=val_index,
            scores=val_cost,
            scale=scale,
            video_frames=30
        )
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, pin_memory=False, num_workers=8)

        if args.test_ratio > 0:
            test_dataset  = VQADataset(
                v_feature_dir=v_feature_dir,
                a_feature_dir=a_feature_dir,
                raw_data_dir=raw_data_dir,
                video_names=test_index,
                scores=test_cost,
                scale=scale,
                video_frames=30
            )
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, pin_memory=False, num_workers=8)
        else:
            test_loader = val_loader

        model = VSFA(d_model=512, nhead=8, num_layers=1, dropout=0.3).to(device)

        if args.pretrained_model_path is not None:
            pretrained_model_path = args.pretrained_model_path + '/' + str(exepoch)
            print(f"Loading base weights from {pretrained_model_path}")
            pretrained_dict = torch.load(pretrained_model_path, map_location=device)
            # 使用 strict=False 允许 adapter 等新增层缺失
            missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
            print(f"Missing keys (will be randomly initialized): {missing_keys}")
            print(f"Unexpected keys (ignored): {unexpected_keys}")
        else:
            print("No pretrained weights provided, training from scratch.")

        criterion = nn.MSELoss() 
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        best_val_criterion = 100 

        # --- 初始化早停机制相关变量 ---
        early_stop_patience = 10  # 连续10个epoch不提升则停止
        early_stop_counter = 0     # 早停计数器

        for epoch in range(args.epochs):
            # Train
            model.train()
            L = 0
            for i, (v_raw, a_spec, v_clip, a_clap, label) in enumerate(train_loader):
                v_raw = v_raw.to(device).float()
                a_spec = a_spec.to(device).float()
                v_clip = v_clip.to(device).float()
                a_clap = a_clap.to(device).float()
                label = label.to(device).float()

                optimizer.zero_grad()  
                outputs = model(v_raw, a_spec, v_clip, a_clap)
                loss_mse = criterion(outputs, label)
                loss_rank = ranking_loss(outputs, label, margin=0.01, threshold=0.01)
                lambda_rank = 0.5   # 排序损失权重，可调节（设为超参）
                loss = loss_mse + lambda_rank * loss_rank
                loss.backward()
                optimizer.step()
                L = L + loss.item()
            train_loss = L / (i + 1)
            print('{}-{}:train loss {}'.format(exepoch, epoch, train_loss))

            model.eval()
            # Val
            y_pred = np.zeros(len(val_index))
            y_val = np.zeros(len(val_index))
            L = 0
            with torch.no_grad():
                for i, (v_raw, a_spec, v_clip, a_clap, label) in enumerate(val_loader):
                    # print("\r valid_index:{}".format(i), end='')
                    y_val[i] = scale * label.item()  #
                    v_raw = v_raw.to(device).float()
                    a_spec = a_spec.to(device).float()
                    v_clip = v_clip.to(device).float()
                    a_clap = a_clap.to(device).float()
                    label = label.to(device).float()

                    outputs = model(v_raw, a_spec, v_clip, a_clap)
                    y_pred[i] = scale * outputs.item()
                    loss = criterion(outputs, label)
                    L = L + loss.item()
            val_loss = L / (i + 1)
            [val_SROCC, val_KROCC, val_PLCC, val_RMSE] = compute_metrics(y_pred, y_val)
            scheduler.step(val_RMSE)

            # Test
            y_pred = np.zeros(len(test_index)) if args.test_ratio > 0 else np.zeros(len(val_index))
            y_test = np.zeros(len(test_index)) if args.test_ratio > 0 else np.zeros(len(val_index))
            L = 0
            with torch.no_grad():
                for i, (v_raw, a_spec, v_clip, a_clap, label) in enumerate(test_loader):
                    # print("\r test_index:{}".format(i), end='')
                    y_test[i] = scale * label.item()  #
                    v_raw = v_raw.to(device).float()
                    a_spec = a_spec.to(device).float()
                    v_clip = v_clip.to(device).float()
                    a_clap = a_clap.to(device).float()
                    label = label.to(device).float()
                    outputs = model(v_raw, a_spec, v_clip, a_clap)
                    y_pred[i] = scale * outputs.item()
                    loss = criterion(outputs, label)
                    L = L + loss.item()
                test_loss = L / (i + 1)
                [SROCC, KROCC, PLCC, RMSE] = compute_metrics(y_pred, y_test)

            if not args.disable_visualization:  # record training curves
                writer.add_scalar("loss/train-{}".format(exepoch), train_loss, epoch)  
                writer.add_scalar("loss/val-{}".format(exepoch), val_loss, epoch)  
                writer.add_scalar("SROCC/val-{}".format(exepoch), val_SROCC, epoch)  
                writer.add_scalar("KROCC/val-{}".format(exepoch), val_KROCC, epoch)  
                writer.add_scalar("PLCC/val-{}".format(exepoch), val_PLCC, epoch)  
                writer.add_scalar("RMSE/val-{}".format(exepoch), val_RMSE, epoch)  
                writer.add_scalar("loss/test-{}".format(exepoch), test_loss, epoch)  
                writer.add_scalar("SROCC/test-{}".format(exepoch), SROCC, epoch)  
                writer.add_scalar("KROCC/test-{}".format(exepoch), KROCC, epoch)  
                writer.add_scalar("PLCC/test-{}".format(exepoch), PLCC, epoch)  
                writer.add_scalar("RMSE/test-{}".format(exepoch), RMSE, epoch)  

            print("Val results: {}-{}, val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(exepoch, epoch, val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            
            # Update the model with the best val_RMSE
            if val_RMSE < best_val_criterion:
                torch.save(model.state_dict(), trained_model_file)
                print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
                best_val_criterion = val_RMSE  
                best_SRCC = SROCC
                best_PLCC = PLCC
                print("Tes results: {}-{}, test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                    .format(exepoch, epoch, test_loss, SROCC, KROCC, PLCC, RMSE))
                early_stop_counter = 0
            else:
                 # 验证集指标未提升，计数器+1
                early_stop_counter += 1
                # 检查是否触发早停
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}. "
                          f"No improvement for {early_stop_patience} consecutive epochs.")
                    break  # 跳出当前epoch循环，进入下一轮exepoch

        all_SRCC += best_SRCC
        all_PLCC += best_PLCC
    
    print(all_SRCC/10)
    print(all_PLCC/10)

if __name__ == "__main__":
    # main(DATASET='LIVE-SJTU', group_size=24)

    from torchinfo import summary

    device = torch.device("cuda:7" if  torch.cuda.is_available() else "cpu")
    model = VSFA(d_model=512, nhead=8, num_layers=1, dropout=0.3).to(device)

    # 构造一个与训练数据维度一致的假输入
    batch_size = 1
    v_raw = torch.randn(batch_size, 30, 3, 224, 224).to(device)
    a_spec = torch.randn(batch_size, 4, 64, 64).to(device)
    v_clip = torch.randn(batch_size, 30, 1024).to(device)
    a_clap = torch.randn(batch_size, 768).to(device)

    summary(model, input_data=[v_raw, a_spec, v_clip, a_clap])