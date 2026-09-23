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
import librosa.display
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
import scipy.io
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

class VQADataset(Dataset):
    def __init__(self,
                 video_dir,       # CLIP 特征目录
                 audio_dir,       # CLAP 特征目录
                 video_names,
                 scores,
                 scale=1,
                 video_frames=30):
        super(VQADataset, self).__init__()
        self.video_frames = video_frames

        self.v_input  = np.zeros((len(video_names), video_frames, 1024))
        self.a_input  = np.zeros((len(video_names), 768))
        self.mos    = np.zeros((len(video_names), 1))

        for i, vname in enumerate(video_names):
            basename = vname.split('.')[0]

            # 文件路径
            v_path  = os.path.join(video_dir,  f"{basename}_CLIP.npy")
            a_path  = os.path.join(audio_dir,  f"{basename}_CLAP.npy")

            # 加载并处理 NaN
            def load_and_clean(path, target_frames, dim, name):
                feat = np.load(path)
                if np.isnan(feat).any():
                    print(f"{name} contains NaN for {vname}, replacing with 0")
                    feat = np.nan_to_num(feat, nan=0.0)
                # 截断或填充
                if target_frames is None:
                    return feat
                if feat.shape[0] < target_frames:
                    pad = np.zeros((target_frames - feat.shape[0], feat.shape[1]))
                    feat = np.vstack([feat, pad])
                else:
                    feat = feat[:target_frames, :dim]
                return feat

            self.v_input[i] = load_and_clean(v_path, video_frames, 1024, "v_input")
            self.a_input[i] = load_and_clean(a_path, None, 768, "a_input")
            self.mos[i]    = scores[i]

        self.scale = scale
        self.label = self.mos / self.scale

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        return (self.v_input[idx],
                self.a_input[idx],
                self.label[idx])

class PositionalEncoding(nn.Module):
    """正弦位置编码，适用于 Transformer"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TransformerEncoderBlock(nn.Module):
    """单层 Transformer Encoder，包含自注意力和前馈网络"""
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
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
        # 自注意力子层
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 前馈子层
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

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

    def forward(self, v_input,  a_input):

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
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
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

    parser.add_argument('--video_dir', type=str, default=f'/home/Users/zcy/tkx/DB/feature/CLIP',
                        help='path save video semantic features')
    parser.add_argument('--audio_dir', type=str, default=f'/home/Users/zcy/knowledge/MSAV_tkx_best_student_single_feature_root/CLAP',
                        help='path save audio semantic features')

    parser.add_argument('--trained_model_path', type=str, default=f'/home/Users/zcy/tkx/DB/weights',
                        help='path to save model checkpoint')
    
    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="/home/Users/zcy/tkx/DB/logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    parser.add_argument('--csv_path', type=str, default=f'/home/Users/zcy/AVQA-Dataset/MSAV/label.csv',
                        help='path to label.csv (first column: video name, second column: MOS)')
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

    video_dir = args.video_dir
    audio_dir = args.audio_dir
    df = pd.read_csv(args.csv_path, header=None)          # 假设 CSV 第一列是视频名，第四列是 MOS
    video_names = df[0].tolist()
    Mos = df[3].tolist()                                  # 根据实际列位置调整

    print('EXP ID: {}'.format(args.exp_id))
    print(args.model)

    time_str = datetime.datetime.now().strftime("%I%M%B%d")

    if not args.disable_visualization:  # Tensorboard Visualization
        writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}'
                               .format(args.log_dir, args.exp_id, args.model,
                                       args.lr, args.batch_size, args.epochs,time_str))

    device = torch.device("cuda:5" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    print(device)

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
                # 防止越界（如果总数不是group size的倍数）
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
        # print(scale)
        train_dataset = VQADataset(video_dir, audio_dir, train_index, train_cost, scale=scale)
        val_dataset   = VQADataset(video_dir, audio_dir, val_index, val_cost, scale=scale)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, pin_memory=False, num_workers=0)

        if args.test_ratio > 0:
            test_dataset  = VQADataset(video_dir, audio_dir, test_index, test_cost, scale=scale)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, pin_memory=False, num_workers=0)
        else:
            test_loader = val_loader

        model = VSFA(d_model=512, nhead=8, num_layers=1, dropout=0.3).to(device)

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
            for i, (v_input, a_input, label) in enumerate(train_loader):
                v_input = v_input.to(device).float()
                a_input = a_input.to(device).float()
                label  = label.to(device).float()

                optimizer.zero_grad()  
                outputs = model(v_input, a_input)
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
                for i, (v_input, a_input, label) in enumerate(val_loader):
                    # print("\r valid_index:{}".format(i), end='')
                    y_val[i] = scale * label.item()  #
                    v_input = v_input.to(device).float()
                    a_input = a_input.to(device).float()
                    label = label.to(device).float()

                    outputs = model(v_input, a_input)
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
                for i, (v_input, a_input, label) in enumerate(test_loader):
                    # print("\r test_index:{}".format(i), end='')
                    y_test[i] = scale * label.item()  #
                    v_input = v_input.to(device).float()
                    a_input = a_input.to(device).float()
                    label = label.to(device).float()
                    outputs = model(v_input, a_input)
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
    main(DATASET='MSAV',group_size=12)
    # from torchinfo import summary

    # device = torch.device("cuda:4" if  torch.cuda.is_available() else "cpu")
    # model = VSFA(d_model=512, nhead=8, num_layers=1, dropout=0.3).to(device)

    # # 构造一个与训练数据维度一致的假输入
    # batch_size = 16
    # v_raw = torch.randn(batch_size, 30, 3, 224, 224).to(device)
    # a_spec = torch.randn(batch_size, 4, 64, 64).to(device)
    # v_clip = torch.randn(batch_size, 30, 1024).to(device)
    # a_clap = torch.randn(batch_size, 768).to(device)

    # summary(model, input_data=[v_clip, a_clap])