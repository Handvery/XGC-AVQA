import os
import torch
import librosa
import numpy as np
from pathlib import Path
import pandas as pd
from transformers import ClapModel, ClapFeatureExtractor

# ===================== 配置区域（请修改此处） =====================
DATASET_NAME = "UnB-AVQ"
DATASET_FOLDER = f"/home/data/tkx/Datasets/{DATASET_NAME}/audio"  # 音频文件夹路径
OUTPUT_FOLDER = f"/home/data/tkx/Datasets/{DATASET_NAME}/features/CLAP"  # 特征保存文件夹
LABEL_FILE = f"/home/data/tkx/Datasets/{DATASET_NAME}/label.csv"
MODEL_NAME = "laion/clap-htsat-fused"                        # CLAP模型
DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
SAVE_CHECKPOINT_INTERVAL = 12                                 # 断点保存间隔（当前未启用）
# ===================================================================

def process_single_audio(
    audio_path: str,
    model,
    feature_extractor,
    device: torch.device,
    target_sr: int = 48000,
    temporal_feature = False
) -> np.ndarray:
    """
    处理单个完整音频，返回 CLAP 最后一个隐藏层的全局平均向量 (768,)。
    """
    try:
        # 1. 加载并重采样
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

        # 3. 预处理（单样本批量）
        inputs = feature_extractor(
            [audio],
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True          # 单样本时仅增加 batch 维，无额外填充
        )

        # 4. 移动到设备
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        model = model.to(device)

        # 5. 提取最后一层隐藏状态，并做时间维平均
        with torch.no_grad():
            if temporal_feature:
                outputs = model.audio_model(**inputs)
                # last_hidden_state: (batch=1, time_steps, 768)
                hidden = outputs.last_hidden_state  # shape: [1, 768, 2, 32]
                # 对 2 个频谱（维度2）做展平，后转置为 (32, 768*2)
                pooled = hidden.squeeze(0)  # shape: [768, 2, 32]
                pooled = pooled.permute(2, 0, 1)  # (32, 768, 2)
                # pooled = pooled.reshape(32, 768 * 2)  # (32, 1536)
                pooled = torch.mean(pooled, dim=2)  # (32, 768)
                return pooled.squeeze(0).cpu().numpy()
            else:
                outputs = model.audio_model(**inputs)
                return outputs.pooler_output.squeeze(0).cpu().numpy()
            

    except Exception as e:
        print(f"⚠️  Error processing {audio_path}: {e}")
        raise e


def load_checkpoint(temp_index_path):
    """加载断点进度"""
    if os.path.exists(temp_index_path):
        try:
            with open(temp_index_path, 'r') as f:
                last_idx = int(f.read().strip())
            print(f"🔄 Found checkpoint! Resuming from audio {last_idx + 1}")
            return last_idx
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint, starting from scratch: {e}")
            return 0
    return 0

def save_checkpoint(current_idx, temp_index_path):
    """保存断点（原子写入）"""
    with open(temp_index_path + '.tmp', 'w') as f:
        f.write(str(current_idx))
    os.replace(temp_index_path + '.tmp', temp_index_path)

def clear_checkpoint(temp_index_path):
    if os.path.exists(temp_index_path):
        os.remove(temp_index_path)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"🚀 Using device: {DEVICE}")
    print(f"📁 Dataset: {DATASET_FOLDER}")
    print(f"📄 Label file: {LABEL_FILE}")
    print(f"💾 Output: {OUTPUT_FOLDER}")

    # 读取标签文件第一列
    df = pd.read_csv(LABEL_FILE, header=None, dtype=str)
    file_names = df.iloc[:, 0].tolist()
    file_names = [f.replace('.mp4', '.wav') for f in file_names]

    total = len(file_names)
    print(f"🎯 Total audios (from label file): {total}")

    model = ClapModel.from_pretrained(MODEL_NAME).to(DEVICE)
    feature_extractor = ClapFeatureExtractor.from_pretrained(MODEL_NAME)

    processed, skipped, errors = 0, 0, 0
    for idx, fname in enumerate(file_names, start=1):
        basename = fname.rsplit('.', 1)[0]
        # parts = basename.split('-')
        # if len(parts) == 3:
        #     A, B, C = parts
        #     basename = f"{A}_{C}"
        # else:
        #     raise ValueError(f"视频文件名格式不正确: {basename}")

        audio_path = os.path.join(DATASET_FOLDER, f"{basename}.wav")
        npy_path = os.path.join(OUTPUT_FOLDER, f"{basename}_CLAP.npy")

        if os.path.exists(npy_path):
            skipped += 1
            if idx % 50 == 0:
                print(f"⏭️ Skipped existing: {idx}/{total}")
            continue

        try:
            feat = process_single_audio(audio_path, model, feature_extractor, DEVICE)
            np.save(npy_path, feat.astype(np.float32))
            processed += 1
            if idx % 10 == 0:
                print(f"✅ Processed {idx}/{total} ({fname}), shape={feat.shape}")
        except Exception as e:
            print(f"❌ Error {idx}/{total} ({fname}): {e}")
            errors += 1

    print("\n" + "=" * 40)
    print(f"📊 Summary:")
    print(f"  - Total: {total}")
    print(f"  - Newly processed: {processed}")
    print(f"  - Skipped: {skipped}")
    print(f"  - Errors: {errors}")
    print("=" * 40)

if __name__ == "__main__":
    main()