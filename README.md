# XGC-AVQA

Official dataset and code of **XGC-AVQA: A Mixed-Source Benchmark and Semantic-Prior-Based Modeling for No-Reference Audio-Visual Quality Assessment**.

**[Dataset](https://drive.google.com/file/d/13H2mRU7suZzskdBtz-fCDPEAx9vfPx21/view?usp=drive_link) 

## Overview

XGC-AVQA is a mixed-source benchmark for no-reference audio-visual quality assessment, covering professionally generated content (PGC), user-generated content (UGC), and AI-generated content (AIGC). We also propose **XGCAVNet**, which combines pretrained semantic representations with distortion-sensitive perceptual residuals to predict overall audio-visual quality.

## Dataset

| Source | Reference sequences | Distorted sequences |
| :--- | ---: | ---: |
| PGC | 25 | 300 |
| UGC | 35 | 420 |
| AIGC | 25 | 300 |
| **Total** | **85** | **1,020** |

## XGCAVNet

- **Frozen semantic encoders:** CLIP ResNet-50 for video frames and HTSAT-CLAP for audio.
- **Perceptual Residual Calibration Module (PRCM):** Lightweight residual branches that supplement semantic features with distortion-sensitive cues.
- **Quality prediction:** Transformer-based temporal modeling, audio-visual feature fusion, and MLP regression.
- **Two-stage training:** Train the base model on pre-extracted features, then introduce PRCM and fine-tune the downstream network while keeping both pretrained encoders frozen.

The model has **4.398M trainable parameters**, accounting for **6.26%** of its 70.264M total parameters.

## Usage

The experiments used **PyTorch 2.5.1**, **CUDA 12.1**, and an **NVIDIA RTX 3090**. The scripts use NumPy, pandas, SciPy, scikit-learn, librosa, decord, transformers, einops, tqdm, and tensorboardX. CLIP extraction additionally requires the CLIP-IQA loader imported from `mmedit.models.backbones.sr_backbones.coopclipiqa`.

Run the workflow in the following order after configuring the dataset paths, feature directories, checkpoint paths, and CUDA device IDs in the scripts:

| Step | Script | Purpose |
| :--- | :--- | :--- |
| 1 | `data_preprocess.py` | Prepare sampled video frames and audio spectrograms. |
| 2 | `CLIP_feature_extract.py` | Extract visual features using `--feature_type single`. |
| 3 | `CLAP_feature_extract.py` | Extract audio features. |
| 4 | `train_base.py` | Train the base model (Stage 1). |
| 5 | `fine_tune.py` | Add PRCM and fine-tune from Stage 1 checkpoints (Stage 2). |
| 6 | `inference.py` | Evaluate trained checkpoints. |


