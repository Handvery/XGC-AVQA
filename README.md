# XGC-AVQA

Official dataset and implementation of **XGC-AVQA: A Mixed-Source Benchmark and Semantic-Prior-Based Modeling for No-Reference Audio-Visual Quality Assessment**.

Chenyang Zhang, Kaixuan Tian, Yiping Duan, Xiaoming Tao, and Chang Wen Chen

**[Dataset](xxx) | [Code](https://github.com/Handvery/XGC-AVQA)**

## Overview

XGC-AVQA is a mixed-source benchmark for no-reference audio-visual quality assessment, covering professionally generated content (PGC), user-generated content (UGC), and AI-generated content (AIGC). We also propose **XGCAVNet**, which combines pretrained semantic representations with distortion-sensitive perceptual residuals to predict overall audio-visual quality.

## Dataset

| Source | Reference sequences | Distorted sequences |
| :--- | ---: | ---: |
| PGC | 25 | 300 |
| UGC | 35 | 420 |
| AIGC | 25 | 300 |
| **Total** | **85** | **1,020** |

- **Duration and resolution:** 6 seconds per clip; 1280 x 720 to 1920 x 1080.
- **Video compression:** HEVC with CRF values of 20, 32, 37, and 47.
- **Audio compression:** AAC at 128, 32, and 16 kbps.
- **Distortion combinations:** 4 video levels x 3 audio levels per reference.
- **Annotations:** Mean opinion scores (MOS) for audio, video, and overall audio-visual quality, collected from 20 participants.

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

The current loaders expect a headerless `label.csv`, with filenames in the first column and overall MOS in the fourth column. Keep the 12 distorted versions of each reference consecutive in the CSV for content-level splitting with `group_size=12`.

The default entry point of `fine_tune.py` prints a model summary. To launch Stage 2 training, call its training function directly after configuring the paths and device:

```bash
python -c "from fine_tune import main; main(DATASET='XGC-AVQA', group_size=12)" \
  --pretrained_model_path /path/to/stage1/checkpoints
```

## Results

Within-dataset results reported in the paper, averaged over 10 content-level 80/20 train/test splits:

| Dataset | SRCC | PLCC | KRCC |
| :--- | ---: | ---: | ---: |
| XGC-AVQA | 0.9495 | 0.9527 | 0.8084 |
| LIVE-SJTU | 0.9641 | 0.9684 | 0.8459 |
| UnB-AVQ | 0.8923 | 0.9048 | 0.7849 |

Cross-dataset evaluation, training on XGC-AVQA:

| Test dataset | SRCC | PLCC | KRCC |
| :--- | ---: | ---: | ---: |
| LIVE-SJTU | 0.8633 | 0.8714 | 0.6830 |
| UnB-AVQ | 0.7950 | 0.8055 | 0.6077 |

## Citation

If you use XGC-AVQA or XGCAVNet, please cite our work:

```bibtex
@misc{zhang2026xgcavqa,
  title  = {{XGC-AVQA}: A Mixed-Source Benchmark and Semantic-Prior-Based Modeling for No-Reference Audio-Visual Quality Assessment},
  author = {Zhang, Chenyang and Tian, Kaixuan and Duan, Yiping and Tao, Xiaoming and Chen, Chang Wen},
  year   = {2026},
  url    = {https://github.com/Handvery/XGC-AVQA}
}
```

## Contact

For questions, please open a [GitHub issue](https://github.com/Handvery/XGC-AVQA/issues) or contact [Chenyang Zhang](mailto:z-cy21@mails.tsinghua.edu.cn).
