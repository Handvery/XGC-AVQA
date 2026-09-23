import os
import torch
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from fine_tune import (
    VQADataset,
    VSFA
)

############################################################
# Metrics
############################################################

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(
        np.negative(np.divide(X - bayta3, np.abs(bayta4)))
    )
    yhat = bayta2 + np.divide(
        bayta1 - bayta2,
        logisticPart
    )
    return yhat


def compute_metrics(y_pred, y):

    SRCC = scipy.stats.spearmanr(y, y_pred)[0]

    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(
            y,
            y_pred,
            method='asymptotic'
        )[0]

    beta_init = [
        np.max(y),
        np.min(y),
        np.mean(y_pred),
        0.5
    ]

    popt, _ = curve_fit(
        logistic_func,
        y_pred,
        y,
        p0=beta_init,
        maxfev=int(1e8)
    )

    y_pred_logistic = logistic_func(
        y_pred,
        *popt
    )

    PLCC = scipy.stats.pearsonr(
        y,
        y_pred_logistic
    )[0]

    RMSE = np.sqrt(
        mean_squared_error(
            y,
            y_pred_logistic
        )
    )

    return SRCC, KRCC, PLCC, RMSE


############################################################
# Inference
############################################################

def evaluate_model(
        model_path,
        dataloader,
        device,
        scale):

    model = VSFA(
        d_model=512,
        nhead=8,
        num_layers=1,
        dropout=0.3
    ).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():

        for i, (
            v_raw,
            a_spec,
            v_clip,
            a_clap,
            label
        ) in enumerate(dataloader):
            print(f"one batch start {i}")

            v_raw = v_raw.float().to(device)
            a_spec = a_spec.float().to(device)
            v_clip = v_clip.float().to(device)
            a_clap = a_clap.float().to(device)

            pred = model(
                v_raw,
                a_spec,
                v_clip,
                a_clap
            )

            y_pred.extend(
                (pred.squeeze(1).cpu().numpy() * scale).tolist()
            )

            y_true.extend(
                (label.squeeze(1).cpu().numpy() * scale).tolist()
            )

            print(f"one batch end {i}")

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return compute_metrics(
        y_pred,
        y_true
    )


############################################################
# Main
############################################################

def main():
    MODEL_PATH = "UnB-AVQ"
    DATASET = "MSAV"

    raw_data_dir = f"/home/data/tkx/Datasets/{DATASET}/preprocess"

    v_feature_dir = f"/home/data/tkx/Datasets/{DATASET}/features/CLIP"

    a_feature_dir = f"/home/data/tkx/Datasets/{DATASET}/features/CLAP"

    csv_path = f"/home/data/tkx/Datasets/{DATASET}/label.csv"

    model_dir = f"/home/data/tkx/method/weights/fine_tune_{MODEL_PATH}"

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    ########################################################
    # Full dataset
    ########################################################

    df = pd.read_csv(
        csv_path,
        header=None
    )

    video_names = df[0].tolist()
    mos = df[3].tolist()

    scale = max(mos)

    dataset = VQADataset(
        v_feature_dir=v_feature_dir,
        a_feature_dir=a_feature_dir,
        raw_data_dir=raw_data_dir,
        video_names=video_names,
        scores=mos,
        scale=scale,
        video_frames=30
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    ########################################################
    # 10 models
    ########################################################

    srcc_list = []
    krcc_list = []
    plcc_list = []
    rmse_list = []

    for idx in range(10):

        model_path = os.path.join(
            model_dir,
            str(idx)
        )

        print(f"\nEvaluating model {idx}")

        SRCC, KRCC, PLCC, RMSE = evaluate_model(
            model_path,
            dataloader,
            device,
            scale
        )

        srcc_list.append(SRCC)
        krcc_list.append(KRCC)
        plcc_list.append(PLCC)
        rmse_list.append(RMSE)

        print(
            f"SRCC={SRCC:.4f} "
            f"KRCC={KRCC:.4f} "
            f"PLCC={PLCC:.4f} "
            f"RMSE={RMSE:.4f}"
        )

    ########################################################
    # Statistics
    ########################################################

    print("\n==============================")
    print("Final Results (10 models)")
    print("==============================")

    print(
        f"SRCC: "
        f"{np.mean(srcc_list):.4f} ± "
        f"{np.std(srcc_list):.4f}"
    )

    print(
        f"KRCC: "
        f"{np.mean(krcc_list):.4f} ± "
        f"{np.std(krcc_list):.4f}"
    )

    print(
        f"PLCC: "
        f"{np.mean(plcc_list):.4f} ± "
        f"{np.std(plcc_list):.4f}"
    )

    print(
        f"RMSE: "
        f"{np.mean(rmse_list):.4f} ± "
        f"{np.std(rmse_list):.4f}"
    )


if __name__ == "__main__":
    main()