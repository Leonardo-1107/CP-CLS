# for testing the model performance on test set, here is SMILE training data (~1912 CTs)
import os
import h5py
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.nn.functional import softmax


from tqdm import tqdm
from monai.data import Dataset
from monai.transforms import (
    Compose, Resized, ToTensord
)

from monai.utils import set_determinism
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from monai.networks.nets import DenseNet121
from torch.utils.data import Sampler

from dataset.dataset import CLSInferenceDataset, label_mapping
from collections import OrderedDict

import matplotlib.pyplot as plt

from joblib import Parallel, delayed


id_to_phase = {v: k for k, v in label_mapping.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_densenet(checkpoint_path, num_classes, in_chans, device):
    """
    Load the model, as light-weight DenseNet121 from MONAI

    DO NOT CHANGE THE MODEL SETTING, as its heavily personlized
    """
    model = DenseNet121(
        spatial_dims=2,
        in_channels=in_chans,
        out_channels=num_classes,
        pretrained=False,
        norm="batch",
        growth_rate=8,
        block_config=(6, 12, 24, 16),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    new_state = OrderedDict()
    for k, v in state_dict.items():
        new_state[k.replace("module.", "")] = v
    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_scan(model, loader, device):
    preds = []
    for batch in loader:
        imgs = batch["image"].to(device)  # (B,3,H,W)
        out = model(imgs)
        pred = torch.argmax(out, dim=1)
        preds.extend(pred.cpu().numpy())
    # Majority vote
    voted = int(np.bincount(preds).argmax())
    return voted, np.array(preds)



def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="validate")
    parser.add_argument("--data_dir", type=str, help="Folder path storing all ct scans")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--slice_num", type=int, default=8)
    parser.add_argument("--csv_reference_path", type=str)
    parser.add_argument("--test_mode", action='store_true')
    parser.add_argument("--multi_gpu", action='store_true')
    args = parser.parse_args()


    
    model = load_densenet(args.checkpoint_path, args.num_classes, 3, device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference")
        model = torch.nn.DataParallel(model)


    scan_paths = sorted(
        [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".nii.gz")]
    )
    phase_to_id = label_mapping
    id_to_phase = {v: k for k, v in label_mapping.items()}
    
    if args.test_mode:
        #NOTE: test mode, labels required
        """
        Ground truth label required

        ONLY for testing.

        Data format:
        path/to/datafolder
            -- patient_id.nii.gz
            -- ...
        
        """
        all_true, all_pred = [], []
        for path in tqdm(scan_paths[:], desc="Test CT scans phases...."):

            # extract phase from filename, e.g. patient_arterial.nii.gz
            fname = os.path.basename(path)
            phase_name = fname.split("_")[-1].replace(".nii.gz", "")
            if 'venous' in phase_name:
                phase_name = 'venous'
            if 'non' in phase_name:
                phase_name = 'non-contrast'

            y_true = phase_to_id.get(phase_name.lower(), -1)
            if y_true == -1:
                print(f"Skipping {fname}: phase not recognized.")
                continue

            ds = CLSInferenceDataset(path, slice_num=args.slice_num)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

            y_pred, _ = predict_scan(model, dl, device)
            all_true.append(y_true)
            all_pred.append(y_pred)

        cm = confusion_matrix(all_true, all_pred)
        acc = accuracy_score(all_true, all_pred)
        report = classification_report(all_true, all_pred, target_names=list(phase_to_id.keys()), digits=3)

        print("\n=== Classification Results ===")
        print("Accuracy:", round(acc, 4))
        print(report)

        plot_confusion_matrix(cm, list(phase_to_id.keys()))
    
    else:
        # NOTE Inference mode, csv in-direct mode
        """
        Given a csv file contraining all the patients need to be classified
        
        Source data format requirement:
            path/to/ct/folder
                -- BDMAP_ID or patiend_id
                    -- ct.nii.gz
        """
        if args.csv_reference_path:
            df = pd.read_csv(args.csv_reference_path)
            pred_dict = {}
            for bid in tqdm(df["bdmap_id"], desc="CSV inference..."):
                ct_path = os.path.join(args.data_dir, bid, "ct.nii.gz")
                if not os.path.exists(ct_path):
                    print(f"Missing: {ct_path}")
                    continue
                ds = CLSInferenceDataset(ct_path, slice_num=args.slice_num)
                dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
                y_pred, _ = predict_scan(model, dl, device)
                phase_pred = id_to_phase[y_pred]
                pred_dict[bid] = phase_pred

            save_path = args.csv_reference_path.replace('.csv', '_phase_pred.csv')
            pd.DataFrame({"bdmap_id": list(pred_dict.keys()), "Phase Label": list(pred_dict.values())}).to_csv(save_path, index=False)
            exit()

        # NOTE Inference mode, patiend_id.nii.gz file direct mode
        """
        Directly applied to data folder
        
        path/to/datafolder
            -- patient_id.nii.gz
            -- ...
        """
        pred_dict = {}
        for path in tqdm(scan_paths[:], desc="Inference CT scans phases...."):

            # extract phase from filename, e.g. patient_arterial.nii.gz
            fname = os.path.basename(path)
            patient_name = fname.replace(".nii.gz", "")
             
            ds = CLSInferenceDataset(path, slice_num=args.slice_num)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

            y_pred, _ = predict_scan(model, dl, device)
            phase_pred = id_to_phase[y_pred]

            pred_dict[patient_name] = phase_pred
        
        pred_df = pd.DataFrame({"Patient Name": pred_dict.keys(), "Phase Label": pred_dict.values()})
        pred_df.to_csv("inference.csv")