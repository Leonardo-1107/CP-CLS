import os
import torch
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, Resized, ToTensord, MapTransform
)
from monai.networks.nets import DenseNet121
from collections import OrderedDict
import argparse

# -----------------------------------------------------
# Args
# -----------------------------------------------------
parser = argparse.ArgumentParser(description="Validate on H5 CT slices using MONAI pipeline")
parser.add_argument("--data_dir", default="/projects/bodymaps/jliu452/Data/Dataset903_CLS/imageTs", help="Folder path storing the h5 files")
parser.add_argument("--label_csv", default="/projects/bodymaps/jliu452/Data/Dataset903_CLS/imageTs/labels.csv", help="CSV with columns: filename,label")
parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
parser.add_argument("--num_classes", default=4, type=int)
parser.add_argument("--resolution", default=512, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--in_chans", default=3, type=int)
parser.add_argument("--save_dir", default="../logs", help="Output folder for results")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------
# H5 Loader (same as training)
# -----------------------------------------------------
from monai.transforms import MapTransform

class H5Loader(MapTransform):
    def __call__(self, data):
        d = dict(data)
        with h5py.File(d["image"], "r") as hf:
            arr = hf["image"][:]  # (H, W, C) or (H, W)
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        d["image"] = arr.astype(np.float32)  # already [0,1]
        return d


# -----------------------------------------------------
# Dataset + Transform
# -----------------------------------------------------
val_transforms = Compose([
    H5Loader(keys=["image"]),
    Resized(keys=["image"], spatial_size=(args.resolution, args.resolution)),
    ToTensord(keys=["image"])
])

df = pd.read_csv(args.label_csv)
label_to_idx = {label: i for i, label in enumerate(sorted(df['label'].unique()))}
idx_to_label = {v: k for k, v in label_to_idx.items()}

data = []
for _, row in df.iterrows():
    img_path = os.path.join(args.data_dir, row['filename'])
    if not os.path.exists(img_path):
        print(f"Warning: missing file {img_path}")
        continue
    data.append({
        "image": img_path,
        "label": label_to_idx[row['label']],
        "filename": row['filename']  # file name tracker
    })

val_ds = Dataset(data=data, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2, pin_memory=True)

# -----------------------------------------------------
# Model
# -----------------------------------------------------
model = DenseNet121(
    spatial_dims=2,
    in_channels=args.in_chans,
    out_channels=args.num_classes,
    pretrained=False,    # no pretraining for your CT data
    norm="batch",
    growth_rate=8,       # smaller than default (32), reduces params
    block_config=(6, 12, 24, 16),  # standard 121 config
).to(device)
checkpoint = torch.load(args.checkpoint_path, map_location=device)

# allow both "state_dict" and pure model weights
state_dict = checkpoint.get("state_dict", checkpoint)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()


print(f"✅ Loaded model from {args.checkpoint_path}")

# -----------------------------------------------------
# Inference
# -----------------------------------------------------
slice_preds, slice_labels, slice_bdmaps = [], [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Running inference..."):
        imgs = batch["image"].to(device)
        labels = batch["label"].cpu().numpy()

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        # infer BDMAP ID from filename
        filenames = batch["filename"]
        bdmap_ids = [''.join(f.split('_')[:2]) for f in filenames] #form BDMAP_xxx

        slice_preds.extend(preds)
        slice_labels.extend(labels)
        slice_bdmaps.extend(bdmap_ids)

# -----------------------------------------------------
# BDMAP-level major voting
# -----------------------------------------------------
bdmap_pred_dict = defaultdict(list)
bdmap_label_dict = {}
for pred, label, bdm_id in zip(slice_preds, slice_labels, slice_bdmaps):
    bdmap_pred_dict[bdm_id].append(pred)
    bdmap_label_dict[bdm_id] = label

bdmap_preds, bdmap_labels = [], []
for bdm_id, preds in bdmap_pred_dict.items():
    majority = Counter(preds).most_common(1)[0][0]
    bdmap_preds.append(majority)
    bdmap_labels.append(bdmap_label_dict[bdm_id])

# -----------------------------------------------------
# Evaluation: Confusion matrix + per-class accuracy
# -----------------------------------------------------
cm = confusion_matrix(bdmap_labels, bdmap_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[idx_to_label[i] for i in range(args.num_classes)]
)
disp.plot(cmap="Blues", values_format='d')
plt.title("BDMAP-level Confusion Matrix")
plt.savefig(os.path.join(args.save_dir, "./IID_result_confusion.png"))
plt.close()

class_acc = []
for i in range(args.num_classes):
    correct = cm[i, i]
    total = cm[i].sum()
    acc = correct / total if total > 0 else 0.0
    class_acc.append({
        "Class Index": i,
        "Class Name": idx_to_label[i],
        "Correct": correct,
        "Total": total,
        "Accuracy (%)": round(acc * 100, 2)
    })

df_acc = pd.DataFrame(class_acc)
df_acc.to_csv(os.path.join(args.save_dir, "./IID_result_class_accuracy.csv"), index=False)
print(df_acc)

print("Evaluation complete.")