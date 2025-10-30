import os
import h5py
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler


from tqdm import tqdm
from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, Resized, RandFlipd, RandRotated, RandZoomd, RandGaussianNoised,
    ToTensord
)

from monai.utils import set_determinism
from sklearn.metrics import classification_report

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import timm   # <--- timm backbone, optional
from monai.networks.nets import DenseNet121

import torchvision.models as models # <---- Simpler models

from torch.utils.data import Sampler
import random
from collections import defaultdict



set_determinism(3407)
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


import argparse
parser = argparse.ArgumentParser(description="validate")
parser.add_argument("--data_dir", default="./FullData/imageTs", help="Folder path storing all the images for validation")
parser.add_argument("--num_classes", default=4, type=int, help="Num of CT phases")
parser.add_argument("--save_path", default="./outputs", help="Path to model checkpoints")
parser.add_argument("--resolution", default=512, type=int, help="DO NOT CHANGE")
parser.add_argument("--num_epochs", default=10000, type=int)
parser.add_argument("--validation_interval_step", default=2000, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--resume_from_checkpoint", type=str)
parser.add_argument("--in_chans", default=3, type=int, help="Number of channels (1 for single slice, 3 for 2.5D input)")
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
# -----------------------------------------------------
# Custom H5 Loader
# -----------------------------------------------------
from monai.transforms import MapTransform

class H5Loader(MapTransform):
    def __call__(self, data):
        d = dict(data)
        with h5py.File(d["image"], "r") as hf:
            arr = hf["image"][:]  # (H,W,C) or (H,W)
        if arr.ndim == 2:
            arr = arr[..., None]  # (H,W,1)
        arr = np.transpose(arr, (2,0,1))  # to (C,H,W)
        d["image"] = arr.astype(np.float32)  # already [0,1] normalized
        return d


class PhaseBalancedSampler(Sampler):
    def __init__(self, dataset, batch_size, num_classes):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        self.class_indices = defaultdict(list)
        for idx, item in enumerate(dataset):
            label = item["label"]
            self.class_indices[label].append(idx)

    def __iter__(self):
        
        while True:
            batch = []
            for cls in range(self.num_classes):
                idx = random.choice(self.class_indices[cls])
                batch.append(idx)
            yield batch  # ensure every phase is included

    def __len__(self):
        
        return min(len(v) for v in self.class_indices.values())




train_transforms = Compose([
    H5Loader(keys=["image"]),   # your custom HDF5 loader
    Resized(keys=["image"], spatial_size=(args.resolution, args.resolution)),

    # ---- Data augmentations ----
    RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),           # horizontal flip
    RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),           # vertical flip
    RandRotated(keys=["image"], range_x=np.pi/12, prob=0.2),       # random rotation ±15°
    RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.2), # slight zoom
    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01), # mild noise

    ToTensord(keys=["image"])
])

val_transforms = Compose([
    H5Loader(keys=["image"]),
    Resized(keys=["image"], spatial_size=(args.resolution, args.resolution)),
    ToTensord(keys=["image"])
])

# -----------------------------------------------------
# Dataset
# -----------------------------------------------------
root_dir = args.data_dir
data = []
for label, cls in enumerate(sorted(os.listdir(root_dir))):
    cls_dir = os.path.join(root_dir, cls)
    for f in os.listdir(cls_dir):
        if f.endswith(".h5"):
            data.append({"image": os.path.join(cls_dir, f), "label": label})

full_ds = Dataset(data=data, transform=train_transforms)
val_size = int(0.2 * len(full_ds))
train_size = len(full_ds) - val_size
train_subset, val_subset = random_split(full_ds, [train_size, val_size])

# Wrap each subset with its transform
train_ds = Dataset(data=train_subset.dataset.data, transform=train_transforms)
val_ds   = Dataset(data=val_subset.dataset.data,   transform=val_transforms)




# use user-difined sampler
train_sampler = DistributedSampler(train_ds, shuffle=True,  drop_last=True)
val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, sampler=train_sampler,
    num_workers=4, pin_memory=True, drop_last=True, persistent_workers=False
)
val_loader = DataLoader(
    val_ds, batch_size=args.batch_size, sampler=val_sampler,
    num_workers=4, pin_memory=True, drop_last=True, persistent_workers=False
)

# -----------------------------------------------------
# Model: timm, MONAI or ResNet 18
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

model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False )

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)



# -----------------------------------------------------
# Training loop
# -----------------------------------------------------
num_epochs = int(args.num_epochs)
validate_every = args.validation_interval_step  # iterations
best_acc = 0.0

if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
    checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
    model.module.load_state_dict(checkpoint)
    if dist.get_rank() == 0:
        print(f"Resumed model from {args.resume_from_checkpoint}")

global_step = 0
acc_list = []

for epoch in range(num_epochs):
    model.train()
    train_sampler.set_epoch(epoch)
    val_sampler.set_epoch(epoch) 


    epoch_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Rank {local_rank})",
                unit="batch", disable=(dist.get_rank() != 0))

    for batch in pbar:
        
        inputs = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"]).long().squeeze().to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        try:
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        except:
            print(f"[Rank {dist.get_rank()}] Exception: {e}", flush=True)
            dist.barrier()
            raise

        if dist.get_rank() == 0:
            pbar.set_postfix({"step": global_step, "loss": f"{loss.item():.4f}"})

        # ---- Validation every X steps ----
        if global_step % validate_every == 0:
            # all ranks reach here
            dist.barrier()
            model.eval()

            local_preds, local_labels = [], []

            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device, non_blocking=True)
                    val_labels = torch.as_tensor(val_batch["label"]).long().squeeze().to(device, non_blocking=True)
                    val_outputs = model(val_inputs)
                    preds = torch.argmax(val_outputs, dim=1)
                    
                    local_preds.extend(preds.cpu().tolist())
                    local_labels.extend(val_labels.cpu().tolist())

            # --- Gather results from all ranks ---
            all_preds_list = [None for _ in range(dist.get_world_size())]
            all_labels_list = [None for _ in range(dist.get_world_size())]
            dist.gather_object(local_preds, all_preds_list if dist.get_rank() == 0 else None)
            dist.gather_object(local_labels, all_labels_list if dist.get_rank() == 0 else None)

            if dist.get_rank() == 0:
                # flatten lists from all ranks
                all_preds = [p for sub in all_preds_list for p in sub]
                all_labels = [l for sub in all_labels_list for l in sub]

                report = classification_report(all_labels, all_preds, digits=4)
                print(f"\nValidation report:\n{report}")

                acc = np.mean(np.array(all_preds) == np.array(all_labels))
                acc_list.append(acc)
                os.makedirs(args.save_path, exist_ok=True)
                pd.DataFrame({"acc": acc_list}).to_csv(os.path.join(args.save_path, "acc_record.csv"), index=False)

                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.module.state_dict(),
                            os.path.join(args.save_path, f"best_model_{int(best_acc*100)}.pth"))
                    print(f"Best model saved with acc={best_acc:.4f}")

            dist.barrier()       # sync before returning to train
            model.train()
        

        global_step += 1


if dist.get_rank() == 0:
    print("Training complete. Best accuracy:", best_acc)

dist.destroy_process_group()