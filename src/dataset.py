import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# NOTE: MIND THE VERSION !!! WHERE VENOUS AND DELAYED MIGHT BE SWITCHED
label_mapping = {
    'non-contrast': 0,
    'arterial': 1,
    'venous': 2,
    'delayed': 3,
}



class CLSInferenceDataset(Dataset):
    """
    Load a single NIfTI volume and yield slice_num 3-slice (2.5D) groups along z-axis.
    Each item -> dict:
        {
            "image": (3, H, W) torch.float32,  # normalized [0,1]
            "slice_idx": int,                  # starting slice index
        }

    """

    def __init__(self, load_path, clip=(-1000, 1000), slice_num=None, transform=None, resize_hw=(512, 512)):
        
        super().__init__()
        self.load_path = load_path
        self.clip_min, self.clip_max = clip
        self.transform = transform
        self.resize_hw = resize_hw

        nii = nib.load(load_path)
        vol = nii.get_fdata().astype(np.float32)  # (H, W, D)

        vol = np.clip(vol, self.clip_min, self.clip_max)
        vol = (vol - self.clip_min) / (self.clip_max - self.clip_min) #[0,1]
        
        self.volume = vol
        self.H, self.W, self.D = vol.shape

        self.valid_indices = list(range(self.D//8, self.D*7//8)) # avoid the upper and lower boundary

        # downsample to slice_num, where evenly distributed from the whole-body scan
        if slice_num is not None and slice_num < len(self.valid_indices):
            indices = np.linspace(0, len(self.valid_indices) - 1, slice_num, dtype=int)
            self.valid_indices = [self.valid_indices[i] for i in indices]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx=None):
        
        if idx is None:
            idx = np.random.randint(0, len(self.valid_indices))
        z = self.valid_indices[idx]

        # --- Extract (H, W, 3)
        stack = self.volume[:, :, z:z + 3]  # 3-adjacent slices
        tensor = torch.from_numpy(stack).permute(2, 0, 1).unsqueeze(0)   # (3,H,W)

        tensor = F.interpolate(
            tensor, size=self.resize_hw, mode='bilinear', align_corners=False
        ).squeeze(0)  # (3,512,512)
        sample = {"image": tensor, "slice_idx": z}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample