import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Constants
AE_SIZE       = 224         # autoencoder input (square)
MASK_DILATE   = 7           # dilate mask to avoid clipping tooth edges
SEG_W, SEG_H  = 512, 256   # UNet input


# ROI extraction

def apply_mask_and_crop(image_rgb: np.ndarray,
                        mask: np.ndarray) -> tuple[np.ndarray, tuple]:
    
    bin_mask = (mask > 0).astype(np.uint8)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (MASK_DILATE, MASK_DILATE))
    bin_mask = cv2.dilate(bin_mask, kernel, iterations=1)

    roi = image_rgb.copy()
    roi[bin_mask == 0] = 0

    ys, xs = np.where(bin_mask)
    if len(ys) == 0:
        return roi, (0, 0)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return roi[y0:y1+1, x0:x1+1], (y0, x0)


# Transform

def get_ae_transform(split: str = "train") -> transforms.Compose:
    
    base = [
        transforms.Resize((AE_SIZE, AE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
    ]
    if split == "train":
        base += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=3, translate=(0.03, 0.03)),
            # Light contrast jitter — simulate X-ray exposure variation
            transforms.ColorJitter(brightness=0.1, contrast=0.2),
        ]
    base.append(transforms.ToTensor())
    return transforms.Compose(base)


# Dataset

class NormalROIDataset(Dataset):

    def __init__(self,
                 image_dir: str,
                 mask_dir:  str | None = None,
                 seg_checkpoint: str | None = None,
                 split:     str = "train",
                 device:    str = "cpu"):

        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.split     = split
        self.device    = device
        self.transform = get_ae_transform(split)

        self._seg_model = None
        if mask_dir is None and seg_checkpoint is not None:
            from segmentation import build_seg_model
            self._seg_model = build_seg_model(seg_checkpoint, device)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
        self.files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])

        print(f"[NormalROIDataset] {split}: {len(self.files)} normal images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname     = self.files[idx]
        img_path  = os.path.join(self.image_dir, fname)

        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        
        mask = self._get_mask(fname, img_rgb)

        
        roi_rgb, _ = apply_mask_and_crop(img_rgb, mask)
        roi_pil    = Image.fromarray(roi_rgb)

        
        roi_tensor = self.transform(roi_pil)

        return roi_tensor

    def _get_mask(self, fname: str, img_rgb: np.ndarray) -> np.ndarray:
        if self.mask_dir:
            for candidate in [fname, os.path.splitext(fname)[0] + ".png"]:
                p = os.path.join(self.mask_dir, candidate)
                if os.path.exists(p):
                    mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        if mask.shape != img_rgb.shape[:2]:
                            mask = cv2.resize(
                                mask, (img_rgb.shape[1], img_rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        return mask

        if self._seg_model is not None:
            from segmentation import predict_mask
            return predict_mask(self._seg_model, img_rgb,
                                device=self.device,
                                orig_size=img_rgb.shape[:2])

        return np.ones(img_rgb.shape[:2], dtype=np.uint8) * 255


# DataLoader builder
class _Subset(NormalROIDataset):
    
    def __init__(self, base_ds: "NormalROIDataset", file_list: list, split: str):
        self.__dict__.update(base_ds.__dict__)
        self.files     = file_list
        self.transform = get_ae_transform(split)


def build_ae_dataloaders(normal_image_dir: str,
                         mask_dir:         str | None = None,
                         seg_checkpoint:   str | None = None,
                         batch_size:       int   = 16,
                         val_split:        float = 0.15,
                         num_workers:      int   = 4,
                         device:           str   = "cpu",
                         seed:             int   = 42):
    
    from sklearn.model_selection import train_test_split

    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    files = sorted([f for f in os.listdir(normal_image_dir)
                    if os.path.splitext(f)[1].lower() in exts])

    if len(files) < 2:
        raise ValueError(f"Not enough normal images in {normal_image_dir}")

    train_files, val_files = train_test_split(files, test_size=val_split,
                                              random_state=seed)

    base     = NormalROIDataset(normal_image_dir, mask_dir,
                                seg_checkpoint, split="train", device=device)
    train_ds = _Subset(base, train_files, split="train")
    val_ds   = _Subset(base, val_files,   split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=num_workers,
                          pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=True, persistent_workers=True)

    print(f"[DataLoader] train={len(train_ds)}  val={len(val_ds)}")
    return train_loader, val_loader
