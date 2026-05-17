import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autoencoder import build_autoencoder
from dataset_ae  import build_ae_dataloaders


CONFIG = {
    
    "normal_image_dir" : "/Users/adityamaharana/Desktop/ToothAnomaly/data/normal/images",
    "mask_dir"         : "/Users/adityamaharana/Desktop/ToothAnomaly/data/normal/masks",
    "seg_checkpoint"   : "/Users/adityamaharana/Desktop/ToothAnomaly/models/unet50.pth",
    "checkpoint"       : "checkpoints/autoencoder.pth",

    
    "latent_channels"  : 128,

    
    "epochs"           : 60,
    "batch_size"       : 16,
    "lr"               : 1e-3,
    "weight_decay"     : 1e-5,
    "ssim_weight"      : 0.5,
    "patience"         : 12,
    "early_stop"       : 20,

    
    "val_split"        : 0.15,
    "num_workers"      : 2,
    "seed"             : 42,
}



class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._make_window(window_size, sigma))

    @staticmethod
    def _make_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)
        return window

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        w   = self.window.to(x.device)

        mu_x  = torch.nn.functional.conv2d(x, w, padding=pad, groups=1)
        mu_y  = torch.nn.functional.conv2d(y, w, padding=pad, groups=1)
        mu_xx = torch.nn.functional.conv2d(x * x, w, padding=pad, groups=1)
        mu_yy = torch.nn.functional.conv2d(y * y, w, padding=pad, groups=1)
        mu_xy = torch.nn.functional.conv2d(x * y, w, padding=pad, groups=1)

        sig_x  = mu_xx - mu_x ** 2
        sig_y  = mu_yy - mu_y ** 2
        sig_xy = mu_xy - mu_x * mu_y

        ssim_map = (
            (2 * mu_x * mu_y  + C1) * (2 * sig_xy + C2)
        ) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2)
        )
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.5):
        super().__init__()
        self.mse        = nn.MSELoss()
        self.ssim       = SSIMLoss()
        self.ssim_w     = ssim_weight

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target) + self.ssim_w * self.ssim(pred, target)


 

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        d = torch.device("cpu")
        print("[Device] CPU (consider using a GPU for faster training)")
    return d




def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for roi in loader:
            roi = roi.to(device, non_blocking=True)

            recon = model(roi)
            loss  = criterion(recon, roi)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * roi.size(0)

    return total_loss / len(loader.dataset)




def train():
    set_seed(CONFIG["seed"])
    device = get_device()
    os.makedirs(os.path.dirname(CONFIG["checkpoint"]), exist_ok=True)

    
    train_loader, val_loader = build_ae_dataloaders(
        normal_image_dir = CONFIG["normal_image_dir"],
        mask_dir         = CONFIG["mask_dir"],
        seg_checkpoint   = CONFIG["seg_checkpoint"],
        batch_size       = CONFIG["batch_size"],
        val_split        = CONFIG["val_split"],
        num_workers      = CONFIG["num_workers"],
        device           = str(device),
        seed             = CONFIG["seed"],
    )

    
    model = build_autoencoder(
        latent_channels = CONFIG["latent_channels"],
        device          = str(device),
    )
    print(f"\n[Model] Params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    
    criterion = CombinedLoss(ssim_weight=CONFIG["ssim_weight"])
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG["lr"],
                           weight_decay=CONFIG["weight_decay"])
    
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=0.5, patience=CONFIG["patience"],
                                  min_lr=1e-6)

    
    best_val_loss    = float("inf")
    early_stop_count = 0

    print(f"\n{'─'*60}")
    print(f"  Training Autoencoder on NORMAL teeth only")
    print(f"  Loss = MSE + {CONFIG['ssim_weight']} × (1−SSIM)")
    print(f"{'─'*60}\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        tr_loss  = run_epoch(model, train_loader, criterion,
                             optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader,   criterion,
                             optimizer, device, train=False)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:3d}/{CONFIG['epochs']}]"
              f"  train {tr_loss:.6f}  val {val_loss:.6f}"
              f"  lr={optimizer.param_groups[0]['lr']:.2e}"
              f"  {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(CONFIG["checkpoint"])
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= CONFIG["early_stop"]:
                print(f"\n[Early stop] Val loss stalled for "
                      f"{CONFIG['early_stop']} epochs.")
                break

    print(f"\n✓ Best val loss : {best_val_loss:.6f}")
    print(f"✓ Checkpoint    : {CONFIG['checkpoint']}")
    print(f"\n→ Next step: calibrate the anomaly threshold using validation images.")
    print(f"  Run: python inference.py --calibrate --image_dir data/normal/val")


if __name__ == "__main__":
    train()
