
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


SEG_W, SEG_H = 512, 256
SEG_THRESHOLD = 0.5


def build_seg_model(checkpoint_path: str,
                    device: str = "cpu") -> smp.Unet:
    
    model = smp.Unet(
        encoder_name    = "resnet18",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
    )

    state = torch.load(checkpoint_path, map_location=device)

    
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[Segmentation] UNet loaded from {checkpoint_path} on {device}")
    return model


def preprocess_for_seg(image_rgb: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image_rgb, (SEG_W, SEG_H))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return torch.from_numpy(img)


def predict_mask(model:     smp.Unet,
                 image_rgb: np.ndarray,
                 device:    str = "cpu",
                 orig_size: tuple | None = None) -> np.ndarray:
    
    tensor = preprocess_for_seg(image_rgb).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits)

    prob_np = probs.squeeze().cpu().numpy()

    
    binary = (prob_np > SEG_THRESHOLD).astype(np.uint8) * 255

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    
    if orig_size is not None:
        orig_h, orig_w = orig_size
        binary = cv2.resize(binary, (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST)

    return binary


# Batch mask generation

def presave_masks_for_dir(image_dir:  str,
                          output_dir: str,
                          checkpoint: str = "unet50.pth",
                          device:     str = "cpu"):
    
    import os
    os.makedirs(output_dir, exist_ok=True)

    seg_model = build_seg_model(checkpoint, device)

    exts  = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if not files:
        print(f"[presave_masks] No images found in {image_dir}")
        return

    print(f"[presave_masks] Generating masks for {len(files)} images → {output_dir}")

    for i, fname in enumerate(files):
        img_bgr = cv2.imread(os.path.join(image_dir, fname))
        if img_bgr is None:
            print(f"  [skip] cannot read {fname}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask    = predict_mask(seg_model, img_rgb, device,
                               orig_size=img_rgb.shape[:2])

        
        out_name = os.path.splitext(fname)[0] + ".png"
        cv2.imwrite(os.path.join(output_dir, out_name), mask)

        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)} done")

    print(f"[presave_masks] Complete — masks saved to {output_dir}")



_CACHED_SEG_MODEL = None
_CACHED_CKPT_PATH = None
_CACHED_DEVICE    = None


def segment(image_rgb:       np.ndarray,
            checkpoint_path: str   = "unet50.pth",
            device:          str   = "cpu",
            return_orig_size: bool = True) -> np.ndarray:
    
    global _CACHED_SEG_MODEL, _CACHED_CKPT_PATH, _CACHED_DEVICE

    if (_CACHED_SEG_MODEL is None or
            _CACHED_CKPT_PATH != checkpoint_path or
            _CACHED_DEVICE != device):
        _CACHED_SEG_MODEL = build_seg_model(checkpoint_path, device)
        _CACHED_CKPT_PATH = checkpoint_path
        _CACHED_DEVICE    = device

    orig_size = image_rgb.shape[:2] if return_orig_size else None
    return predict_mask(_CACHED_SEG_MODEL, image_rgb, device, orig_size)
