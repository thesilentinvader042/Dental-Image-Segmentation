import os
import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from segmentation import build_seg_model, predict_mask
from autoencoder  import DentalAutoencoder, build_autoencoder
from utils        import (compute_dual_signal_error, error_map_to_binary,
                           mask_to_bboxes, nms_boxes,
                           scale_boxes_to_original, draw_boxes,
                           visualise_result, calibrate_threshold)
from dataset_ae   import apply_mask_and_crop, AE_SIZE

_AE_TRANSFORM = transforms.Compose([
    transforms.Resize((AE_SIZE, AE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")



def run_inference(image_path: str,
                  seg_model,
                  ae_model:   DentalAutoencoder,
                  device:     torch.device,
                  threshold:  float = 0.40,
                  alpha:      float = 0.50,
                  min_area:   int   = 80,
                  save_path:  str | None = None) -> dict:
    
    # 1. Load
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    # 2. Segment
    mask = predict_mask(seg_model, image_rgb,
                        device=str(device), orig_size=(orig_h, orig_w))

    # 3. ROI
    roi_rgb, (y0, x0) = apply_mask_and_crop(image_rgb, mask)
    roi_h, roi_w      = roi_rgb.shape[:2]

    # 4. AE input tensor
    roi_tensor = _AE_TRANSFORM(Image.fromarray(roi_rgb)
                                ).unsqueeze(0).to(device)

    # 5. Reconstruction
    ae_model.eval()
    with torch.no_grad():
        recon_tensor = ae_model(roi_tensor)

    input_np = roi_tensor .squeeze().cpu().numpy()
    recon_np = recon_tensor.squeeze().cpu().numpy()

    # 6. DUAL-SIGNAL ERROR FUSION
    signals = compute_dual_signal_error(input_np, recon_np, alpha=alpha)

    # 7. Threshold fused map → binary mask
    binary_mask = error_map_to_binary(signals["fused_map"],
                                      threshold=threshold,
                                      min_area=min_area)

    # 8. Boxes: AE space → original image coords
    boxes_ae   = mask_to_bboxes(binary_mask)
    boxes_ae   = nms_boxes(boxes_ae, iou_threshold=0.40)
    boxes_orig = scale_boxes_to_original(
        boxes_ae, roi_size=(roi_h, roi_w),
        orig_size=(orig_h, orig_w), roi_offset=(y0, x0)
    )

    # Scale all error maps to original image size
    def _up(m): return cv2.resize(m, (orig_w, orig_h))
    signals_full = {
        "mse_map"   : _up(signals["mse_map"]),
        "ssim_map"  : _up(signals["ssim_map"]),
        "fused_map" : _up(signals["fused_map"]),
        "mean_error": signals["mean_error"],
    }

    recon_disp = (recon_np * 255).astype(np.uint8)  # (224,224) grayscale uint8

    print(f"[SegRAD] {os.path.basename(image_path)}"
          f"  score={signals['mean_error']:.4f}"
          f"  {'ANOMALY' if boxes_orig else 'normal'}"
          f"  {len(boxes_orig)} region(s)")

    visualise_result(image_rgb, recon_disp, signals_full,
                     boxes_orig, save_path=save_path)

    return {
        "mean_error"   : signals["mean_error"],
        "is_anomalous" : len(boxes_orig) > 0,
        "signals"      : signals_full,
        "boxes"        : boxes_orig,
        "annotated_bgr": draw_boxes(image_bgr, boxes_orig),
    }




def calibrate(image_dir: str, seg_model, ae_model,
              device: torch.device, alpha: float = 0.7) -> float:
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in sorted(os.listdir(image_dir))
             if os.path.splitext(f)[1].lower() in exts]
    errors = []
    ae_model.eval()
    for fname in files:
        bgr = cv2.imread(os.path.join(image_dir, fname))
        if bgr is None: continue
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = predict_mask(seg_model, rgb,
                            device=str(device), orig_size=rgb.shape[:2])
        roi, _ = apply_mask_and_crop(rgb, mask)
        t = _AE_TRANSFORM(Image.fromarray(roi)).unsqueeze(0).to(device)
        with torch.no_grad():
            r = ae_model(t)
        s = compute_dual_signal_error(
            t.squeeze().cpu().numpy(),
            r.squeeze().cpu().numpy(), alpha=alpha
        )
        errors.append(s["mean_error"])
    return calibrate_threshold(errors, percentile=95.0)



def run_batch(image_dir, seg_ckpt, ae_ckpt,
              output_dir="results", threshold=0.40,
              alpha=0.5, min_area=80):
    os.makedirs(output_dir, exist_ok=True)
    device    = get_device()
    seg_model = build_seg_model(seg_ckpt, device=str(device))
    ae_model  = build_autoencoder(device=str(device))
    ae_model.load(ae_ckpt, device=str(device))

    exts  = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    files = sorted([f for f in os.listdir(image_dir)
                    if os.path.splitext(f)[1].lower() in exts])
    print(f"\n[SegRAD Batch] {len(files)} images | threshold={threshold} | alpha={alpha}\n")

    summary = []
    for fname in files:
        img_path = os.path.join(image_dir, fname)
        out_path = os.path.join(output_dir,
                                f"result_{os.path.splitext(fname)[0]}.png")
        try:
            r = run_inference(img_path, seg_model, ae_model,
                              device, threshold, alpha, min_area,
                              save_path=out_path)
            summary.append({
                "filename"    : fname,
                "mean_error"  : round(r["mean_error"], 5),
                "is_anomalous": r["is_anomalous"],
                "n_regions"   : len(r["boxes"]),
            })
        except Exception as e:
            print(f"[Error] {fname}: {e}")
            summary.append({"filename": fname, "error": str(e)})

    import pandas as pd
    csv_path = os.path.join(output_dir, "segrad_summary.csv")
    pd.DataFrame(summary).to_csv(csv_path, index=False)
    print(f"\n[SegRAD] Done — {csv_path}")



def main():
    p = argparse.ArgumentParser(description="SegRAD — Dual-Signal OPG anomaly detection")
    p.add_argument("--image",       type=str)
    p.add_argument("--image_dir",   type=str)
    p.add_argument("--output_dir",  type=str, default="results")
    p.add_argument("--seg_ckpt",    type=str, required=True)
    p.add_argument("--ae_ckpt",     type=str, required=True)
    p.add_argument("--threshold",   type=float, default=0.40)
    p.add_argument("--alpha",       type=float, default=0.50,
                   help="MSE weight in dual-signal fusion (0.0–1.0)")
    p.add_argument("--min_area",    type=int, default=80)
    p.add_argument("--calibrate",   action="store_true")
    args = p.parse_args()

    device    = get_device()
    seg_model = build_seg_model(args.seg_ckpt, device=str(device))
    ae_model  = build_autoencoder(device=str(device))
    ae_model.load(args.ae_ckpt, device=str(device))

    if args.calibrate:
        t = calibrate(args.image_dir, seg_model, ae_model,
                      device, alpha=args.alpha)
        print(f"\n→ Re-run with: --threshold {t:.4f}")
    elif args.image:
        run_inference(args.image, seg_model, ae_model, device,
                      threshold=args.threshold, alpha=args.alpha,
                      min_area=args.min_area)
    elif args.image_dir:
        run_batch(args.image_dir, args.seg_ckpt, args.ae_ckpt,
                  args.output_dir, args.threshold,
                  args.alpha, args.min_area)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
