import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import List, Tuple

BBox = Tuple[int, int, int, int]



# MSE error map


def compute_mse_map(input_arr:  np.ndarray,
                    recon_arr:  np.ndarray) -> np.ndarray:
    diff = (input_arr.astype(np.float32) -
            recon_arr.astype(np.float32)) ** 2
    if diff.ndim == 3:
        diff = diff.mean(axis=2)

    mn, mx = diff.min(), diff.max()
    return (diff - mn) / (mx - mn + 1e-8)



# SSIM structural error map

def compute_ssim_map(input_arr:   np.ndarray,
                     recon_arr:   np.ndarray,
                     window_size: int   = 11,
                     sigma:       float = 1.5) -> np.ndarray:
    C1, C2 = (0.01 ** 2), (0.03 ** 2)

    # Build Gaussian window
    coords = np.arange(window_size) - window_size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window = np.outer(g, g).astype(np.float32)

    pad = window_size // 2
    x = input_arr.astype(np.float32)
    y = recon_arr .astype(np.float32)

    def _convolve(arr):
        return cv2.filter2D(arr, -1, window,
                            borderType=cv2.BORDER_REFLECT_101)

    mu_x  = _convolve(x)
    mu_y  = _convolve(y)
    mu_xx = _convolve(x * x)
    mu_yy = _convolve(y * y)
    mu_xy = _convolve(x * y)

    sig_x  = mu_xx - mu_x ** 2
    sig_y  = mu_yy - mu_y ** 2
    sig_xy = mu_xy - mu_x * mu_y

    ssim_map = (
        (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    ) / (
        (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
    )

    # Clip to [0, 1] (numerical noise can push slightly outside)
    ssim_map = np.clip(ssim_map, 0.0, 1.0)

    # Invert: 1 - SSIM so high values = high structural error
    error_map = 1.0 - ssim_map

    # Normalise to [0, 1]
    mn, mx = error_map.min(), error_map.max()
    return (error_map - mn) / (mx - mn + 1e-8)



# Dual-Signal Fusion

def compute_dual_signal_error(input_arr:   np.ndarray,
                               recon_arr:   np.ndarray,
                               alpha:       float = 0.5) -> dict:
    
    mse_map  = compute_mse_map(input_arr,  recon_arr)
    ssim_map = compute_ssim_map(input_arr, recon_arr)

    fused = alpha * mse_map + (1.0 - alpha) * ssim_map

    # Final normalisation so the threshold is always in [0, 1]
    mn, mx = fused.min(), fused.max()
    fused  = (fused - mn) / (mx - mn + 1e-8)

    return {
        "mse_map"   : mse_map,
        "ssim_map"  : ssim_map,
        "fused_map" : fused,
        "mean_error": float(fused.mean()),
    }




def error_map_to_binary(error_map: np.ndarray,
                        threshold: float = 0.40,
                        min_area:  int   = 80) -> np.ndarray:
    
    blurred = cv2.GaussianBlur(error_map, (5, 5), sigmaX=1.0)
    binary  = (blurred > threshold).astype(np.uint8) * 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned



def mask_to_bboxes(binary_mask: np.ndarray) -> List[BBox]:
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [cv2.boundingRect(c) for c in contours]


def nms_boxes(boxes: List[BBox], iou_threshold: float = 0.40) -> List[BBox]:
    if not boxes:
        return []
    rects = np.array([[x, y, x+w, y+h] for x,y,w,h in boxes],
                     dtype=np.float32)
    x1,y1,x2,y2 = rects[:,0],rects[:,1],rects[:,2],rects[:,3]
    areas = (x2-x1)*(y2-y1)
    order = areas.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
        iou   = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order = order[1:][iou < iou_threshold]
    return [boxes[k] for k in keep]


def scale_boxes_to_original(boxes:      List[BBox],
                             roi_size:  Tuple[int,int],
                             orig_size: Tuple[int,int],
                             roi_offset: Tuple[int,int] = (0,0)) -> List[BBox]:
    roi_h, roi_w = roi_size
    scale_y, scale_x = roi_h/224, roi_w/224
    y_off, x_off = roi_offset
    return [
        (int(x*scale_x)+x_off, int(y*scale_y)+y_off,
         int(w*scale_x),       int(h*scale_y))
        for x,y,w,h in boxes
    ]



def apply_colormap(heatmap: np.ndarray, alpha: float = 0.55,
                   image: np.ndarray | None = None) -> np.ndarray:
    hm_u8   = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    if image is not None:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if colored.shape[:2] != image.shape[:2]:
            colored = cv2.resize(colored, (image.shape[1], image.shape[0]))
        return cv2.addWeighted(image, 1-alpha, colored, alpha, 0)
    return colored


def draw_boxes(image: np.ndarray, boxes: List[BBox],
               label: str = "anomaly",
               color: Tuple = (0,255,80), thickness: int = 2) -> np.ndarray:
    out = image.copy()
    for (x,y,w,h) in boxes:
        cv2.rectangle(out, (x,y), (x+w,y+h), color, thickness)
        cv2.putText(out, label, (x, max(y-6,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def visualise_result(original_rgb:  np.ndarray,
                     recon_gray:    np.ndarray,
                     error_signals: dict,
                     boxes:         List[BBox],
                     save_path:     str | None = None) -> None:
    
    orig_h, orig_w = original_rgb.shape[:2]
    bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

    def _resize_map(m):
        return cv2.resize(m, (orig_w, orig_h)) if m.shape[:2] != (orig_h, orig_w) else m

    mse_full   = _resize_map(error_signals["mse_map"])
    ssim_full  = _resize_map(error_signals["ssim_map"])
    fused_full = _resize_map(error_signals["fused_map"])

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    fig.patch.set_facecolor("#0f0f0f")

    # 1 — Original
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original OPG", color="white", fontsize=11)
    axes[0].axis("off")

    # 2 — Reconstruction
    recon_disp = cv2.resize(recon_gray, (orig_w, orig_h))
    axes[1].imshow(recon_disp, cmap="gray")
    axes[1].set_title("AE Reconstruction\n(model's normal version)",
                      color="white", fontsize=11)
    axes[1].axis("off")

    # 3 — MSE map
    axes[2].imshow(apply_colormap(mse_full, alpha=0.6, image=bgr)[:,:,::-1])
    axes[2].set_title("MSE error map\n(intensity anomalies)",
                      color="white", fontsize=11)
    axes[2].axis("off")

    # 4 — SSIM structural map
    axes[3].imshow(apply_colormap(ssim_full, alpha=0.6, image=bgr)[:,:,::-1])
    axes[3].set_title("SSIM error map\n(structural anomalies)",
                      color="white", fontsize=11)
    axes[3].axis("off")

    # 5 — Fused + boxes
    fused_overlay = apply_colormap(fused_full, alpha=0.55, image=bgr)
    fused_annotated = draw_boxes(fused_overlay, boxes, color=(0,255,80))
    axes[4].imshow(fused_annotated[:,:,::-1])
    axes[4].set_title(
        f"Dual-signal fusion\n{len(boxes)} anomaly region(s)  "
        f"[score: {error_signals['mean_error']:.3f}]",
        color="white", fontsize=11
    )
    axes[4].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Visualise] Saved → {save_path}")
    plt.show()
    plt.close()



def calibrate_threshold(normal_errors: List[float],
                        percentile: float = 95.0) -> float:
    arr = np.array(normal_errors)
    t   = float(np.percentile(arr, percentile))
    print(f"[Calibrate] Normal error stats:")
    print(f"            mean={arr.mean():.4f}  std={arr.std():.4f}")
    print(f"            p90={np.percentile(arr,90):.4f}  "
          f"p95={np.percentile(arr,95):.4f}  "
          f"p99={np.percentile(arr,99):.4f}")
    print(f"[Calibrate] Suggested threshold ({percentile:.0f}th pct): {t:.4f}")
    return t
