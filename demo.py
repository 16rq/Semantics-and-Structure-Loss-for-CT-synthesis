#!/usr/bin/env python3
"""LS2 demo: semantic loss (window / density) + structure loss (gradient).

Formulas copied from ct_ldm_rgb_ours.p_losses.

  python demo.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.autograd import Variable

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
MIN_HU, MAX_HU = -1000.0, 1000.0
SIZE = 256
WINDOW_SCALE = 0.05  # original: scale = 0.05


def load_dcm(path) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return np.asarray(arr[0] if arr.ndim == 3 else arr)


def resize(arr, size, nearest=False):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    old = np.array(img.GetSize(), dtype=np.float64)
    img.SetSpacing((old[0] / size, old[1] / size))
    out = sitk.Resample(
        img, [size, size], sitk.Transform(),
        sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear,
        img.GetOrigin(), img.GetSpacing(), img.GetDirection(), 0.0, img.GetPixelID(),
    )
    return sitk.GetArrayFromImage(out)


def hu_to_unit(hu: np.ndarray) -> torch.Tensor:
    hu = np.clip(hu.astype(np.float32), MIN_HU, MAX_HU)
    x = (hu - MIN_HU) / (MAX_HU - MIN_HU) * 2.0 - 1.0
    return torch.from_numpy(np.ascontiguousarray(x))[None, None]


# ---- original helpers (ct_ldm_rgb_ours.py) ----

def soft_histogram(x, bins, min_val, max_val, sigma=0.01):
    device = x.device
    bin_centers = torch.linspace(min_val, max_val, steps=bins).to(device)
    x = x.view(-1, 1)
    diff = x - bin_centers
    kernel = torch.exp(-0.5 * (diff / sigma) ** 2)
    kernel = kernel / (sigma * torch.sqrt(torch.tensor(2 * math.pi, device=device)))
    hist = kernel.sum(dim=0)
    return hist / (torch.sum(hist) + 1e-8)


def wasserstein_distance_soft(x, y, bins=64, value_range=(-1, 1), sigma=0.01):
    hist_x = soft_histogram(x, bins, value_range[0], value_range[1], sigma)
    hist_y = soft_histogram(y, bins, value_range[0], value_range[1], sigma)
    return torch.abs(torch.cumsum(hist_x, 0) - torch.cumsum(hist_y, 0)).sum()


def apply_window(fake_image, wl, ww):
    fake_window = ((fake_image + 1) / 2.0) * 2000 - 1000
    min_HU = wl - ww / 2.0
    max_HU = wl + ww / 2.0
    fake_window = fake_window.clone()
    fake_window[fake_window > max_HU] = max_HU
    fake_window[fake_window < min_HU] = min_HU
    fake_window = (fake_window - min_HU) / (max_HU - min_HU)
    return fake_window * 2 - 1


def window_binary_mask(real_image, wl, ww):
    real_hu = ((real_image + 1) / 2.0) * 2000 - 1000
    lower_bound = wl - ww / 2
    upper_bound = wl + ww / 2
    mask = (real_hu >= lower_bound) & (real_hu <= upper_bound)
    return torch.where(mask, torch.ones((), device=real_image.device), torch.zeros((), device=real_image.device))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(_2D.expand(channel, 1, window_size, window_size).contiguous())


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).type_as(img1)
            self.window, self.channel = window, channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def compute_attention_map(grad_x, grad_y, body_mask=None, threshold=0.0001):
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    if body_mask is not None:
        grad_magnitude = grad_magnitude * body_mask
    grad_magnitude = torch.where(grad_magnitude > threshold, grad_magnitude, torch.zeros_like(grad_magnitude))
    max_val = grad_magnitude.max()
    attention_map = torch.zeros_like(grad_magnitude) if max_val < 1e-8 else grad_magnitude / (max_val + 1e-8)
    c = attention_map.shape[1]
    kernel = torch.ones(c, 1, 3, 3, device=attention_map.device, dtype=attention_map.dtype) / 9.0
    attention_map = F.conv2d(attention_map, kernel, padding=1, groups=c)
    if attention_map.max() < 1e-8:
        return torch.zeros_like(attention_map)
    return attention_map / (attention_map.max() + 1e-8)


def semantic_loss(pred, real, mask_lbl):
    """Original window / density loss: 0.05 * (global_W1 + local_W1)."""
    pred3 = pred.repeat(1, 3, 1, 1)
    real3 = real.repeat(1, 3, 1, 1)
    mask3 = mask_lbl.repeat(1, 3, 1, 1)

    fake_ct_bone = apply_window(pred3, wl=700, ww=900)
    fake_ct_lung = apply_window(pred3, wl=-700, ww=500)
    fake_ct_HU = apply_window(pred3, wl=0, ww=500)
    bone_image = apply_window(real3, wl=700, ww=900)
    lung_image = apply_window(real3, wl=-700, ww=500)
    HU_image = apply_window(real3, wl=0, ww=500)
    windows = {
        "bone_mask": window_binary_mask(real3, 700, 900),
        "lung_mask": window_binary_mask(real3, -700, 500),
        "HU_mask": window_binary_mask(real3, 0, 500),
        "fake_ct_bone": fake_ct_bone, "fake_ct_lung": fake_ct_lung, "fake_ct_HU": fake_ct_HU,
        "bone_image": bone_image, "lung_image": lung_image, "HU_image": HU_image,
    }
    global_density_loss = {}
    for window in ["bone", "lung", "HU"]:
        masked_gt = windows[f"{window}_image"][windows[f"{window}_mask"].bool()]
        masked_pred = windows[f"fake_ct_{window}"][windows[f"{window}_mask"].bool()]
        if (torch.sum(masked_pred) == 0) or (torch.sum(masked_gt) == 0):
            global_density_loss[window] = torch.zeros((), device=pred.device)
        else:
            global_density_loss[window] = wasserstein_distance_soft(masked_pred, masked_gt)
    loss_global = torch.stack(list(global_density_loss.values())).mean()

    local_density_loss = {}
    for region in torch.unique(mask_lbl):
        rid = int(region.item())
        if rid == 0:
            continue
        masked_gt = real3[mask3 == rid]
        masked_pred = pred3[mask3 == rid]
        if (torch.sum(masked_pred) == 0) or (torch.sum(masked_gt) == 0):
            continue
        local_density_loss[rid] = wasserstein_distance_soft(masked_pred, masked_gt)
    loss_local = torch.stack(list(local_density_loss.values())).mean() if local_density_loss else torch.zeros((), device=pred.device)

    loss = WINDOW_SCALE * (loss_global + loss_local)
    return loss, loss_global, loss_local, len(local_density_loss)


def structure_loss(pred, real, ssim_fn):
    """Original Scharr/attention loss: 10*MSE + 0.1*(1-SSIM) on Sobel attention maps."""
    pred3 = pred.repeat(1, 3, 1, 1)
    real3 = real.repeat(1, 3, 1, 1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    kx, ky = kx.to(pred.device), ky.to(pred.device)
    grad_real_x = F.conv2d(real3, kx, padding=1, groups=3)
    grad_real_y = F.conv2d(real3, ky, padding=1, groups=3)
    grad_fake_x = F.conv2d(pred3, kx, padding=1, groups=3)
    grad_fake_y = F.conv2d(pred3, ky, padding=1, groups=3)
    attn_r = compute_attention_map(grad_real_x, grad_real_y)
    attn_f = compute_attention_map(grad_fake_x, grad_fake_y)
    mse = F.mse_loss(attn_r, attn_f)
    ssim_v = ssim_fn(attn_r, attn_f)
    loss = 10 * mse + 0.1 * (1 - ssim_v)
    return loss, mse, ssim_v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", default=str(DATA / "ct_0b57b247b6_42966_00066.dcm"))
    p.add_argument("--pred", default=str(DATA / "pred_0b57b247b6_42966_00066.dcm"))
    p.add_argument("--mask", default=str(DATA / "AutoPET_0b57b247b6_42966_00066.dcm"))
    args = p.parse_args()

    real = hu_to_unit(resize(load_dcm(args.real), SIZE))
    pred = hu_to_unit(resize(load_dcm(args.pred), SIZE))
    mask = torch.from_numpy(resize(load_dcm(args.mask), SIZE, nearest=True).astype(np.int64))[None, None]

    L_sem, g, loc, n_roi = semantic_loss(pred, real, mask)
    L_str, mse, ssim_v = structure_loss(pred, real, SSIM(window_size=7))
    L_ls2 = L_sem + L_str

    print("semantic loss     {:.6f}".format(float(L_sem)))
    print("structure loss    {:.6f}".format(float(L_str)))
    print("LS2               {:.6f}".format(float(L_ls2)))


if __name__ == "__main__":
    main()
