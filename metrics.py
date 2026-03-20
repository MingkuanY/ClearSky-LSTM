import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from data import DBZ_MAX, DBZ_MIN
from loss_functions import denormalize_dbz

EPS = 1e-8


def _ensure_dbz(x: torch.Tensor) -> torch.Tensor:
    # Input may be normalized [0,1] or already dBZ
    if x.dtype.is_floating_point and x.min() >= -0.1 and x.max() <= 1.1:
        return denormalize_dbz(x)
    return x


def _as_bt_hw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    if x.ndim != 4:
        raise ValueError(f"Expect BxTxHxW or BxTx1xHxW, got shape {tuple(x.shape)}")
    return x


def _valid_mask(target_dbz: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = target_dbz > DBZ_MIN
    else:
        mask = mask.bool()
    return mask


def _reduce_per_lead(error_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = error_map * mask
    valid_count = mask.sum(dim=(0, 2, 3)).clamp(min=1)
    return (masked.sum(dim=(0, 2, 3)) / valid_count)


def _reduce_global(error_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = error_map * mask
    valid_count = mask.sum().clamp(min=1)
    return masked.sum() / valid_count


def regression_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
    pred_dbz = _ensure_dbz(pred)
    target_dbz = _ensure_dbz(target)

    pred_dbz = _as_bt_hw(pred_dbz)
    target_dbz = _as_bt_hw(target_dbz)

    valid = _valid_mask(target_dbz, mask=mask)
    valid = valid.to(pred_dbz.dtype)

    abs_error = (pred_dbz - target_dbz).abs()
    sq_error = (pred_dbz - target_dbz) ** 2

    mae_lead = _reduce_per_lead(abs_error, valid)
    mse_lead = _reduce_per_lead(sq_error, valid)
    rmse_lead = torch.sqrt(mse_lead + EPS)
    mae = _reduce_global(abs_error, valid)
    mse = _reduce_global(sq_error, valid)
    rmse = torch.sqrt(mse + EPS)

    return {
        "mae_lead": mae_lead,
        "mse_lead": mse_lead,
        "rmse_lead": rmse_lead,
        "mae": mae.item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
    }


def _binary_contingency(pred_dbz, target_dbz, threshold, valid_mask):
    p = (pred_dbz >= threshold) & valid_mask
    t = (target_dbz >= threshold) & valid_mask

    tp = (p & t).sum(dim=(0, 2, 3)).float()
    fp = (p & ~t).sum(dim=(0, 2, 3)).float()
    fn = (~p & t).sum(dim=(0, 2, 3)).float()

    csi = tp / (tp + fp + fn + EPS)
    pod = tp / (tp + fn + EPS)
    far = fp / (tp + fp + EPS)

    return csi, pod, far


def contingency_metrics(pred: torch.Tensor, target: torch.Tensor, thresholds=(20.0, 40.0, 50.0), mask: torch.Tensor | None = None):
    pred_dbz = _ensure_dbz(pred)
    target_dbz = _ensure_dbz(target)

    pred_dbz = _as_bt_hw(pred_dbz)
    target_dbz = _as_bt_hw(target_dbz)

    valid = _valid_mask(target_dbz, mask=mask)

    output = {}
    for thr in thresholds:
        csi, pod, far = _binary_contingency(pred_dbz, target_dbz, thr, valid)
        output[f"CSI_{int(thr)}"] = csi
        output[f"POD_{int(thr)}"] = pod
        output[f"FAR_{int(thr)}"] = far

    return output


def fractions_skill_score(pred: torch.Tensor, target: torch.Tensor, thresholds=(20.0, 40.0, 50.0), window_sizes=(3, 7, 11), mask: torch.Tensor | None = None):
    pred_dbz = _ensure_dbz(pred)
    target_dbz = _ensure_dbz(target)
    pred_dbz = _as_bt_hw(pred_dbz)
    target_dbz = _as_bt_hw(target_dbz)

    valid = _valid_mask(target_dbz, mask=mask).to(pred_dbz.dtype)

    metrics = {}
    batch, lead, h, w = pred_dbz.shape

    for thr in thresholds:
        pbin = (pred_dbz >= thr).to(pred_dbz.dtype) * valid
        tbin = (target_dbz >= thr).to(pred_dbz.dtype) * valid

        for window in window_sizes:
            kernel = torch.ones(1, 1, window, window, device=pred_dbz.device, dtype=pred_dbz.dtype)
            pad = window // 2

            # reshape to merge batch+lead
            p_reshaped = pbin.reshape(batch * lead, 1, h, w)
            t_reshaped = tbin.reshape(batch * lead, 1, h, w)
            valid_reshaped = valid.reshape(batch * lead, 1, h, w)

            raw_valid_count = F.conv2d(valid_reshaped, kernel, padding=pad)
            valid_count = raw_valid_count.clamp(min=1.0)

            p_f = F.conv2d(p_reshaped, kernel, padding=pad) / valid_count
            t_f = F.conv2d(t_reshaped, kernel, padding=pad) / valid_count

            p_f = p_f.reshape(batch, lead, h, w)
            t_f = t_f.reshape(batch, lead, h, w)
            neighborhood_valid = (raw_valid_count > 0).reshape(batch, lead, h, w).to(pred_dbz.dtype)

            diff_sq = (p_f - t_f) ** 2
            ref_term = p_f ** 2 + t_f ** 2

            neighborhood_count = neighborhood_valid.sum(dim=(0, 2, 3)).clamp(min=1.0)
            mse_f = (diff_sq * neighborhood_valid).sum(dim=(0, 2, 3)) / neighborhood_count
            mse_ref = (ref_term * neighborhood_valid).sum(dim=(0, 2, 3)) / neighborhood_count
            fss = 1.0 - mse_f / (mse_ref + EPS)

            metrics[f"FSS_thr{int(thr)}_w{window}"] = fss

    return metrics


def rapsd_field(field: torch.Tensor, d: float = 1.0, return_freq=True):
    # field: (H, W), real-valued reflectivity field
    if field.ndim != 2:
        raise ValueError("rapsd_field requires 2D input")

    field = field.detach().cpu().float()

    fft = torch.fft.fft2(field)
    psd2d = (fft.abs() ** 2) / field.numel()
    psd2d = torch.fft.fftshift(psd2d)

    h, w = field.shape
    y = torch.arange(h, dtype=torch.float32) - (h // 2)
    x = torch.arange(w, dtype=torch.float32) - (w // 2)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")
    radius = torch.sqrt(x_grid ** 2 + y_grid ** 2)

    max_radius = int(radius.max().item())
    bins = torch.arange(max_radius + 1)

    bin_psd = torch.zeros_like(bins, dtype=torch.float32)

    r_flat = radius.flatten().long()
    psd_flat = psd2d.flatten()

    for b in range(len(bins)):
        mask = r_flat == b
        if mask.any():
            bin_psd[b] = psd_flat[mask].mean()

    freqs = bins / (max(h, w) * d)
    if return_freq:
        return freqs, bin_psd
    return bin_psd


def _masked_field(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(field.dtype)
    return field * valid


def rapsd_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    d: float = 1.0,
    mask: torch.Tensor | None = None,
):
    """
    Reflectivity-space PSD discrepancy metric for nowcasting.

    Compare radially averaged 2D power spectral densities on reflectivity fields.
    The returned scalar is the RMSE between log10 PSD curves (excluding the
    zero-frequency/DC bin), averaged over samples and reported per lead time and
    overall.
    """
    pred_dbz = _ensure_dbz(pred)
    target_dbz = _ensure_dbz(target)
    pred_dbz = _as_bt_hw(pred_dbz)
    target_dbz = _as_bt_hw(target_dbz)
    valid = _valid_mask(target_dbz, mask=mask)

    batch, lead, h, w = pred_dbz.shape
    dist_per_lead = []

    for t in range(lead):
        diffs = []
        for b in range(batch):
            field_mask = valid[b, t]
            _, p_psd = rapsd_field(_masked_field(pred_dbz[b, t], field_mask), d=d, return_freq=True)
            _, t_psd = rapsd_field(_masked_field(target_dbz[b, t], field_mask), d=d, return_freq=True)
            n = min(len(p_psd), len(t_psd))
            if n <= 1:
                diffs.append(0.0)
                continue

            p_curve = torch.log10(p_psd[1:n] + EPS)
            t_curve = torch.log10(t_psd[1:n] + EPS)
            diffs.append(torch.sqrt(((p_curve - t_curve) ** 2).mean()).item())
        dist_per_lead.append(float(torch.tensor(diffs).mean()))

    return {"RAPSD_dist_lead": torch.tensor(dist_per_lead), "RAPSD_dist": float(torch.tensor(dist_per_lead).mean())}
