import torch
import torch.nn.functional as F

from data import DBZ_MIN
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


def _masked_field(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(field.dtype)
    return field * valid


_RAPSD_CACHE: dict[tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_rapsd_bins(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (h, w, device)
    cached = _RAPSD_CACHE.get(key)
    if cached is not None:
        return cached

    y = torch.arange(h, device=device, dtype=torch.float32) - (h // 2)
    x = torch.arange(w, device=device, dtype=torch.float32) - (w // 2)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")
    radius_bins = torch.sqrt(x_grid ** 2 + y_grid ** 2).reshape(-1).long()
    counts = torch.bincount(radius_bins)
    _RAPSD_CACHE[key] = (radius_bins, counts)
    return radius_bins, counts


def rapsd_field(field: torch.Tensor, d: float = 1.0, return_freq=True):
    # field: (H, W), real-valued reflectivity field
    if field.ndim != 2:
        raise ValueError("rapsd_field requires 2D input")

    field = field.float()
    h, w = field.shape
    radius_bins, counts = _get_rapsd_bins(h, w, field.device)

    fft = torch.fft.fft2(field)
    psd2d = torch.fft.fftshift((fft.abs() ** 2) / field.numel())
    bin_sums = torch.bincount(radius_bins, weights=psd2d.reshape(-1), minlength=counts.numel())
    bin_psd = bin_sums / counts.clamp(min=1)

    freqs = torch.arange(counts.numel(), device=field.device, dtype=torch.float32) / (max(h, w) * d)
    if return_freq:
        return freqs, bin_psd
    return bin_psd


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
    valid = _valid_mask(target_dbz, mask=mask).to(pred_dbz.dtype)

    batch, lead, h, w = pred_dbz.shape
    pred_fields = (pred_dbz * valid).reshape(batch * lead, h, w).float()
    target_fields = (target_dbz * valid).reshape(batch * lead, h, w).float()

    radius_bins, counts = _get_rapsd_bins(h, w, pred_fields.device)
    n_bins = counts.numel()

    pred_fft = torch.fft.fft2(pred_fields, dim=(-2, -1))
    target_fft = torch.fft.fft2(target_fields, dim=(-2, -1))
    pred_psd = torch.fft.fftshift((pred_fft.abs() ** 2) / (h * w), dim=(-2, -1)).reshape(batch * lead, -1)
    target_psd = torch.fft.fftshift((target_fft.abs() ** 2) / (h * w), dim=(-2, -1)).reshape(batch * lead, -1)

    pred_bin_sums = torch.zeros(batch * lead, n_bins, device=pred_fields.device, dtype=pred_fields.dtype)
    target_bin_sums = torch.zeros_like(pred_bin_sums)
    index = radius_bins.unsqueeze(0).expand(batch * lead, -1)
    pred_bin_sums.scatter_add_(1, index, pred_psd)
    target_bin_sums.scatter_add_(1, index, target_psd)

    counts = counts.to(pred_fields.device, pred_fields.dtype).clamp(min=1)
    pred_curve = pred_bin_sums / counts.unsqueeze(0)
    target_curve = target_bin_sums / counts.unsqueeze(0)

    if n_bins <= 1:
        dist_per_lead = torch.zeros(lead, dtype=torch.float32, device=pred_fields.device)
    else:
        log_diff_sq = (
            torch.log10(pred_curve[:, 1:] + EPS) - torch.log10(target_curve[:, 1:] + EPS)
        ) ** 2
        sample_dist = torch.sqrt(log_diff_sq.mean(dim=1))
        dist_per_lead = sample_dist.reshape(batch, lead).mean(dim=0)

    dist_per_lead_cpu = dist_per_lead.detach().cpu()
    return {
        "RAPSD_dist_lead": dist_per_lead_cpu,
        "RAPSD_dist": float(dist_per_lead_cpu.mean()),
    }
