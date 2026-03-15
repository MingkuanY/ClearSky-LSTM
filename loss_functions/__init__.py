import torch

from data import DBZ_MAX, DBZ_MIN

# https://proceedings.neurips.cc/paper_files/paper/2017/file/a6db4ed04f1621a119799fd3d7545d3d-Paper.pdf
"""
We perform input: reflectivity -> output: reflectivity.

# Caution: In the original paper, authors converted reflectivity to rainfall.

First, reflectivity -> pixel value (floor(255 * (dBZ + 10) / 70 + 0.5)) and clip (0 <= x <= 255).
---
The radar reflectivity values are converted to rainfall intensity values (mm/h) using the Z-R relationship: dBZ = 10 log a + 10b log R where R is
the rain-rate level, a = 58.53 and b = 1.56. The overall statistics and the average monthly rainfall
distribution of the HKO-7 dataset are given in the appendix
"""

def denormalize_dbz(x_norm: torch.Tensor) -> torch.Tensor:
    return x_norm * (DBZ_MAX - DBZ_MIN) + DBZ_MIN

def reflectivity_weights(
    target_norm_dbz: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    target_norm_dbz: (..., H, W) reflectivity in dBZ
    mask: same shape or broadcastable, 1=valid, 0=invalid

    returns:
        same-shape weight tensor
    """
    target_dbz = denormalize_dbz(target_norm_dbz)
    
    w = torch.ones_like(target_dbz)

    """
    https://www.noaa.gov/jetstream/reflectivity
    -35 <= dbz <= 0 extremely light (drizzle / snow)
    0 <= dbz < 20 very light precipitation or general clutter
    20 <= dbz < 40 light precipitation
    40 <= dbz < 50 moderate precipitation
    50 <= dbz < 65 heavy precipitation or some hail
    dbz >= 65 extremely heavy precipitation including water-coated hail
    """
    
    w = torch.where((target_dbz >= 0.0) & (target_dbz < 20.0),
                    torch.full_like(w, 2.0), w)
    w = torch.where((target_dbz >= 20.0) & (target_dbz < 40.0),
                    torch.full_like(w, 5.0), w)
    w = torch.where((target_dbz >= 40.0) & (target_dbz < 50.0),
                    torch.full_like(w, 10.0), w)
    w = torch.where((target_dbz >= 50.0) & (target_dbz < 65.0),
                    torch.full_like(w, 20.0), w)
    w = torch.where(target_dbz >= 65.0,
                    torch.full_like(w, 50.0), w)

    if mask is not None:
        w = w * mask.to(w.dtype)

    return w

class ReflectivityBMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_dbz: torch.Tensor,
        target_dbz: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w = reflectivity_weights(target_dbz, mask)
        loss_map = w * (pred_dbz - target_dbz) ** 2

        # paper-style: divide by number of frames, not by sum of weights
        num_frames = target_dbz.numel() // (target_dbz.shape[-2] * target_dbz.shape[-1])
        return loss_map.sum() / num_frames


class ReflectivityBMAELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_dbz: torch.Tensor,
        target_dbz: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w = reflectivity_weights(target_dbz, mask)
        loss_map = w * (pred_dbz - target_dbz).abs()

        num_frames = target_dbz.numel() // (target_dbz.shape[-2] * target_dbz.shape[-1])
        return loss_map.sum() / num_frames

# Not paper based, just a combination of two
class ReflectivityBalancedLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bmse = ReflectivityBMSELoss()
        self.bmae = ReflectivityBMAELoss()

    def forward(
        self,
        pred_dbz: torch.Tensor,
        target_dbz: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (
            self.alpha * self.bmse(pred_dbz, target_dbz, mask)
            + self.beta * self.bmae(pred_dbz, target_dbz, mask)
        )

# https://arxiv.org/pdf/1511.08861
# https://github.com/NVlabs/PL4NN
# MS-SSIM + L1
class PerceptualLoss:
    pass



