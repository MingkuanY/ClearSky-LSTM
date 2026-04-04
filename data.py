"""
Two usage modes
---------------
Uncached  (slow, for quick tests):
    ds = NEXRADDataset(raw_root="data/raw", stations=["KAMX"])

Cached (fast, for real training — run cache_nexrad.py first):
    ds = NEXRADDataset(raw_root="data/raw", stations=["KAMX"],
                       cache_root="data/cache")

Dataset
-------

  x : [T_in,  1, H, W]  — past frames fed to the encoder
  y : [T_out, 1, H, W]  — future frames the model must predict
"""

import os
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

DBZ_MIN: float = -32.0   # noise floor / fill value (dBZ)
DBZ_MAX: float = 70.0    # maximum observable reflectivity (dBZ)

GRID_SHAPE: tuple[int, int] = (256, 256)
GRID_RADIUS: float = 64_000.0   # metres

def normalize(ref: np.ndarray) -> np.ndarray:
    """Scale reflectivity from [DBZ_MIN, DBZ_MAX] dBZ to [0, 1] float32."""
    return np.clip((ref - DBZ_MIN) / (DBZ_MAX - DBZ_MIN), 0.0, 1.0).astype(np.float32)


def parse_nexrad_file(
    path: str | os.PathLike,
    grid_shape: tuple[int, int] = GRID_SHAPE,
    grid_radius: float = GRID_RADIUS,
) -> np.ndarray:
    """Read one NEXRAD Level II file and return a (H, W) reflectivity array in dBZ.

    path        : Path to a local NEXRAD Level II binary file.
    grid_shape  : Output (H, W) in pixels.
    grid_radius : Half-width of the Cartesian domain in metres.

    Returns
    -------
    ref : float32 ndarray of shape (H, W), values in dBZ.
    """
    import pyart

    H, W = grid_shape
    pixel_m = (2 * grid_radius) / H

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        radar = pyart.io.read_nexrad_archive(str(path))

    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(1, H, W),
        grid_limits=(
            (0.0, 1_000.0),
            (-grid_radius, grid_radius),
            (-grid_radius, grid_radius),
        ),
        grid_resolution=(pixel_m, pixel_m),
        fields=["reflectivity"],
        weighting_function="nearest",
    )

    ref_ma = grid.fields["reflectivity"]["data"][0]
    ref = np.ma.filled(ref_ma, fill_value=DBZ_MIN).astype(np.float32)
    return ref

def _is_scan_file(p: Path) -> bool:
    """True for real scan files; False for NEXRAD metadata stubs and hidden files."""
    name = p.name
    return (
        p.is_file()
        and not name.startswith(".")
        and not name.endswith("_MDM")
        and not name.endswith(".npy")
    )


def _sorted_scan_paths(data_root: str | os.PathLike, station: str) -> list[Path]:
    """Return sorted raw scan paths for *station* under *data_root*.
    """
    root = Path(data_root)
    paths = [p for p in root.rglob(f"{station}/*") if _is_scan_file(p)]
    return sorted(paths)


def _sorted_cache_paths(data_root: str | os.PathLike, station: str) -> list[Path]:
    """Return sorted cached scan paths for *station* under *data_root*."""
    root = Path(data_root)
    paths = [
        p for p in root.rglob(f"{station}/*.npy")
        if p.is_file() and not p.name.startswith(".")
    ]
    return sorted(paths)


def _cache_path_for(raw_path: Path, raw_root: Path, cache_root: Path) -> Path:
    """Derive the .npy cache path corresponding to a raw scan path."""
    rel = raw_path.relative_to(raw_root)
    return cache_root / rel.parent / (rel.name + ".npy")
class NEXRADDataset(Dataset):
    """Sliding-window sequence dataset over one or more NEXRAD stations.

    raw_root    : Root of the downloaded raw files
                  (data/raw/YYYY/MM/DD/<STATION>/<file>).
    stations    : List of station IDs, e.g. ["KAMX", "KFTG"].
    t_in        : Number of past frames per input sequence.
    t_out       : Number of future frames to predict.
    cache_root  : If provided, load pre-computed .npy grids from here instead
                  of running pyart on every __getitem__ call.
                  Run cache_nexrad.py once to populate this directory.
    cache_only  : If True, build windows from cached .npy files and never fall
                  back to raw files. If raw_root is missing and cache_root is
                  provided, this mode is enabled automatically.
    grid_shape  : Spatial resolution (H, W) — must match what was used when
                  building the cache (default matches cache_nexrad.py).
    grid_radius : Cartesian domain half-width in metres.
    transform   : Optional callable applied to (x, y) after stacking.
    """

    def __init__(
        self,
        raw_root: str | os.PathLike,
        stations: Sequence[str],
        t_in: int = 6,
        t_out: int = 6,
        cache_root: str | os.PathLike | None = None,
        cache_only: bool = False,
        grid_shape: tuple[int, int] = GRID_SHAPE,
        grid_radius: float = GRID_RADIUS,
        transform=None,
    ):
        self.t_in = t_in
        self.t_out = t_out
        self.window = t_in + t_out
        self.raw_root   = Path(raw_root)
        self.cache_root = Path(cache_root) if cache_root else None
        self.cache_only = cache_only or (
            self.cache_root is not None and not self.raw_root.exists()
        )
        self.grid_shape = grid_shape
        self.grid_radius = grid_radius
        self.transform = transform

        self._windows: list[tuple[list[Path], int]] = []

        for station in stations:
            if self.cache_only:
                if self.cache_root is None:
                    raise ValueError("cache_only=True requires cache_root to be set.")
                paths = _sorted_cache_paths(self.cache_root, station)
            else:
                paths = _sorted_scan_paths(self.raw_root, station)
            n = len(paths)
            if n == 0:
                root = self.cache_root if self.cache_only else self.raw_root
                mode_hint = (
                    "Run cache_nexrad.py first."
                    if self.cache_only
                    else "Run download_nexrad.py first."
                )
                print(f"Warning: no scans found for {station} under {root}. {mode_hint}")
                continue
            if n < self.window:
                print(
                    f"Warning: {station} has only {n} scans "
                    f"(need {self.window}); skipping."
                )
                continue
            for start in range(n - self.window + 1):
                self._windows.append((paths, start))

        if not self._windows:
            missing_root = self.cache_root if self.cache_only else self.raw_root
            raise RuntimeError(
                f"No valid windows found. Check that {missing_root} is populated."
            )

    def _load_frame(self, path: Path) -> np.ndarray:
        """Return a (H, W) float32 array in dBZ, using cache when available."""
        if self.cache_only:
            return np.load(str(path))

        raw_path = path
        if self.cache_root is not None:
            cache_path = _cache_path_for(raw_path, self.raw_root, self.cache_root)
            if cache_path.exists():
                return np.load(str(cache_path))
        return parse_nexrad_file(raw_path, self.grid_shape, self.grid_radius)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        paths, start = self._windows[idx]
        window_paths = paths[start : start + self.window]

        frames: list[torch.Tensor] = []
        for p in window_paths:
            ref  = self._load_frame(p)
            norm = normalize(ref)
            frames.append(torch.from_numpy(norm).unsqueeze(0))  # (1, H, W)

        x = torch.stack(frames[: self.t_in])   # [T_in,  1, H, W]
        y = torch.stack(frames[self.t_in :])   # [T_out, 1, H, W]

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y
