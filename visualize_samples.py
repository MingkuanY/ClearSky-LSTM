"""
Quick sanity-check: parse a handful of local NEXRAD files and display the
resulting reflectivity grids side-by-side.

Run this after downloading some data with download_nexrad.py to verify that
the preprocessing pipeline (pyart read → Cartesian grid → normalize) is
working correctly before committing to a full training run.

How to use:
-----
    # Show 6 consecutive scans for one station
    python visualize_samples.py --station KFTG --n 6

    # Show the first scan of every station in data/raw/
    python visualize_samples.py --all-stations --data-root data/raw

    # Save to a file instead of opening a window
    python visualize_samples.py --station KFTG --out check.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from data import parse_nexrad_file, normalize, DBZ_MIN, DBZ_MAX, _sorted_scan_paths

_NWS_COLORS = [
    (0.00, "#ffffff"),   # < 0 dBZ  — clear / noise (white)
    (0.24, "#04e9e7"),   # ~5 dBZ   — very light
    (0.31, "#019ff4"),   # ~10 dBZ  — light
    (0.38, "#0300f4"),   # ~15 dBZ  — light-moderate
    (0.45, "#02fd02"),   # ~20 dBZ  — moderate (green)
    (0.52, "#01c501"),   # ~25 dBZ
    (0.59, "#008e00"),   # ~30 dBZ  — moderate-heavy
    (0.66, "#fdf802"),   # ~35 dBZ  — yellow
    (0.72, "#e5bc00"),   # ~38 dBZ
    (0.79, "#fd9500"),   # ~43 dBZ  — orange
    (0.86, "#fd0000"),   # ~48 dBZ  — red
    (0.93, "#d40000"),   # ~53 dBZ  — dark red
    (1.00, "#bc00bc"),   # 70 dBZ   — purple (extreme)
]
_NWS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nws_reflectivity",
    [(pos, color) for pos, color in _NWS_COLORS],
)


def plot_frames(
    frames: list[np.ndarray],
    titles: list[str],
    out_path: str | None = None,
    normalised: bool = False,
):
    """Plot a list of (H, W) reflectivity arrays in a single figure row.
    
    frames      : List of (H, W) float32 arrays.
    titles      : One title string per frame.
    out_path    : If given, save the figure to this path instead of showing it.
    normalised  : If True, arrays are in [0, 1]; labels are shown in dBZ scale.
                  If False, arrays are in raw dBZ.
    """
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, frame, title in zip(axes, frames, titles):
        if normalised:
            display = frame * (DBZ_MAX - DBZ_MIN) + DBZ_MIN
        else:
            display = frame
        im = ax.imshow(
            display,
            cmap=_NWS_CMAP,
            vmin=DBZ_MIN,
            vmax=DBZ_MAX,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Reflectivity (dBZ)", fontsize=9)
    fig.suptitle("NEXRAD Reflectivity — Preprocessing Check", fontsize=11, fontweight="bold")

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


def load_and_parse(paths: list[Path], normalised: bool = False) -> list[np.ndarray]:
    frames = []
    for p in paths:
        print(f"  Parsing {p.name} ...", end=" ", flush=True)
        ref = parse_nexrad_file(p)
        print(f"  min={ref.min():.1f} dBZ  max={ref.max():.1f} dBZ  "
              f"mean={ref.mean():.1f} dBZ")
        if normalised:
            ref = normalize(ref)
        frames.append(ref)
    return frames


def _discover_stations(data_root: Path) -> list[str]:
    """Return sorted list of station IDs present under *data_root*.

    Files live at  data_root/YYYY/MM/DD/<STATION>/<filename>
    """
    stations: set[str] = set()
    for p in data_root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            parts = p.relative_to(data_root).parts
            if len(parts) >= 2:
                stations.add(parts[-2])
    return sorted(stations)


def main():
    parser = argparse.ArgumentParser(
        description="Visualise parsed NEXRAD reflectivity grids.\n\n"
                    "Files are expected at  <data-root>/YYYY/MM/DD/<STATION>/<filename>\n"
                    "(matching the layout produced by download_nexrad.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--station",
        metavar="STATION",
        help="Show N consecutive scans for this station, e.g. KFTG",
    )
    group.add_argument(
        "--all-stations",
        action="store_true",
        help="Show the first scan of every station found under --data-root.",
    )
    parser.add_argument(
        "--data-root",
        default="data/raw",
        metavar="DIR",
        help="Root directory produced by download_nexrad.py (default: data/raw).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6,
        metavar="N",
        help="Number of consecutive scans to show (used with --station, default: 6).",
    )
    parser.add_argument(
        "--normalised",
        action="store_true",
        help="Display normalised [0,1] arrays (colorbar still shows dBZ scale).",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="Save figure to this path instead of opening a window.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: data root {data_root} does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.station:
        all_paths = _sorted_scan_paths(data_root, args.station)
        if not all_paths:
            print(
                f"Error: no scans found for station {args.station} under {data_root}.\n"
                f"Run: python download_nexrad.py --stations {args.station} --start YYYY-MM-DD --end YYYY-MM-DD",
                file=sys.stderr,
            )
            sys.exit(1)
        paths = all_paths[: args.n]
        titles = [p.name for p in paths]
        print(f"Loading {len(paths)} scans for {args.station} from {data_root}:")
        frames = load_and_parse(paths, normalised=args.normalised)
        plot_frames(frames, titles, out_path=args.out, normalised=args.normalised)

    else:
        stations = _discover_stations(data_root)
        if not stations:
            print(f"Error: no station data found under {data_root}.", file=sys.stderr)
            sys.exit(1)
        all_paths_list: list[Path] = []
        titles: list[str] = []
        for station in stations:
            scans = _sorted_scan_paths(data_root, station)
            if scans:
                print(f"  First scan for {station}: {scans[0].name}")
                all_paths_list.append(scans[0])
                titles.append(f"{station}\n{scans[0].name}")
        frames = load_and_parse(all_paths_list, normalised=args.normalised)
        plot_frames(frames, titles, out_path=args.out, normalised=args.normalised)


if __name__ == "__main__":
    main()
