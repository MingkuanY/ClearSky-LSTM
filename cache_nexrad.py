"""
cache_nexrad.py
---------------
One-time preprocessing pass: convert raw NEXRAD Level II files to cached
.npy reflectivity grids so training doesn't pay the pyart gridding cost
every epoch.

    Reads  : data/raw/YYYY/MM/DD/<STATION>/<filename>
    Writes : data/cache/YYYY/MM/DD/<STATION>/<filename>.npy

Each .npy file is a float32 array of shape (H, W) with raw dBZ values
(un-normalised). Normalisation is applied on-the-fly in NEXRADDataset.

Usage
-----
    # Cache everything in data/raw/ using all available CPU cores
    python cache_nexrad.py

    # Cache only specific stations, limit parallelism
    python cache_nexrad.py --stations KAMX KFTG --workers 4

    # Custom paths
    python cache_nexrad.py --raw-root data/raw --cache-root data/cache
"""

import argparse
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np

from data import GRID_RADIUS, GRID_SHAPE, parse_nexrad_file


def _is_scan_file(p: Path) -> bool:
    """True for real scan files; False for metadata stubs and hidden files."""
    name = p.name
    return (
        p.is_file()
        and not name.startswith(".")
        and not name.endswith("_MDM")
        and not name.endswith(".npy")
    )


def _cache_one(job: tuple[Path, Path]) -> tuple[str, str]:
    """Parse one raw scan and save as .npy. Returns (filename, status)."""
    raw_path, cache_path = job

    if cache_path.exists():
        return raw_path.name, "skipped"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = parse_nexrad_file(raw_path, GRID_SHAPE, GRID_RADIUS)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), ref)
        return raw_path.name, "ok"
    except Exception as exc:
        return raw_path.name, f"error: {exc}"


def build_jobs(
    raw_root: Path,
    cache_root: Path,
    stations: list[str] | None,
) -> list[tuple[Path, Path]]:
    """Return sorted list of (raw_path, cache_path) pairs to process."""
    jobs = []
    for raw_path in sorted(raw_root.rglob("*")):
        if not _is_scan_file(raw_path):
            continue
        # Station ID is the directory name immediately above the file:
        # data/raw/YYYY/MM/DD/<STATION>/<file>
        station = raw_path.parent.name
        if stations and station not in stations:
            continue
        rel = raw_path.relative_to(raw_root)
        cache_path = cache_root / rel.parent / (rel.name + ".npy")
        jobs.append((raw_path, cache_path))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-cache NEXRAD raw files as .npy reflectivity grids."
    )
    parser.add_argument("--raw-root",   default="data/raw",   metavar="DIR")
    parser.add_argument("--cache-root", default="data/cache", metavar="DIR")
    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        metavar="STATION",
        help="Limit to these station IDs (default: all found under --raw-root).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        metavar="N",
        help="Number of parallel worker processes (default: CPU count - 1).",
    )
    args = parser.parse_args()

    raw_root   = Path(args.raw_root)
    cache_root = Path(args.cache_root)

    if not raw_root.exists():
        print(f"Error: raw root {raw_root} does not exist. Run download_nexrad.py first.")
        return

    jobs = build_jobs(raw_root, cache_root, args.stations)
    if not jobs:
        print("No scan files found to cache.")
        return

    n_total = len(jobs)
    n_skip  = sum(1 for _, cp in jobs if cp.exists())
    print(
        f"Found {n_total} scan files  "
        f"({n_skip} already cached, {n_total - n_skip} to process)  "
        f"using {args.workers} workers."
    )

    ok = skipped = errors = 0
    with mp.Pool(processes=args.workers) as pool:
        for i, (name, status) in enumerate(
            pool.imap_unordered(_cache_one, jobs), start=1
        ):
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"  ERROR {name}: {status}")
            if i % 100 == 0 or i == n_total:
                print(
                    f"  [{i:>6}/{n_total}]  "
                    f"cached={ok}  skipped={skipped}  errors={errors}"
                )

    print(f"\nDone.  Cached: {ok}  Skipped: {skipped}  Errors: {errors}")


if __name__ == "__main__":
    main()
