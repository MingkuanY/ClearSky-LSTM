"""
download_nexrad.py
------------------
Download NEXRAD Level II files from the public AWS S3 archive to local.

Bucket : unidata-nexrad-level2  (us-east-1, no credentials required)
Layout : s3://<bucket>/<STATION>/<YYYY>/<MM>/<DD>/<filename>

How to use (examples):
-----
    # Single station, single day
    python download_nexrad.py --stations KAMX --start 2022-07-01 --end 2022-07-01
    
    # RECOMMENDED: Single station, multiple days (this is for Miami in the summer, took around 25 min for me to download)
    python download_nexrad.py --stations KAMX --start 2022-07-01 --end 2022-07-15

    # Multiple stations, full season
    python download_nexrad.py \
        --stations KFTG KAMX KLOT KBMX KSFX \
        --start 2021-04-01 --end 2022-09-30 \
        --out data/raw
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import boto3
import botocore
from botocore.s3.transfer import TransferConfig

BUCKET = "unidata-nexrad-level2"
REGION = "us-east-1"
DEFAULT_WORKERS = 8


def make_s3_client(max_pool_connections: int = DEFAULT_WORKERS):
    return boto3.client(
        "s3",
        region_name=REGION,
        config=botocore.client.Config(
            signature_version=botocore.UNSIGNED,
            max_pool_connections=max_pool_connections,
        ),
    )


def iter_keys(s3, station: str, start: date, end: date):
    """Yield all S3 keys for *station* between *start* and *end* (inclusive).

    The NEXRAD Level II bucket uses date-first keys:
        YYYY/MM/DD/<STATION>/<filename>
    """
    current = start
    while current <= end:
        prefix = f"{current.year:04d}/{current.month:02d}/{current.day:02d}/{station}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]
        current += timedelta(days=1)


def download_key(s3, key: str, out_root: str, transfer_config: TransferConfig | None = None) -> str:
    """Download *key* to *out_root*/<key> and return the local path."""
    local_path = os.path.join(out_root, key)
    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(
        Bucket=BUCKET,
        Key=key,
        Filename=local_path,
        Config=transfer_config,
    )
    return local_path


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def plan_downloads(s3, stations: list[str], start: date, end: date, out_root: str) -> tuple[list[str], int]:
    """Return keys missing locally plus the count already present."""
    keys_to_download: list[str] = []
    skipped = 0

    for station in stations:
        print(f"\n=== Station: {station} ===")
        station_missing = 0
        station_skipped = 0

        for key in iter_keys(s3, station, start, end):
            local = os.path.join(out_root, key)
            if os.path.exists(local):
                skipped += 1
                station_skipped += 1
                continue
            keys_to_download.append(key)
            station_missing += 1

        print(f"  queued: {station_missing}  skipped: {station_skipped}")

    return keys_to_download, skipped


def download_keys_bulk(s3, keys: list[str], out_root: str, workers: int) -> int:
    """Download keys concurrently and return the number completed."""
    if not keys:
        return 0

    transfer_config = TransferConfig(
        max_concurrency=min(4, max(1, workers)),
        use_threads=True,
    )
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_key = {
            executor.submit(download_key, s3, key, out_root, transfer_config): key
            for key in keys
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            future.result()
            completed += 1
            print(f"  [{completed}/{len(keys)}] downloaded {key}")

    return completed


def main():
    parser = argparse.ArgumentParser(description="Download NEXRAD Level II files from AWS S3.")
    parser.add_argument(
        "--stations",
        nargs="+",
        required=True,
        metavar="STATION",
        help="WSR-88D station IDs, e.g. KFTG KAMX KLOT",
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        required=True,
        metavar="YYYY-MM-DD",
        help="First date to download (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        required=True,
        metavar="YYYY-MM-DD",
        help="Last date to download (inclusive).",
    )
    parser.add_argument(
        "--out",
        default="data/raw",
        metavar="DIR",
        help="Local root directory for downloaded files (default: data/raw).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Number of concurrent downloads (default: {DEFAULT_WORKERS}).",
    )
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must be on or after --start.")
    if args.workers < 1:
        parser.error("--workers must be at least 1.")

    s3 = make_s3_client(max_pool_connections=args.workers)
    keys_to_download, total_skipped = plan_downloads(
        s3, args.stations, args.start, args.end, args.out
    )

    print(
        f"\nStarting bulk download with {args.workers} workers "
        f"for {len(keys_to_download)} files."
    )
    total_downloaded = download_keys_bulk(s3, keys_to_download, args.out, args.workers)

    print(f"\nFinished. Downloaded: {total_downloaded}  Skipped (already exist): {total_skipped}")


if __name__ == "__main__":
    main()
