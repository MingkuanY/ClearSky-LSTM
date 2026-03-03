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
from datetime import date, timedelta

import boto3
import botocore

BUCKET = "unidata-nexrad-level2"
REGION = "us-east-1"


def make_s3_client():
    return boto3.client(
        "s3",
        region_name=REGION,
        config=botocore.client.Config(signature_version=botocore.UNSIGNED),
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


def download_key(s3, key: str, out_root: str) -> str:
    """Download *key* to *out_root*/<key> and return the local path."""
    local_path = os.path.join(out_root, key)
    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(
        Bucket=BUCKET,
        Key=key,
        Filename=local_path,
    )
    return local_path


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


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
    args = parser.parse_args()

    s3 = make_s3_client()
    total_downloaded = 0
    total_skipped = 0

    for station in args.stations:
        print(f"\n=== Station: {station} ===")
        for key in iter_keys(s3, station, args.start, args.end):
            local = os.path.join(args.out, key)
            if os.path.exists(local):
                total_skipped += 1
                continue
            print(f"  Downloading {key} ...", end=" ", flush=True)
            download_key(s3, key, args.out)
            print("done")
            total_downloaded += 1

    print(f"\nFinished. Downloaded: {total_downloaded}  Skipped (already exist): {total_skipped}")


if __name__ == "__main__":
    main()
