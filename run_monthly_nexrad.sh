#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_monthly_nexrad.sh 2025-04 2025-10
#
# Arguments:
#   $1 = start month (YYYY-MM)
#   $2 = end month   (YYYY-MM)

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <start-month: YYYY-MM> <end-month: YYYY-MM>"
  exit 1
fi

START_MONTH="$1"
END_MONTH="$2"
STATION="KAMX"

# Basic format check
if [[ ! "$START_MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]] || [[ ! "$END_MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
  echo "Error: both arguments must be in YYYY-MM format"
  exit 1
fi

current="${START_MONTH}-01"
final="${END_MONTH}-01"

while [[ "$current" < "$(date -d "$final +1 month" +%F)" ]]; do
  year=$(date -d "$current" +%Y)
  month=$(date -d "$current" +%m)

  month_start=$(date -d "$current" +%F)
  month_end=$(date -d "$current +1 month -1 day" +%F)

  echo "Processing ${year}-${month}: ${month_start} ~ ${month_end}"

  python download_nexrad.py \
    --stations "$STATION" \
    --start "$month_start" \
    --end "$month_end" \
    --workers 8 \
    --out data/raw

  python cache_nexrad.py \
    --stations "$STATION" \
    --start "$month_start" \
    --end "$month_end"

  rm -rf "data/raw/${year}/${month}"

  current=$(date -d "$current +1 month" +%F)
done
