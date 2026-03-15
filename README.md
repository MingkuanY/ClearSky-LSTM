# ClearSky-LSTM

Precipitation nowcasting via a ConvLSTM–UNet hybrid trained on NEXRAD Level II radar reflectivity.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Data pipeline

Data is not tracked in git, so run the following pipeline locally to test it.

### 1 — Download

Fetches raw NEXRAD Level II files from the public AWS S3 archive (`unidata-nexrad-level2`).
Files land at `data/raw/YYYY/MM/DD/<STATION>/<filename>`.

```bash
# Miami (KAMX) - 15 days of peak convective activity used in experiments
python download_nexrad.py \
    --stations KAMX \
    --start 2022-07-01 --end 2022-07-15 \
    --workers 8 \
    --out data/raw
```

Already-downloaded files are skipped automatically.
Use a larger `--workers` value if your network and disk can keep up; the downloader
now fetches files concurrently instead of one-by-one.

### 2 — Cache (run once, before training)

Converts each raw binary file to a `float32` `.npy` reflectivity grid
(256×256 px, ±64 km, raw dBZ values). This eliminates the ~1-3 s/scan
pyart gridding cost from every training step.

```bash
python cache_nexrad.py            # uses all CPU cores by default
python cache_nexrad.py --workers 4 --stations KAMX   # limit cores/stations
```

Cached files land at `data/cache/YYYY/MM/DD/<STATION>/<filename>.npy`.
Caching is also idempotent.

### 3 — Verify (sanity check)

Parses a handful of files and plots the resulting reflectivity grids to
confirm the pipeline is working before committing to a full training run.

```bash
python visualize_samples.py --station KAMX --n 6
python visualize_samples.py --station KAMX --out check.png   # save to file
```

You should see mostly white (clear air) with blue->green->yellow->red patches
where precipitation is present.

---

## Using the dataset

```python
from data import NEXRADDataset

ds = NEXRADDataset(
    raw_root="data/raw",
    stations=["KAMX"],
    t_in=6,           # past frames fed to encoder - x: [T_in,  1, 256, 256]
    t_out=6,          # future frames to predict   - y: [T_out, 1, 256, 256]
    cache_root="data/cache",   # omit to use pyart directly (slow)
)
x, y = ds[0]   # x: [6, 1, 256, 256], y: [6, 1, 256, 256], values in [0, 1]
```

Each frame is normalised to `[0, 1]` from the standard NEXRAD dBZ range `[−32, 70]`.

---

## Citation

NEXRAD data: NOAA National Weather Service Radar Operations Center (1991).
_NOAA Next Generation Radar (NEXRAD) Level 2 Base Data._
doi:[10.7289/V5W9574V](https://doi.org/10.7289/V5W9574V).
Accessed via [unidata-nexrad-level2](https://registry.opendata.aws/noaa-nexrad/) on AWS S3.
