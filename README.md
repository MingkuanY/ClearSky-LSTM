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
python cache_nexrad.py --stations KAMX --start 2022-07-01 --end 2022-07-15
```

Cached files land at `data/cache/YYYY/MM/DD/<STATION>/<filename>.npy`.
Caching is also idempotent.

Training and testing can run from cache only. Once `data/cache` is populated,
`clearsky_lstm.py` uses cached-only loading and will fail fast if required cache
files are missing instead of falling back to raw parsing.

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
    interval=0,       # 0 = consecutive frames, 1 = skip one between frames
    cache_root="data/cache",
    cache_only=True,
)
x, y = ds[0]   # x: [6, 1, 256, 256], y: [6, 1, 256, 256], values in [0, 1]
```

Each frame is normalised to `[0, 1]` from the standard NEXRAD dBZ range `[−32, 70]`.

`interval` controls spacing within each sample while the dataset still slides by
one frame at a time. With `t_in=2`, `t_out=2`:

- `interval=0` uses `[[1,2],[3,4]]`, `[[2,3],[4,5]]`, ...
- `interval=1` uses `[[1,3],[5,7]]`, `[[2,4],[6,8]]`, ...
- `interval=2` uses `[[1,4],[7,10]]`, `[[2,5],[8,11]]`, ...

Windows are built from each station's full time-sorted sequence, so they can
cross date boundaries. If late-night scans from `2022-10-10` are immediately
followed by early-morning scans from `2022-10-11`, one sample can include both
dates as long as those files are present in the cache or raw tree.

`window_stride` controls how densely sliding windows are generated. The default
is `1`, which keeps every possible window start. Larger values reduce dataset
size and training time by skipping overlapping starts. For example, `window_stride=5`
keeps every 5th sample window.

---

## Training

Training now requires explicit train and test date ranges. The training date
range is used to build the dataset that is later split into train/validation,
and the test date range is used only for the final test set.

```bash
python clearsky_lstm.py \
    --model base_network \
    --stations KAMX \
    --t-in 6 \
    --t-out 6 \
    --train-start-date 2024-04-01 \
    --train-end-date 2024-10-31 \
    --test-start-date 2025-04-01 \
    --test-end-date 2025-10-31 \
    --loss-function l1 \
    --window-stride 5 \
    --precision amp
```

### Training options

- `--train-start-date`, `--train-end-date`: required inclusive date range for train/validation data.
- `--test-start-date`, `--test-end-date`: required inclusive date range for held-out test data.
- `--loss-function`: loss used for both training and evaluation.
- `--window-stride`: stride between consecutive sample start indices.
- `--precision`: `amp` or `float32`. Default is `amp`; AMP is enabled only on CUDA and falls back to `float32` on other devices.

### Supported loss functions

- `l1`
- `l2`
- `reflectivity_bmse`
- `reflectivity_bmae`
- `reflectivity_balanced`
- `ssim`

### Precision

Mixed precision (`--precision amp`) is the default for CUDA training and evaluation.
It usually reduces runtime and memory usage. If you need full precision for debugging
or reproducibility checks, use `--precision float32`.

### Output layout

Sample visualizations and metric outputs are grouped by date, model, loss function,
and run ID:

- `samples/{date}/{model}/{loss_function}/{randomid}/`
- `results/{date}/{model}/{loss_function}/{randomid}/`

### Progress and cache error reporting

Training, validation, and test loops show per-iteration progress bars when `tqdm`
is available.

If cached `.npy` files are truncated or corrupted, dataset loading now prints the
exact bad file path and the full sample window so the cache entry can be deleted
or regenerated.

---

## Citation

NEXRAD data: NOAA National Weather Service Radar Operations Center (1991).
_NOAA Next Generation Radar (NEXRAD) Level 2 Base Data._
doi:[10.7289/V5W9574V](https://doi.org/10.7289/V5W9574V).
Accessed via [unidata-nexrad-level2](https://registry.opendata.aws/noaa-nexrad/) on AWS S3.
