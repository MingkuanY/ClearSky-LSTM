#!python3

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys


LOSSES = ["l2", "l1", "reflectivity_bmse", "reflectivity_bmae", "ssim"]
MODELS = ["base_network_cand", "smaat_unet"]
INTERVALS = [1, 5, 11]
LEARNING_RATES = [0.0001, 0.001]

COMMON_ARGS = {
    "stations": ["KAMX"],
    "t_in": 5,
    "t_out": 5,
    "batch_size": 16,
    "epochs": 20,
    "window_stride": 5,
    "precision": "amp",
    "train_start_date": "2024-04-01",
    "train_end_date": "2024-10-31",
    "test_start_date": "2025-04-01",
    "test_end_date": "2025-10-31",
    "seed": 13,
    "val_frac": 0.1,
    "num_workers": 4,
    "weight_decay": 0.0,
    "teacher_forcing": 0,
}

CONVLSTM_ARGS = {
    "hidden_ch": [32, 32],
    "num_layers": 2,
}


def lr_slug(lr):
    return f"{lr:g}".replace(".", "p")


def checkpoint_path(model, loss_function, interval, lr):
    name = f"{model}_{loss_function}_int{interval}_lr{lr_slug(lr)}_tin5_tout5.pt"
    return os.path.join("checkpoints", name)


def deterministic_run_id(params):
    run_params = {
        "model": params["model"],
        "loss_function": params["loss_function"],
        "stations": params["stations"],
        "t_in": params["t_in"],
        "t_out": params["t_out"],
        "interval": params["interval"],
        "window_stride": params["window_stride"],
        "val_frac": params["val_frac"],
        "train_start_date": params["train_start_date"],
        "train_end_date": params["train_end_date"],
        "test_start_date": params["test_start_date"],
        "test_end_date": params["test_end_date"],
        "batch_size": params["batch_size"],
        "num_workers": params["num_workers"],
        "epochs": params["epochs"],
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "precision": params["precision"],
        "hidden_ch": params["hidden_ch"],
        "num_layers": params["num_layers"],
        "teacher_forcing": params["teacher_forcing"],
        "seed": params["seed"],
    }
    encoded = json.dumps(run_params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:10]


def build_case(model, loss_function, interval, lr):
    params = {
        **COMMON_ARGS,
        "model": model,
        "loss_function": loss_function,
        "interval": interval,
        "lr": lr,
        "model_out": checkpoint_path(model, loss_function, interval, lr),
    }
    if model == "base_network_cand":
        params.update(CONVLSTM_ARGS)
    else:
        params.update({"hidden_ch": [64, 64, 64], "num_layers": 2})
    params["run_id"] = deterministic_run_id(params)
    return params


def build_command(params, run_stamp, skip_if_complete):
    cmd = [
        sys.executable,
        "clearsky_lstm.py",
        "--loss-function",
        params["loss_function"],
        "--model-out",
        params["model_out"],
        "--model",
        params["model"],
        "--stations",
        *params["stations"],
        "--t-in",
        str(params["t_in"]),
        "--t-out",
        str(params["t_out"]),
        "--batch-size",
        str(params["batch_size"]),
        "--epochs",
        str(params["epochs"]),
        "--lr",
        str(params["lr"]),
        "--interval",
        str(params["interval"]),
        "--window-stride",
        str(params["window_stride"]),
        "--precision",
        params["precision"],
        "--train-start-date",
        params["train_start_date"],
        "--train-end-date",
        params["train_end_date"],
        "--test-start-date",
        params["test_start_date"],
        "--test-end-date",
        params["test_end_date"],
        "--seed",
        str(params["seed"]),
        "--run-stamp",
        run_stamp,
        "--run-id",
        params["run_id"],
    ]
    if params["model"] == "base_network_cand":
        cmd.extend(["--hidden-ch", *[str(ch) for ch in params["hidden_ch"]]])
        cmd.extend(["--num-layers", str(params["num_layers"])])
    if skip_if_complete:
        cmd.append("--skip-if-complete")
    return cmd


def iter_cases():
    for loss_function, model, interval, lr in itertools.product(
        LOSSES, MODELS, INTERVALS, LEARNING_RATES
    ):
        yield build_case(model, loss_function, interval, lr)


def main():
    parser = argparse.ArgumentParser(description="Run the ClearSky LSTM experiment grid.")
    parser.add_argument(
        "--run-stamp",
        default="grid",
        help="Top-level results/samples folder. Use a stable value so completed runs can be skipped later.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--no-skip", action="store_true", help="Run cases even when test_metrics.json already exists.")
    parser.add_argument("--start-at", type=int, default=1, help="1-based case index to start from.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of cases to consider.")
    args = parser.parse_args()

    cases = list(iter_cases())
    total = len(cases)
    selected = cases[args.start_at - 1 :]
    if args.limit is not None:
        selected = selected[: args.limit]

    for offset, params in enumerate(selected, start=args.start_at):
        results_dir = os.path.join(
            "results",
            args.run_stamp,
            params["model"],
            params["loss_function"],
            params["run_id"],
        )
        test_metrics_path = os.path.join(results_dir, "test_metrics.json")
        if not args.no_skip and os.path.exists(test_metrics_path):
            print(f"[{offset}/{total}] skip complete: {test_metrics_path}")
            continue

        cmd = build_command(params, args.run_stamp, skip_if_complete=not args.no_skip)
        print(f"[{offset}/{total}] running {params['model']} {params['loss_function']} interval={params['interval']} lr={params['lr']}")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
