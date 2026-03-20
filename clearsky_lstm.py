# Models
from models.conv_lstm import ConvLSTMForecaster
from models.smaat_unet import SmaAtUNet

# Data loader
from data import NEXRADDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# Data visualization
from collections import defaultdict
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import string

# Blur metric
import cv2

# Metrics
from metrics import (
    regression_metrics,
    contingency_metrics,
    fractions_skill_score,
    rapsd_distance,
)

# Train/test utils
import argparse
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def train_one_epoch(model, loader, optimizer, criterion, device, args):
    """ Training loop for one epoch """
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        if args.model == "base_network":
            pred = model(
                x,
                t_out=y.shape[1],
                teacher_forcing=args.teacher_forcing,
                y=y
            )
        else:
            pred = model(x)

        # These print statements are for testing purposes only
        # They print range of predicted values vs range of ground truth values
        if i == 0:
            print(f"Input Range: {x.min().item():.4f} to {x.max().item():.4f}")
            print(f"Pred Range: {pred.min().item():.4f} to {pred.max().item():.4f}")
            print(f"Target Range: {y.min().item():.4f} to {y.max().item():.4f}")
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, args, epoch=0):
    """ Model evaluate loop (no training) """
    model.eval()
    total_loss = 0.0
    total_blur = 0.0
    n_batches = 0

    # accumulators
    stats = {
        "mae": 0.0,
        "mse": 0.0,
        "rmse": 0.0,
        "RAPSD_dist": 0.0,
    }
    per_lead = defaultdict(lambda: None)

    threshold_keys = ["CSI_20", "POD_20", "FAR_20", "CSI_40", "POD_40", "FAR_40", "CSI_50", "POD_50", "FAR_50"]

    fss_keys = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            if args.model == "base_network":
                pred = model(x, t_out=y.shape[1])
            else:
                pred = model(x)

            loss = criterion(pred, y)

            total_loss += loss.item()
            n_batches += 1

            if i == 0:
                save_comparison(x[0], y[0], pred[0], epoch, i, out_dir=args.sample_dir)
                save_preds_only(pred[0], epoch, i, out_dir=os.path.join(args.sample_dir, "preds"))

            total_blur += compute_blur_score(pred[0])

            r_metrics = regression_metrics(pred, y)
            stats["mae"] += r_metrics["mae"]
            stats["mse"] += r_metrics["mse"]
            stats["rmse"] += r_metrics["rmse"]

            for k in ["mae_lead", "mse_lead", "rmse_lead"]:
                if per_lead[k] is None:
                    per_lead[k] = r_metrics[k].clone()
                else:
                    per_lead[k] += r_metrics[k]

            cont_metrics = contingency_metrics(pred, y)
            for key in threshold_keys:
                if per_lead[key] is None:
                    per_lead[key] = cont_metrics[key].clone()
                else:
                    per_lead[key] += cont_metrics[key]

            fss_metrics = fractions_skill_score(pred, y)
            for key, value in fss_metrics.items():
                if key not in fss_keys:
                    fss_keys.append(key)
                if per_lead[key] is None:
                    per_lead[key] = value.clone()
                else:
                    per_lead[key] += value

            rapsd = rapsd_distance(pred, y)
            stats["RAPSD_dist"] += rapsd["RAPSD_dist"]
            if per_lead.get("RAPSD_dist_lead") is None:
                per_lead["RAPSD_dist_lead"] = rapsd["RAPSD_dist_lead"].clone()
            else:
                per_lead["RAPSD_dist_lead"] += rapsd["RAPSD_dist_lead"]

    avg_loss = total_loss / max(n_batches, 1)
    avg_blur = total_blur / max(n_batches, 1)

    for m in ["mae", "mse", "rmse", "RAPSD_dist"]:
        stats[m] = stats[m] / max(n_batches, 1)

    for k, v in per_lead.items():
        per_lead[k] = (v / max(n_batches, 1)).tolist() if isinstance(v, torch.Tensor) else v

    eval_stats = {
        "loss": avg_loss,
        "blur": avg_blur,
        "mae": stats["mae"],
        "mse": stats["mse"],
        "rmse": stats["rmse"],
        "rapsd_dist": stats["RAPSD_dist"],
        "per_lead": per_lead,
    }

    return eval_stats

def compute_blur_score(preds):
    """
    Computes Laplacian Variance for a sequence of predictions.
    preds: [T, 1, H, W] in range [0, 1]
    """
    # Convert to 8-bit numpy for OpenCV
    # Move to CPU, detach from graph, and scale
    seq = (preds.detach().cpu().numpy() * 255).astype('uint8')
    
    scores = []
    for t in range(seq.shape[0]):
        frame = seq[t, 0] # [H, W]
        score = cv2.Laplacian(frame, cv2.CV_64F).var()
        scores.append(score)
    
    return sum(scores) / len(scores)

def save_comparison(input_frames, target_frames, pred_frames, epoch, batch_idx, out_dir="samples"):
    """
    Saves a PNG comparing the last input frame, ground truth sequence, and predicted sequence.
        input_frames: [T_in, 1, H, W]
        target_frames: [T_out, 1, H, W]
        pred_frames: [T_out, 1, H, W]
    """
    os.makedirs(out_dir, exist_ok=True)
    t_out = target_frames.shape[0]
    
    # Create a figure: 2 rows (Target vs Pred), t_out columns
    fig, axes = plt.subplots(2, t_out, figsize=(t_out * 3, 6))
    
    for t in range(t_out):
        # Top row: Ground Truth
        ax_gt = axes[0, t]
        im_gt = ax_gt.imshow(target_frames[t, 0].cpu().numpy(), vmin=0, vmax=1, cmap='viridis')
        ax_gt.set_title(f"Target T+{t+1}")
        ax_gt.axis('off')
        
        # Bottom row: Prediction
        ax_pred = axes[1, t]
        im_pred = ax_pred.imshow(pred_frames[t, 0].detach().cpu().numpy(), vmin=0, vmax=1, cmap='viridis')
        ax_pred.set_title(f"Pred T+{t+1}")
        ax_pred.axis('off')

    plt.tight_layout()
    plt.savefig(f"{out_dir}/epoch{epoch}_batch{batch_idx}.png")
    plt.close()

def save_preds_only(pred_frames, epoch, batch_idx, out_dir="samples/preds"):
    """
    Saves only the predicted frames as raw images to avoid 
    inflated blur scores from Matplotlib text and borders.
    pred_frames: [T_out, 1, H, W] tensor
    """
    os.makedirs(out_dir, exist_ok=True)
    t_out = pred_frames.shape[0]
    
    # Use [0] to get the first sequence if batch_size > 1
    preds = pred_frames.detach().cpu().numpy() # [T, 1, H, W]

    for t in range(t_out):
        # 1. Normalize and scale to 8-bit
        frame = preds[t, 0]
        frame = np.clip(frame, 0, 1) # Safety clip
        frame_8bit = (frame * 255).astype(np.uint8)

        # 2. Apply Viridis colormap 
        color_frame = cv2.applyColorMap(frame_8bit, cv2.COLORMAP_VIRIDIS)

        # 3. Save each frame
        filename = f"epoch{epoch}_b{batch_idx}_T{t+1}.png"
        cv2.imwrite(os.path.join(out_dir, filename), color_frame)

def main():
    # ---------------- 1. Parse Args ----------------
    ap = argparse.ArgumentParser(description="Radar precipitation nowcasting training")

    # Model choice
    ap.add_argument("--model", type=str, default="base_network", choices=["base_network", "smaat_unet"], help="Model architecture to use")

    # Sequence config
    ap.add_argument("--stations", nargs="+", default=["KAMX"], help="Radar station IDs to use")
    ap.add_argument("--t-in", type=int, default=6, help="Number of past radar frames used as input")
    ap.add_argument("--t-out", type=int, default=6, help="Number of future radar frames to predict")

    # Train/test/val splits
    ap.add_argument("--val-frac", type=float, default=0.1, help="Fraction of dataset used for validation")
    ap.add_argument("--test-frac", type=float, default=0.1, help="Fraction of dataset used for testing")

    # Data loading params
    ap.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    ap.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay")

    # ConvLSTM architecture
    ap.add_argument("--hidden-ch", type=int, nargs="+", default=[64,64,64], help="Number of ConvLSTM hidden channels")
    ap.add_argument("--num-layers", type=int, default=2, help="Number of stacked ConvLSTM layers")
    ap.add_argument("--teacher-forcing", type=float, default=0, help="Probability of using ground truth frame during training")

    # Visualization/outdirs/reproducibility
    ap.add_argument("--model-out", type=str, default="checkpoints/final_model.pt", help="Path to save final model parameters")
    ap.add_argument("--seed", type=int, default=13, help="Random seed")

    print("Starting model...")
    args = ap.parse_args()

    # Enforce mandatory output folder for metrics and samples
    stamp = datetime.datetime.now().strftime("%Y%m%d")
    random_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    subpath = os.path.join(stamp, args.model, random_id)

    args.sample_dir = os.path.join("samples", subpath)
    args.results_dir = os.path.join("results", subpath)

    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Arguments parsed! sample_dir={args.sample_dir} results_dir={args.results_dir}")

    # ---------------- 2. Data loading ----------------
    print("Building dataset...")
    ds = NEXRADDataset(
        raw_root="data/raw",
        stations=args.stations,
        t_in=args.t_in,               # past frames fed to encoder - x: [T_in,  1, 256, 256]
        t_out=args.t_out,             # future frames to predict   - y: [T_out, 1, 256, 256]
        cache_root="data/cache",      # omit to use pyart directly (slow)
    )

    # Split data into train/val/test sets
    n = len(ds)
    n_val = int(args.val_frac * n)
    n_test = int(args.test_frac * n)
    n_train = n - n_val - n_test
    print("Train/val/test split complete!")

    torch.manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Only use this below commented line if testing on the first batch only
    # train_loader = [next(iter(train_loader))]

    # ------------ 3. Build selected model ------------
    if args.model == "smaat_unet":
        model = SmaAtUNet(in_channels=args.t_in, out_channels=args.t_out)
    elif args.model == "base_network":
        model = ConvLSTMForecaster(hidden_ch=args.hidden_ch, num_layers=args.num_layers)
    else:
        raise ValueError("Invalid model name")
    
    print("Model built!")
    

    # ------------ 4. Train selected model ------------
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = torch.nn.L1Loss()

    print("Beginning training...")
    print("Note: higher blur score = sharper image!")
    history = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args)
        val_stats = evaluate(model, val_loader, criterion, device, args, epoch)

        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "loss": float(val_stats["loss"]),
            "blur": float(val_stats["blur"]),
            "mae": float(val_stats["mae"]),
            "mse": float(val_stats["mse"]),
            "rmse": float(val_stats["rmse"]),
            "rapsd_dist": float(val_stats["rapsd_dist"]),
            "per_lead": val_stats["per_lead"],
        }

        history.append(epoch_results)
        with open(os.path.join(args.results_dir, "train_val_metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | train: {train_loss:.3f} "
            f"| val: {val_stats['loss']:.3f} | blur: {val_stats['blur']:.3f} "
            f"| MAE: {val_stats['mae']:.3f} | RMSE: {val_stats['rmse']:.3f} | RAPSD-dist: {val_stats['rapsd_dist']:.3f}"
        )

    # ------------ 5. Evaluate model ------------
    print("Evaluating model...")
    test_stats = evaluate(model, test_loader, criterion, device, args)

    with open(os.path.join(args.results_dir, "test_metrics.json"), "w") as f:
        json.dump({"test": test_stats}, f, indent=2)

    print(
        f"Final loss on test set: {test_stats['loss']:.3f} "
        f"// final blur on test set: {test_stats['blur']:.3f} "
        f"// MAE: {test_stats['mae']:.3f} // RMSE: {test_stats['rmse']:.3f} "
        f"// RAPSD-dist: {test_stats['rapsd_dist']:.3f}"
    )

    model_dir = os.path.dirname(args.model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"Saved final model parameters to {args.model_out}")

    print("Done!")

if __name__ == "__main__":
    main()

"""
    Sample commands for running the script:

    python clearsky_lstm.py \
    --model base_network \
    --stations KAMX \
    --t-in 6 \
    --t-out 6 \
    --batch-size 8 \
    --epochs 20 \
    --lr 0.001 \
    --hidden-ch 64 64\
    --num-layers 2 \
    --teacher-forcing 0.5 \
    --save-samples \
    --model-out "checkpoints/baseline_final.pt"
    
    python clearsky_lstm.py \
    --model smaat_unet \
    --stations KAMX \
    --t-in 6 \
    --t-out 6 \
    --batch-size 8 \
    --epochs 20 \
    --lr 0.001 \
    --save-samples \
    --model-out "checkpoints/smaat_unet_final.pt"
"""
