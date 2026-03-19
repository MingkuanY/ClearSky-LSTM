# Models
from models.conv_lstm import ConvLSTMForecaster
from models.smaat_unet import SmaAtUNet

# Data loader
from data import NEXRADDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# Data visualization
import matplotlib.pyplot as plt
import numpy as np
import os

# Blur metric
import cv2

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
    total_loss = 0
    total_blur = 0

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
            if args.save_samples and i == 0:
                save_comparison(x[0], y[0], pred[0], epoch, i, out_dir=args.sample_dir)
                save_preds_only(pred[0], epoch, i, out_dir=os.path.join(args.sample_dir, "preds"))
                total_blur += compute_blur_score(pred[0])
            
    avg_blur = total_blur / len(loader)
    avg_loss = total_loss / len(loader)
    return avg_loss, avg_blur

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
    ap.add_argument("--save-samples", action="store_true", help="Save prediction visualizations")
    ap.add_argument("--sample-dir", type=str, default="samples", help="Directory for saving prediction samples")
    ap.add_argument("--model-out", type=str, default="checkpoints/final_model.pt", help="Path to save final model parameters")
    ap.add_argument("--seed", type=int, default=13, help="Random seed")

    print("Starting model...")
    args = ap.parse_args()
    print("Arguments parsed!")

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
    else:
        model = ConvLSTMForecaster(hidden_ch=args.hidden_ch, num_layers=args.num_layers)
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
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args)
        val_loss, avg_blur = evaluate(model, val_loader, criterion, device, args, epoch)
        print(f"Epoch {epoch + 1}/{args.epochs} | train: {train_loss:.3f} | val: {val_loss:.3f} | blur: {avg_blur:.3f}")

    # ------------ 5. Evaluate model ------------
    print("Evaluating model...")
    test_loss, test_blur = evaluate(model, test_loader, criterion, device, args)
    print(f"Final loss on test set: {test_loss:.3f} // final blur on test set: {test_blur:.3f}")

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
    --model-out
    
    python clearsky_lstm.py \
    --model smaat_unet \
    --stations KAMX \
    --t-in 6 \
    --t-out 6 \
    --batch-size 8 \
    --epochs 20 \
    --lr 0.001 \
    --save-samples \
    --model-out
"""
