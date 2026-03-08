# Models
from models.conv_lstm import ConvLSTMForecaster
from models.smaat_unet import SmaAtUNet

# Data loader
from data import NEXRADDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# Train/test utils
import argparse
import torch

def train_one_epoch(model, loader, optimizer, criterion, device, args):
    """ Training loop for one epoch """
    model.train()
    total_loss = 0

    for x, y in loader:
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
            pred = model(x)  # adjust later for SmaAtUNet

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, args):
    """ Model evaluate loop (no training) """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if args.model == "base_network":
                pred = model(x, t_out=y.shape[1])
            else:
                pred = model(x)  # adjust later for SmaAtUNet output shape

            loss = criterion(pred, y)
            total_loss += loss.item()

    return total_loss / len(loader)

# save checkpoints, metrics, and sample predictions


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
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay")

    # ConvLSTM architecture
    ap.add_argument("--hidden-ch", type=int, default=64, help="Number of ConvLSTM hidden channels")
    ap.add_argument("--num-layers", type=int, default=2, help="Number of stacked ConvLSTM layers")
    ap.add_argument("--teacher-forcing", type=float, default=0.5, help="Probability of using ground truth frame during training")

    # Visualization/outdirs/reproducibility
    ap.add_argument("--save-samples", action="store_true", help="Save prediction visualizations")
    ap.add_argument("--sample-dir", type=str, default="samples", help="Directory for saving prediction samples")
    ap.add_argument("--seed", type=int, default=13, help="Random seed")

    args = ap.parse_args()


    # ---------------- 2. Data loading ----------------
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

    torch.manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)


    # ------------ 3. Build selected model ------------
    if args.model == "smaat_unet":
        model = SmaAtUNet()
    else:
        model = ConvLSTMForecaster(hidden_ch=args.hidden_ch, num_layers=args.num_layers)
    

    # ------------ 4. Train selected model ------------
    device = "cpu"

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args)
        val_loss = evaluate(model, val_loader, criterion, device, args)
        print(f"Epoch {epoch + 1}/{args.epochs} | train: {train_loss:.3f} | val: {val_loss:.3f}")

    # ------------ 4. Evaluate model ------------
    test_loss = evaluate(model, test_loader, criterion, device, args)
    print(f"Final loss on test set: {test_loss:.3f}")

if __name__ == "__main__":
    main()

"""
    python clearsky_lstm.py \
    --model base_network \
    --stations KAMX \
    --t-in 6 \
    --t-out 6 \
    --batch-size 8 \
    --epochs 20 \
    --lr 0.001 \
    --hidden-ch 64 \
    --num-layers 2 \
    --teacher-forcing 0.5
"""