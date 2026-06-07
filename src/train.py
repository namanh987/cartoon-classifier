"""
train.py — Fine-tune ResNet-18 for cartoon character classification.

Usage:
    python src/train.py                          # uses data/ folder by default
    python src/train.py --data_root /path/to/data
    python src/train.py --epochs 50 --lr 0.0005
"""

import argparse
import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.v2 as tfs_2
import tqdm

from model import get_model

# Project root = one level above this file (src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms():
    """Return (train_transform, val_transform) using ResNet-18 normalisation stats."""
    resnet_weights = models.ResNet18_Weights.DEFAULT
    mean = resnet_weights.transforms().mean
    std  = resnet_weights.transforms().std

    train_tf = tfs_2.Compose([
        tfs_2.Resize(256),
        tfs_2.RandomCrop(224),
        tfs_2.RandomHorizontalFlip(),
        tfs_2.RandomRotation(10),
        tfs_2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        tfs_2.ToImage(),
        tfs_2.ToDtype(torch.float32, scale=True),
        tfs_2.Normalize(mean=mean, std=std),
    ])

    val_tf = tfs_2.Compose([
        tfs_2.Resize(256),
        tfs_2.CenterCrop(224),
        tfs_2.ToImage(),
        tfs_2.ToDtype(torch.float32, scale=True),
        tfs_2.Normalize(mean=mean, std=std),
    ])

    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_func, device, epoch, epochs):
    """One training pass. Returns mean loss over all batches."""
    model.train()
    loss_mean = 0.0
    pbar = tqdm.tqdm(loader, desc=f'Epoch [{epoch}/{epochs}] train')

    for i, (x, y) in enumerate(pbar, start=1):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss   = loss_func(logits, y)
        loss.backward()
        optimizer.step()

        loss_mean += (loss.item() - loss_mean) / i
        pbar.set_postfix(loss=f'{loss_mean:.4f}')

    return loss_mean


@torch.no_grad()
def evaluate(model, loader, loss_func, device):
    """Evaluate on loader. Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)

        total_loss += loss_func(logits, y).item() * y.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += y.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(loss_train, loss_val, acc_val, save_path):
    epochs_range = np.arange(1, len(loss_train) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, loss_train, label='Train loss')
    ax1.plot(epochs_range, loss_val,   label='Val loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss curves'); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs_range, acc_val, color='green', label='Val accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[✓] Plot saved → {save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[✓] Device: {device}')

    # ── Data ──────────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms()

    train_data = datasets.ImageFolder(os.path.join(args.data_root, 'train'), transform=train_tf)
    val_data   = datasets.ImageFolder(os.path.join(args.data_root, 'val'),   transform=val_tf)

    print(f'[✓] Classes ({len(train_data.classes)}): {train_data.classes}')
    print(f'    Train: {len(train_data)} images  |  Val: {len(val_data)} images')

    train_loader = data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = data.DataLoader(
        val_data, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(num_classes=len(train_data.classes)).to(device)

    # ── Optimiser + scheduler + loss ──────────────────────────────────────
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_func = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    loss_train_hist, loss_val_hist, acc_val_hist = [], [], []
    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    for epoch in range(1, args.epochs + 1):
        train_loss        = train_one_epoch(model, train_loader, optimizer, loss_func, device, epoch, args.epochs)
        val_loss, val_acc = evaluate(model, val_loader, loss_func, device)
        scheduler.step()

        loss_train_hist.append(train_loss)
        loss_val_hist.append(val_loss)
        acc_val_hist.append(val_acc)

        print(
            f'Epoch {epoch:>3}/{args.epochs} | '
            f'train_loss={train_loss:.4f} | '
            f'val_loss={val_loss:.4f} | '
            f'val_acc={val_acc:.2%} | '
            f'lr={scheduler.get_last_lr()[0]:.2e}'
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0

            torch.save(
                {
                    'epoch':      epoch,
                    'state_dict': best_state,
                    'val_loss':   best_val_loss,
                    'val_acc':    val_acc,
                    'classes':    train_data.classes,
                },
                os.path.join(args.output_dir, 'best_model.pth'),
            )
            print(f'  [✓] Checkpoint saved (val_loss={best_val_loss:.4f})')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'[!] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)')
                break

    # ── Final summary ─────────────────────────────────────────────────────
    save_plots(
        loss_train_hist, loss_val_hist, acc_val_hist,
        save_path=os.path.join(args.output_dir, 'training_history.png'),
    )
    print(f'\n[✓] Best val_loss : {best_val_loss:.4f}')
    print(f'[✓] Best val_acc  : {max(acc_val_hist):.2%}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train cartoon character classifier')
    parser.add_argument('--data_root',   default=os.path.join(PROJECT_ROOT, 'data'),    help='Folder with train/ and val/ subfolders')
    parser.add_argument('--output_dir',  default=os.path.join(PROJECT_ROOT, 'outputs'), help='Where to save checkpoints and plots')
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--patience',    type=int,   default=15,  help='Early stopping patience')
    parser.add_argument('--num_workers', type=int,   default=2,   help='Set 0 on Windows if DataLoader errors')
    args = parser.parse_args()

    main(args)