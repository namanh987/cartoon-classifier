"""
test.py — Evaluate a trained checkpoint on the test set.

Usage:
    python src/test.py --checkpoint outputs/best_model.pth
    python src/test.py --checkpoint outputs/best_model.pth --data_root data
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.v2 as tfs_2
import sklearn.metrics as metrics

from model import get_model

# Project root = one level above this file (src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Transform  (no augmentation at test time)
# ---------------------------------------------------------------------------

def get_test_transform():
    resnet_weights = models.ResNet18_Weights.DEFAULT
    mean = resnet_weights.transforms().mean
    std  = resnet_weights.transforms().std
    return tfs_2.Compose([
        tfs_2.Resize(256),
        tfs_2.CenterCrop(224),
        tfs_2.ToImage(),
        tfs_2.ToDtype(torch.float32, scale=True),
        tfs_2.Normalize(mean=mean, std=std),
    ]), mean, std


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(model, loader, device):
    """Accumulate predictions over all batches. Returns (preds, targets, logits)."""
    all_preds   = []
    all_targets = []
    all_outputs = []

    for x, y in loader:
        x      = x.to(device)
        logits = model(x)
        preds  = logits.argmax(dim=1).cpu()

        all_preds.append(preds)
        all_targets.append(y)
        all_outputs.append(logits.cpu())

    return torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_outputs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt    = torch.load(args.checkpoint, map_location=device)
    classes = ckpt['classes']
    print(f'[✓] Checkpoint from epoch {ckpt["epoch"]} | val_acc={ckpt["val_acc"]:.2%}')

    # ── Data — test split (never seen during training or model selection) ──
    test_tf, mean, std = get_test_transform()
    test_data   = datasets.ImageFolder(os.path.join(args.data_root, 'test'), transform=test_tf)
    test_loader = data.DataLoader(
        test_data, batch_size=32,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    char_dict = dict(enumerate(test_data.classes))

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_preds, all_targets, all_outputs = run_evaluation(model, test_loader, device)

    accuracy = (all_preds == all_targets).float().mean().item()
    print(f'\n[✓] Test Accuracy: {accuracy:.2%}\n')
    print('Classification Report:')
    print(metrics.classification_report(all_targets, all_preds, target_names=list(char_dict.values())))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(all_targets, all_preds))

    # ── Visualise one random sample ───────────────────────────────────────
    raw_data        = datasets.ImageFolder(os.path.join(args.data_root, 'test'))
    rnd             = np.random.randint(0, len(test_data))
    raw_img, true_label = raw_data[rnd]

    pred_label  = all_preds[rnd].item()
    probs       = torch.softmax(all_outputs[rnd], dim=0)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(raw_img)
    color = 'green' if pred_label == true_label else 'red'
    ax1.set_title(
        f'Predicted: {char_dict[pred_label].capitalize()}\n'
        f'True: {char_dict[true_label].capitalize()}',
        color=color, fontsize=13,
    )
    ax1.axis('off')

    bars = ax2.bar(
        [char_dict[i.item()].capitalize() for i in sorted_idx],
        sorted_probs.numpy(),
        color=['green' if i.item() == true_label else 'steelblue' for i in sorted_idx],
    )
    ax2.set_title('Prediction Probabilities')
    ax2.set_xlabel('Character')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    for bar, prob in zip(bars, sorted_probs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{prob:.1%}', ha='center', fontsize=10)

    fig.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'test_sample.png')
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f'[✓] Sample plot saved → {save_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate cartoon character classifier on test set')
    parser.add_argument('--checkpoint',  required=True,                                          help='Path to best_model.pth')
    parser.add_argument('--data_root',   default=os.path.join(PROJECT_ROOT, 'data'),             help='Folder with test/ subfolder')
    parser.add_argument('--output_dir',  default=os.path.join(PROJECT_ROOT, 'outputs'),          help='Where to save output plots')
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    main(args)