"""
inference.py — Run inference on a single image or a folder, with optional Grad-CAM.

Usage:
    python inference.py --checkpoint outputs/best_model.pth --image photo.jpg
    python inference.py --checkpoint outputs/best_model.pth --image photo.jpg --gradcam
    python inference.py --checkpoint outputs/best_model.pth --folder test_images/ --gradcam --topk 3
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.v2 as tfs_2
from PIL import Image


from src.model import get_model

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def get_transform():
    resnet_weights = models.ResNet18_Weights.DEFAULT
    mean = resnet_weights.transforms().mean
    std  = resnet_weights.transforms().std
    tf = tfs_2.Compose([
        tfs_2.Resize(256),
        tfs_2.CenterCrop(224),
        tfs_2.ToImage(),
        tfs_2.ToDtype(torch.float32, scale=True),
        tfs_2.Normalize(mean=mean, std=std),
    ])
    return tf


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    ckpt    = torch.load(checkpoint_path, map_location=device)
    classes = ckpt['classes']
    model   = get_model(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'[✓] Loaded checkpoint — epoch {ckpt["epoch"]} | val_acc={ckpt["val_acc"]:.2%}')
    print(f'[✓] Classes: {classes}')
    return model, classes


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al. 2017).
    Hooks into ResNet-18's final conv block (layer4) to produce a saliency map
    showing which image regions drove the prediction.
    """

    def __init__(self, model: nn.Module):
        self.model      = model
        self._features  = None
        self._gradients = None
        model.layer4[-1].register_forward_hook(self._save_features)
        model.layer4[-1].register_full_backward_hook(self._save_gradients)

    def _save_features(self, _m, _i, output):
        self._features = output.detach()

    def _save_gradients(self, _m, _gi, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self._features).sum(dim=1)).squeeze()
        cam     = cam.cpu().numpy()

        # Resize to input resolution
        from PIL import Image as _I
        cam = np.array(_I.fromarray(cam).resize((x.shape[3], x.shape[2]), _I.BILINEAR))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heatmap = (plt.get_cmap('jet')(cam)[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).resize(img.size, Image.BILINEAR)
    blended = (1 - alpha) * np.array(img.convert('RGB'), dtype=np.float32) \
            + alpha       * np.array(heatmap, dtype=np.float32)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def predict(model, classes, img_path, transform, device, topk=1):
    img    = Image.open(img_path).convert('RGB')
    x      = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top_probs, top_idxs = probs.topk(min(topk, len(classes)))
    return img, [(classes[i], p) for i, p in zip(top_idxs.tolist(), top_probs.tolist())]


def predict_gradcam(model, classes, img_path, transform, device, save_path=None, topk=1):
    img = Image.open(img_path).convert('RGB')
    x   = transform(img).unsqueeze(0).to(device)
    x.requires_grad_(True)

    gradcam  = GradCAM(model)
    logits   = model(x)
    probs    = torch.softmax(logits, dim=1)[0]
    pred_idx = logits.argmax(dim=1).item()

    logits[0, pred_idx].backward()
    cam     = gradcam(x, class_idx=pred_idx)
    overlay = overlay_heatmap(img, cam)

    top_probs, top_idxs = probs.topk(min(topk, len(classes)))
    results = [(classes[i], p) for i, p in zip(top_idxs.tolist(), top_probs.tolist())]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img);     ax1.set_title('Original');      ax1.axis('off')
    ax2.imshow(overlay)
    ax2.set_title('Grad-CAM\n' + '\n'.join([f'{n}: {p:.1%}' for n, p in results]))
    ax2.axis('off')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [✓] Grad-CAM → {save_path}')
    else:
        plt.show()
    plt.close(fig)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Cartoon classifier inference')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image',      default=None, help='Single image path')
    parser.add_argument('--folder',     default=None, help='Folder of images')
    parser.add_argument('--gradcam',    action='store_true')
    parser.add_argument('--topk',       type=int, default=1)
    parser.add_argument('--save_dir',   default='outputs/gradcam')
    args = parser.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, classes = load_model(args.checkpoint, device)
    transform = get_transform()

    if args.image:
        paths = [args.image]
    elif args.folder:
        paths = [str(p) for p in Path(args.folder).iterdir() if p.suffix.lower() in IMAGE_EXTS]
        print(f'[✓] Found {len(paths)} images in {args.folder}')
    else:
        parser.error('Provide --image or --folder')

    os.makedirs(args.save_dir, exist_ok=True)

    for img_path in paths:
        stem = Path(img_path).stem
        if args.gradcam:
            results = predict_gradcam(
                model, classes, img_path, transform, device,
                save_path=os.path.join(args.save_dir, f'{stem}_gradcam.png'),
                topk=args.topk,
            )
        else:
            _, results = predict(model, classes, img_path, transform, device, topk=args.topk)

        result_str = ' | '.join([f'{n}: {p:.1%}' for n, p in results])
        print(f'{Path(img_path).name:40s} → {result_str}')


if __name__ == '__main__':
    main()