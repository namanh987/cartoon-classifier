import torch.nn as nn
from torchvision import models


def get_model(num_classes: int) -> nn.Module:
    """
    Load pretrained ResNet-18, freeze the backbone, replace the classifier head.

    The head is a two-layer MLP:
        Linear(512 -> 128, no bias) -> BN -> ReLU -> Dropout -> Linear(128 -> num_classes)

    Note: no Softmax — CrossEntropyLoss expects raw logits.
    The caller must call .to(device) after this function returns.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model   = models.resnet18(weights=weights)

    # Freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace head — backbone frozen, head trainable
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128, bias=False),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes),
    )
    # Only head parameters will receive gradients
    model.fc.requires_grad_(True)

    return model