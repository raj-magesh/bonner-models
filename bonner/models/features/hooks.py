from typing import Mapping, Tuple
import os

from PIL import Image
import torch
import torchvision.transforms as tr


def global_maxpool(features: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    for key, value in features.items():
        if value.ndim == 4:
            features[key] = torch.nn.MaxPool2d(value.size()[2:])(value).squeeze()
    return features


def preprocess(
    path: os.PathLike,
    resize_size: int = 256,
    crop_size: int = 224,
    mean: Tuple = (0.485, 0.456, 0.406),
    std: Tuple = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    transform = tr.Compose(
        [
            tr.Resize(size=resize_size),
            tr.CenterCrop(size=crop_size),
            tr.ToTensor(),
            tr.Normalize(mean=mean, std=std),
        ]
    )
    return transform(Image.open(path).convert("RGB"))
