from pathlib import Path
import os

from PIL import Image
import numpy as np
import torch
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.datasets import DTD

from bonner.models import extract_features
from bonner.models.hooks import global_maxpool

weights = AlexNet_Weights.IMAGENET1K_V1

model = ("alexnet-IMAGENET1K_V1", alexnet(weights=weights))
input = DTD(os.getenv("DATASETS_HOME"), download=True)
input = (
    "DTD",
    list((Path(input.root) / "dtd" / "dtd" / "images").rglob("*.jpg"))[:500],
)


def pre_hook(filepath: Path) -> torch.Tensor:
    return weights.transforms()(Image.open(filepath))


x = extract_features(
    model=model,
    stimuli=input,
    stimulus_ids=np.array([str(_) for _ in range(500)]),
    nodes=["features.0", "classifier.1", "classifier.2"],
    pre_hook=("", pre_hook),
    # post_hook=("global_maxpool", global_maxpool),
    use_cached=True,
)
print(1)
