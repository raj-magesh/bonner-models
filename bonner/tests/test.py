from pathlib import Path
import os

from PIL import Image
import torch
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.datasets import DTD

from bonner.models import FeatureExtractor
from bonner.models.hooks import global_maxpool

weights = AlexNet_Weights.IMAGENET1K_V1


def preprocess(filepath: Path) -> torch.Tensor:
    return weights.transforms()(Image.open(filepath))


dataset = DTD(os.getenv("DATASETS_HOME"), download=True)
stimuli = list((Path(dataset.root) / "dtd" / "dtd" / "images").rglob("*.jpg"))

extractor = FeatureExtractor(
    model=alexnet(weights=weights),
    nodes=[
        "features.1",
        "features.4",
    ],
    pre_hook=preprocess,
    post_hook=global_maxpool,
    model_identifier="AlexNet-IMAGENET1K_V1",
    post_hook_identifier="maxpool",
)

features = extractor(
    stimuli=stimuli,
    stimulus_ids=[],
    stimulus_set_identifier="DTD",
    custom_identifier="",
    use_cached=True,
    batch_size=256,
)
