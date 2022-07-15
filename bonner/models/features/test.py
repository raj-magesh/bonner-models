from pathlib import Path

from torchvision.models import alexnet
from torchvision.datasets import DTD

from .extractor import extract_features
from .hooks import preprocess, global_maxpool

root = Path("/mnt/data/projects/models")
model = ("alexnet-imagenet", alexnet(pretrained=True))
input = DTD(root / "datasets", download=True)
input = ("DTD", [str(path) for path in input.root.rglob("*.jpg")])


x = extract_features(
    model=model,
    stimuli=input,
    nodes=["features.0", "classifier.1", "classifier.2"],
    pre_hook=("basic", preprocess),
    post_hook=("maxpool", global_maxpool),
    use_cached=True,
)
