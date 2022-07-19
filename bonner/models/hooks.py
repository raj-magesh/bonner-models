from typing import Mapping

import torch


def global_maxpool(features: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    for key, value in features.items():
        if value.ndim == 4:
            features[key] = torch.nn.MaxPool2d(value.size()[2:])(value).squeeze()
    return features
