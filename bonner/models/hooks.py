import torch


def global_maxpool(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for key, value in features.items():
        if value.ndim == 4:
            features[key] = torch.nn.MaxPool2d(value.size()[2:])(value).squeeze()
    return features
