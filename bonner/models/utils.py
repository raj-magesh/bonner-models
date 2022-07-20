from typing import Dict
from pathlib import Path
import os

import xarray as xr

_MODELS_HOME = Path(os.getenv("MODELS_HOME", str(Path.home())))


def concatenate_features(features: Dict[str, xr.DataArray]) -> xr.DataArray:
    for node, feature in features.items():
        feature["node"] = ("channel", [node] * feature.sizes["channel"])

    return xr.concat(features.values(), dim="channel").rename("concatenated")


# def flatten(feature: xr.DataArray]) -> xr.DataArray:
# TODO add support for flattening convolutional features
