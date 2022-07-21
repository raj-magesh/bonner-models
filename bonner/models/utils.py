from pathlib import Path
import os

import xarray as xr

BONNER_MODELS_HOME = Path(os.getenv("BONNER_MODELS_HOME", str(Path.home())))


def concatenate_features(features: dict[str, xr.DataArray]) -> xr.DataArray:
    for node, feature in features.items():
        feature["node"] = ("channel", [node] * feature.sizes["channel"])

    return xr.concat(features.values(), dim="channel").rename("concatenated")
