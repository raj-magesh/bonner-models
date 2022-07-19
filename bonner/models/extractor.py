from typing import Callable, Tuple, Iterable, Mapping, Dict, List, Any

from tqdm import tqdm
import netCDF4
import xarray as xr
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchdata.datapipes.iter import Mapper, Batcher, IterableWrapper

from .utils import _MODELS_HOME


def _extract_features(
    model: Tuple[str, torch.nn.modules.module.Module],
    stimuli: Tuple[str, Iterable[Any]],
    nodes: List[str],
    stimulus_ids: Iterable[str] = [],
    batch_size: int = 256,
    pre_hook: Tuple[str, Callable[[Any], torch.Tensor]] = ("", lambda x: x),
    post_hook: Tuple[
        str, Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]]
    ] = ("", lambda x: x),
    custom_identifier: str = "",
    use_cached: bool = True,
) -> Dict[str, xr.DataArray]:
    parent_dir = (
        _MODELS_HOME
        / "features"
        / ".".join(
            [
                _
                for _ in [
                    model[0],
                    stimuli[0],
                    pre_hook[0],
                    post_hook[0],
                    custom_identifier,
                ]
                if len(_) != 0
            ]
        )
    )
    # FIXME directory name could be too long: figure out way to systematically hash/truncate
    parent_dir.mkdir(exist_ok=True, parents=True)
    filepaths = {node: parent_dir / f"{node}.nc" for node in nodes}

    model, stimuli, pre_hook, post_hook = (
        model[1],
        stimuli[1],
        pre_hook[1],
        post_hook[1],
    )

    nodes_to_compute = nodes.copy()
    for node in nodes:
        if filepaths[node].exists():
            if use_cached:
                nodes_to_compute.remove(node)  # don't re-compute
            else:
                filepaths[node].unlink()  # delete pre-cached features

    if nodes_to_compute:
        model.eval()

        datapipe = IterableWrapper(stimuli)
        datapipe = Mapper(datapipe, fn=pre_hook)
        datapipe = Batcher(datapipe, batch_size=batch_size)
        extractor = create_feature_extractor(model, return_nodes=nodes_to_compute)

        netcdf4_files = {
            node: netCDF4.Dataset(filepaths[node], "w", format="NETCDF4")
            for node in nodes_to_compute
        }

        start = 0
        for batch, batch_data in enumerate(tqdm(datapipe, desc="batch", leave=False)):
            features = extractor(torch.stack(batch_data))
            features = post_hook(features)

            for node, netcdf4_file in netcdf4_files.items():
                features_node = features[node].detach().cpu().numpy()

                if batch == 0:
                    _initialize_netcdf4_file(
                        node, features_node, len(stimuli), stimulus_ids, netcdf4_file
                    )

                features_saved = netcdf4_file.variables[node]
                features_saved[start : start + len(batch_data), ...] = features_node

            start += len(batch_data)

        for netcdf4_file in netcdf4_files.values():
            netcdf4_file.sync()
            netcdf4_file.close()

    return {node: xr.open_dataarray(filepath) for node, filepath in filepaths.items()}


def _initialize_netcdf4_file(
    node: str,
    features: torch.Tensor,
    n_stimuli: int,
    stimulus_ids: Iterable[str],
    file: netCDF4.Dataset,
) -> None:
    if features.ndim == 4:
        dimensions = ("presentation", "channel", "spatial_x", "spatial_y")
    elif features.ndim == 2:
        dimensions = ("presentation", "channel")

    for dimension, length in zip(dimensions, (n_stimuli, *features.shape[1:])):
        file.createDimension(dimension, length)
        if dimension == "presentation" and len(stimulus_ids) != 0:
            variable = file.createVariable(dimension, str, (dimension,))
            variable[:] = stimulus_ids
        else:
            variable = file.createVariable(dimension, np.int64, (dimension,))
            variable[:] = np.arange(length)

    dtype = np.dtype(getattr(np, str(features.dtype).replace("torch.", "")))
    file.createVariable(node, dtype, dimensions)
