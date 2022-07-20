from pathlib import Path
from typing import Callable, Dict, List

from tqdm import tqdm
import netCDF4
import xarray as xr
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchdata.datapipes.iter import Mapper, Batcher, IterableWrapper

from .utils import _MODELS_HOME


class FeatureExtractor:
    def __init__(
        self,
        model: torch.nn.modules.module.Module,
        nodes: List[str],
        *,
        pre_hook: Callable[[Path], torch.Tensor],
        post_hook: Callable[
            [Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
        ] = lambda x: x,
        model_identifier: str = "",
        pre_hook_identifier: str = "",
        post_hook_identifier: str = "",
    ) -> None:
        """
        TODO write documentation
        TODO this function writes to netCDF files on disk at every batch: is this asynchronous and does it begin computing the next batch in parallel or not?
        TODO automatically extract identifiers if unspecified

        :param model: a PyTorch model
        :type model: torch.nn.modules.module.Module
        :param nodes: list of layer names to extract features from, in standard PyTorch format (e.g. 'classifier.0')
        :type nodes: List[str]
        :param pre_hook: a function that takes in the path to a stimulus and preprocesses it into a tensor
        :type pre_hook: Callable[[Path], torch.Tensor]
        :param post_hook: a function that is applied to the dictionary (Dict[str, torch.Tensor]) of features extracted by the model, defaults to lambdax:x
        :type post_hook: _type_, optional
        :param model_identifier: identifier for the model, defaults to ""
        :type model_identifier: str, optional
        :param pre_hook_identifier: identifier for the pre_hook, defaults to ""
        :type pre_hook_identifier: str, optional
        :param post_hook_identifier: identifier for the post_hook, defaults to ""
        :type post_hook_identifier: str, optional
        """
        self.model = model
        self.model.eval()
        self.nodes = nodes
        self.pre_hook = pre_hook
        self.post_hook = post_hook
        self.model_identifier = model_identifier
        self.pre_hook_identifier = pre_hook_identifier
        self.post_hook_identifier = post_hook_identifier

    def extract(
        self,
        stimuli: List[Path],
        *,
        stimulus_ids: List[str] = [],
        stimulus_set_identifier: str = "",
        custom_identifier: str = "",
        use_cached: bool = True,
        batch_size: int = 256,
    ) -> Dict[str, xr.DataArray]:
        if not stimulus_ids:
            stimulus_ids = [str(id) for id, _ in enumerate(stimuli)]
        else:
            assert len(stimuli) == len(
                stimulus_ids
            ), "the number of stimulus_ids does not match the number of stimuli"
        stimulus_ids = np.array(stimulus_ids)

        cache_dir = self._create_cache_directory(
            stimulus_set_identifier=stimulus_set_identifier,
            custom_identifier=custom_identifier,
        )
        filepaths = {node: cache_dir / f"{node}.nc" for node in self.nodes}

        nodes_to_compute = self.nodes.copy()
        for node in self.nodes:
            if filepaths[node].exists():
                if use_cached:
                    nodes_to_compute.remove(node)  # don't re-compute
                else:
                    filepaths[node].unlink()  # delete pre-cached features

        if nodes_to_compute:
            datapipe = IterableWrapper(stimuli)
            datapipe = Mapper(datapipe, fn=self.pre_hook)
            datapipe = Batcher(datapipe, batch_size=batch_size)
            extractor = create_feature_extractor(
                self.model, return_nodes=nodes_to_compute
            )

            netcdf4_files = {
                node: netCDF4.Dataset(filepaths[node], "w", format="NETCDF4")
                for node in nodes_to_compute
            }

            start = 0
            for batch, batch_data in enumerate(
                tqdm(datapipe, desc="batch", leave=False)
            ):
                features = extractor(torch.stack(batch_data))
                features = self.post_hook(features)

                for node, netcdf4_file in netcdf4_files.items():
                    features_node = features[node].detach().cpu().numpy()

                    if batch == 0:
                        self._initialize_netcdf4_file(
                            file=netcdf4_file,
                            node=node,
                            features=features_node,
                        )

                    end = start + len(batch_data)
                    features_saved = netcdf4_file.variables[node]
                    features_saved[start:end, ...] = features_node
                    netcdf4_file.variables["presentation"][start:end] = stimulus_ids[
                        start:end
                    ]

                start += len(batch_data)

            for netcdf4_file in netcdf4_files.values():
                netcdf4_file.sync()
                netcdf4_file.close()

        return {
            node: xr.open_dataarray(filepath) for node, filepath in filepaths.items()
        }

    def _create_cache_directory(
        self, *, stimulus_set_identifier: str = "", custom_identifier: str = ""
    ) -> Path:
        cache_dir = (
            _MODELS_HOME
            / "features"
            / ".".join(
                [
                    _
                    for _ in [
                        self.model_identifier,
                        stimulus_set_identifier,
                        self.pre_hook_identifier,
                        self.post_hook_identifier,
                        custom_identifier,
                    ]
                    if len(_) != 0
                ]
            )
        )
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    @staticmethod
    def _initialize_netcdf4_file(
        *,
        file: netCDF4.Dataset,
        node: str,
        features: torch.Tensor,
    ) -> None:
        if features.ndim == 4:
            dimensions = ["presentation", "channel", "spatial_x", "spatial_y"]
        elif features.ndim == 2:
            dimensions = ["presentation", "channel"]

        for dimension, length in zip(dimensions, (None, *features.shape[1:])):
            file.createDimension(dimension, length)
            if dimension == "presentation":
                variable = file.createVariable(dimension, str, (dimension,))
            else:
                variable = file.createVariable(dimension, np.int64, (dimension,))
                variable[:] = np.arange(length)

        dtype = np.dtype(getattr(np, str(features.dtype).replace("torch.", "")))
        file.createVariable(node, dtype, dimensions)
