# Bonner Lab Models

Utilities for working with PyTorch models.

## Installation

`pip install git+https://github.com/BonnerLab/bonner-models`

## Usage

See `bonner/tests/test.py` for some example usage.

## API

`bonner.models` exposes the following public classes:

- `FeatureExtractor`: to extract features
- `hooks.<>`: hooks for post-processing extracted features

## Environment variables

All data will be stored at the path specified by `BONNER_MODELS_HOME`.

## Dependencies

- `tqdm`
- `numpy`
- `xarray`
- `netCDF4`
- `torch`
- `torchvision`
- `torchdata`
- `Pillow`

## Things to do

- TODO implement logging
- TODO write tests
- TODO write documentation
- TODO remove `setup.py` once `setuptools` [allows editable installs with a `pyproject.toml`](https://github.com/pypa/setuptools/issues/2816)
