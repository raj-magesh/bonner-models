from setuptools import setup, find_namespace_packages

requirements = (
    "tqdm",
    "numpy",
    "xarray",
    "netCDF4",
    "torch",
    "torchvision",
    "torchdata",
    "Pillow",
)

setup(
    name="bonner-models",
    version="0.1.0",
    packages=find_namespace_packages(),
    install_requires=requirements,
)
