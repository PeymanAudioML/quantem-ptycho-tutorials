"""Setup script for pyms, ensure that all dependent packages are installed."""
from setuptools import setup, find_packages

setup(
    name="AmpflowS",
    version="0.1",
    description="Amplitude flow scattering matrix reconstruction",
    author="Hamish Brown",
    author_email="hamishgallowaybrown@gmail.com",
    url="https://github.com/HamishGBrown/AmpflowS/",
    packages=find_packages(),
    install_requires=[
        "cupy >= 6.0",
        "h5py >= 2.10",
        "ipython >= 4.0",
        "numpy >= 1.17",
        "tqdm >= 4.0",
    ],
)
