Source code for the article "Super-resolution on unstructured coastal wave
computations with Graph Neural Networks and Polynomial Regressions"
- Kuehn, Abadie, Delpey, Roeber (2024)

**Warning**
This repository is still under (heavy) construction! 

# Installation Steps
After cloning the repository you have to install the necessary packages.
The code should work on `python>=3.10` with `pytorch=2.2.0`. For more information
about CPU/GPU-support refer to the
[official Pytorch website](https://pytorch.org/get-started/locally/).

Other necessary packages are `torch_geometric`, `torch_scatter` and all packages
listed in the `requirements.txt` file. 

The following is a _rudimentary_ example of how you could install these packages
on your machine.

1. Install Anaconda (see their
[official documentation](https://docs.anaconda.com/free/anaconda/install/index.html))
to create a virtual environment. To install the aforementioned Python and PyTorch
versions at the same time, you could do this via 
`conda create -n super-resolution-env python=3.10 pytorch=2.2.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`. Note that this installs the GPU version.
Do not forget to activate your new conda environment before proceeding. You can
do this with `conda activate super-resolution-env`. 
2. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/),
a library for training Graph Neural Networks, with pip (or conda if you like):
`pip install torch_geometric`. You will also need some optional dependencies
that you can install with (adjust your PyTorch and Cuda / CPU version if necessary):
`pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
-f https://data.pyg.org/whl/torch-2.2.0+cu118.html`.
3. Install the rest of the packages with `pip install -r requirements.txt`. 


