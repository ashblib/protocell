## How does it work?

ProtoCell uses the [prototypical network](https://arxiv.org/abs/1703.05175) model to learn a non-linear mapping of the full gene expression vector into a low dimensional embedding space using a neural network, and takes the prototype for each cell type to be the centroid of its vectors in the embedding space. Classification is then performed for a query cell by computing its embedding and finding the nearest euclidean distance to a class prototype.

### Installation

It's recommended to install the dependencies for ProtoCell in a new [miniconda environment](https://docs.conda.io/en/latest/miniconda.html). First install miniconda, then run `conda create -n protocell`.

ProtoCell uses [PyTorch](https://pytorch.org/) with Python 3.6 has a few dependencies that need to be installed:
```
- pytorch
- pandas
- tqdm
```

Make sure to follow instructions for installing PyTorch with GPU support if you have a GPU -- it's much faster!

### Usage Guide
