## How does it work?

ProtoCell uses the [prototypical network](https://arxiv.org/abs/1703.05175) model to learn a non-linear mapping of the full gene expression vector into a low dimensional embedding space using a neural network, and takes the prototype for each cell type to be the centroid of its vectors in the embedding space. Classification is then performed for a query cell by computing its embedding and finding the nearest euclidean distance to a class prototype.

The model code is adapted from the implementation at https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch.

### Installation

It's recommended to install the dependencies for ProtoCell in a new [miniconda environment](https://docs.conda.io/en/latest/miniconda.html). First install miniconda, then run `conda create -n protocell`.

ProtoCell uses [PyTorch](https://pytorch.org/) 1.1 with Python 3.6 has a few core dependencies that need to be installed:
```
- pytorch
- pandas
- tqdm
```

Make sure to follow instructions for installing PyTorch with GPU support if you have a GPU -- it's much faster!

### Usage Guide

To make predictions on a set of query cells with a pre-trained network, you should have two folders: one for the cells you wish to use for computing prototypes and one for the cells you'd like to make predictions on (queries). Each should contain an unnormalized count matrix of gene expression for each cell. For cells to be used for prototypes, it should contain a .csv file with cell type annotations under the header "annotation". For the query cells, it should contain a .csv file with cell barcodes for the query cells under the header "index". The format for annotations/barcodes is to make it easy to export data from applications that use DataFrame-like formats (Seurat/ScanPy).

To train a new network, you should have one folder with an unnormalized count matrix for cells you wish to use for training and a .csv file with their annotations under the header "annotation". If you wish to make predictions for small cell types, do not provide cells of these types for training--only provide them with the set used for prototypes when making predictions. When training and predicting cross-dataset, make sure that the genes are aligned in the respective count matrices as this is not handled yet.

Example commands for training and computing predictions:

```
python train.py --root="path/to/training_data" --file_type="csv" --experiment_root="where/to/save/model" --cuda
```

```
python get_preds.py --model_path="experiment_root/last_model_1_shot_0.pth" --prototypes="path/to/support" --queries="path/to/queries" --cuda
```

The prediction script outputs a .csv file with barcodes and predicted annotations for each query cell, along with a .csv file containing raw distances to each prototype.
