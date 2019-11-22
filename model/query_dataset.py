import os
from scipy.io import mmread
import pandas as pd
import numpy as np
import torch.utils.data as data
import torch
import pickle as pkl
import operator
import sys
import collections
import scipy.sparse as sp_sparse
import tables

class QueryDataset(data.Dataset):
    def __init__(self, root='', ftype='', cache_path='', use_cached=False):
        super(QueryDataset, self).__init__()
        self.query_x, self.query_bc = self.load_query(root, ftype=ftype, cache_path=cache_path, use_cached=use_cached)

    def __len__(self):
        return len(self.query_x)

    def __getitem__(self, idx):
        return self.query_x[idx], self.query_bc[idx]

    def load_query(self, root, ftype='csv', cache_path='', use_cached=False):
        if not use_cached:
            if (cache_path != ''):
                cache = True 
            else:
                cache = False  # Cache processed data if a path was provided

            all_x = []  # Holds all of the gene count vectors

            # Load count matrix based on file type
            if (ftype=='hdf') or (ftype=='h5'):
                count_matrix = pd.read_hdf(os.path.join(root, 'matrix.h5'))
            elif ftype=='mtx':
                count_matrix = pd.DataFrame(mmread(os.path.join(root, 'matrix.mtx')).todense())
            elif ftype=='csv':
                count_matrix = pd.read_csv(os.path.join(root, 'matrix.csv'), header=None)
            else:
                print('***Error*** Not a valid file type -- "{}"'.format(ftype))
                sys.exit()

            # Read annotation file
            annotations_df = pd.read_csv(os.path.join(root, 'barcodes.csv'))
            barcodes = list(annotations_df['index'].values)

            # Get transformed expression vectors
            for index, row in count_matrix.iterrows():
                expression = row.values.astype(np.float32)
                count_sum = np.sum(expression)
                all_x.append(np.log((expression/count_sum)*10000 + 1.0))

            # Cache data to a pickle
            if cache:
                with open(cache_path, "wb") as f:
                        pkl.dump([all_x, barcodes], f)

            return all_x, barcodes
        else:
            with open(root, "rb") as f:
                data = pkl.load(f)
            return data[0], data[1]

class PrototypeDataset(data.Dataset):
    def __init__(self, root='', ftype='', cache_path='', use_cached=False):
        super(PrototypeDataset, self).__init__()
        self.proto_x, self.proto_y, self.proto_labels = self.load_proto(root, ftype=ftype, cache_path=cache_path, use_cached=use_cached)

    def __len__(self):
        return len(self.proto_y)

    def __getitem__(self, idx):
        return self.proto_x[idx], self.proto_y[idx]

    def load_proto(self, root, ftype='csv', cache_path='', use_cached=False):
        if not use_cached:
            if cache_path != '':
                cache = True 
            else: 
                cache = False  # Cache processed data if a path was provided

            all_x = []  # Holds all of the gene count vectors
            all_y = []  # Holds all integer class ids

            # Load count matrix based on file type
            if ftype=='mtx':
                count_matrix = pd.DataFrame(mmread(os.path.join(root, 'matrix.mtx')).todense())
            elif ftype=='csv':
                count_matrix = pd.read_csv(os.path.join(root, 'matrix.csv'), header=None)
            else:
                print('***Error*** Not a valid file type -- "{}"'.format(ftype))
                sys.exit()

            # Read annotation file
            annotations_df = pd.read_csv(os.path.join(root, 'annotations.csv'))
            annotations = list(annotations_df['annotation'].values)

            # Get unique annotation classes and make maps to integer class id
            class_names = np.unique(annotations)
            class_map = {class_names[x]: x for x in range(len(class_names))}
            id_to_class = {x: class_names[x] for x in range(len(class_names))}

            # Add transformed expression vector and its class id to data lists
            for index, row in count_matrix.iterrows():
                label = annotations[index]
                class_id = class_map[label]
                expression = row.values.astype(np.float32)
                count_sum = np.sum(expression)
                all_x.append(np.log((expression/count_sum)*10000 + 1.0))
                all_y.append(class_id)

            # Cache data to a pickle
            if cache:
                with open(cache_path, "wb") as f:
                        pkl.dump([all_x, all_y, id_to_class], f)

            return all_x, all_y, id_to_class
        else:
            with open(root, "rb") as f:
                data = pkl.load(f)
            return data[0], data[1], data[2]