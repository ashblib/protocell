import os
from scipy.io import mmread
import pandas as pd
import numpy as np
import torch.utils.data as data
import torch
import pickle as pkl

class ClassSplitDataset(data.Dataset):
    def __init__(self, root='', ftype="pkl", mode='train', test_split=25, min_samples=25, cv_iter=0, use_test=True):
        self.x, self.y, self.class_map = self.load_data(root=root, ftype=ftype, mode=mode, test_split=test_split, min_samples=min_samples, cv_iter=cv_iter, use_test=use_test)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def load_data(self, root, ftype, mode, test_split, min_samples, cv_iter, use_test):
        new_x = []
        new_y = []
        if not ftype == "pkl":
            x = []
            y = []

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
                x.append(np.log((expression/count_sum)*10000 + 1.0))
                y.append(class_id)

        elif ftype == "pkl":
            with open(root, 'rb') as f:
                data = pkl.load(f)
            x = data[0]
            y = data[1]
            class_map = data[2]

        if not ftype == "pkl":
            class_map = id_to_class
        classes, counts = np.unique(y, return_counts=True)
        classes = list(classes)
        classes = [classes[idx] for idx in range(len(classes)) if counts[idx] > min_samples]
        n_classes = len(classes)
        np.random.seed(0)
        shuffle_idx = np.random.permutation(n_classes)
        val_classes_idx = shuffle_idx[:test_split]
        other_classes_idx = shuffle_idx[test_split:]
        np.random.seed(cv_iter)
        shuffle_idx = np.random.permutation(n_classes-test_split)
        other_classes_idx = [other_classes_idx[idx] for idx in shuffle_idx]
        if use_test:
            if mode == 'train':
                new_classes_idx = other_classes_idx[test_split:]
            elif mode == 'test':
                new_classes_idx = other_classes_idx[:test_split]
            else:
                new_classes_idx = val_classes_idx
        else:
            if mode == 'train':
                new_classes_idx = other_classes_idx
            else:
                new_classes_idx = val_classes_idx
        new_classes = set([classes[idx] for idx in new_classes_idx])
        for idx in range(len(y)):
            c_id = y[idx]
            if c_id in new_classes:
                 new_x.append(x[idx])
                 new_y.append(c_id)

        remap_idx = {}
        class_map_new = {}
        i = 0
        for key, value in class_map.items():
            if key in new_classes:
                remap_idx[key] = i
                class_map_new[i] = value
                i += 1

        new_y = [remap_idx[x] for x in new_y]

        return new_x, new_y, class_map_new

class TrainingDataset(data.Dataset):
    def __init__(self, root='', ftype="pkl", mode='train', train_frac=0.5, cv_iter=0):
        self.x, self.y, self.class_map = self.load_data(root=root, ftype=ftype, mode=mode, train_frac=train_frac, cv_iter=cv_iter)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def load_data(self, root, ftype, mode, train_frac, cv_iter):
        new_x = []
        new_y = []
        if not ftype == "pkl":
            x = []
            y = []

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
                x.append(np.log((expression/count_sum)*10000 + 1.0))
                y.append(class_id)

        # Load cached pkl
        elif ftype == "pkl":
            with open(root, 'rb') as f:
                data = pkl.load(f)
            x = data[0]
            y = data[1]
            class_map = data[2]

        if not ftype == "pkl":
            class_map = id_to_class
        
        # Shuffle the data based on the seed provided by cv_iter
        np.random.seed(cv_iter)
        shuffle_idx = np.random.permutation(len(y))
        x = [x[idx] for idx in shuffle_idx]
        y = [y[idx] for idx in shuffle_idx]

        # Compute train/test splits for each class based on train_frac
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        split_map = {classes[x]: float(counts[x])*train_frac for x in range(n_classes)}
        
        # Sample data for train/test based on splits
        final_x = []
        final_y = []
        n_added = [0 for _ in range(n_classes)]
        if mode == "train":
            for idx, y in enumerate(y):
                split = split_map[y]
                if n_added[y] < split:
                    final_x.append(x[idx])
                    final_y.append(y)
                    n_added[y] += 1
        elif mode == "test":
            for idx, y in enumerate(y):
                split = split_map[y]
                if n_added[y] > split:
                    final_x.append(x[idx])
                    final_y.append(y)
                else:
                    n_added[y] += 1

        return final_x, final_y, class_map
