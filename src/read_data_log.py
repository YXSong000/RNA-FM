import os
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import h5py


class SuperTileRNADataset(Dataset):
    def __init__(self, csv_path: str, features_path, feature_use='cluster_features', quick=None):
        self.csv_path = csv_path
        self.quick = quick
        self.features_path = features_path
        self.feature_use = feature_use
        if type(self.csv_path) == str:
            self.data = pd.read_csv(self.csv_path)
        else:
            self.data = self.csv_path

        # find the number of genes
        row = self.data.iloc[0]
        # remove the rna_file_name column
        row = row[[x for x in row.keys() if x != 'rna_file_name']]
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        self.num_genes = len(rna_data)

        # find the feature dimension, assume all images in the reference file have the same dimension
        path = os.path.join(self.features_path, row['wsi_file_name'].split('/')[-1].replace('.svs', ''), row['wsi_file_name'].split('/')[-1].replace('.svs', '')+'.h5')
        f = h5py.File(path, 'r')
        features = f[self.feature_use][:]
        self.feature_dim = features.shape[1]
        f.close()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.features_path, row['wsi_file_name'].split('/')[-1].replace('.svs', ''), row['wsi_file_name'].split('/')[-1].replace('.svs', '') +'.h5')
        row = row[[x for x in row.keys() if x != 'rna_file_name']]
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        # Apply log2(v + 1) transform so high-abundance genes do not dominate the MSE loss
        rna_data = np.log2(rna_data + 1.0, out=rna_data)
        rna_data = torch.tensor(rna_data, dtype=torch.float32)
        try:
            if 'GTEX' not in path:
                path = path.replace('.svs','')
            f = h5py.File(path, 'r')
            features = f[self.feature_use][:]
            f.close()
            features = torch.tensor(features, dtype=torch.float32)
        except Exception as e:
            print(e)
            print(path)
            features = None

        return features, rna_data, row['wsi_file_name'].split('/')[-1].replace('.svs', '')
