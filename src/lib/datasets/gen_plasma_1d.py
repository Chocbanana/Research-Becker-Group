import os
from collections.abc import Sequence, Callable

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torch import nn

class GenPlasma1DDataset(Dataset):
    def __init__(self,
                 mat_dirs: Sequence[str],
                 mats: Sequence[str] = ["f"],
                 return_imname = False,
                 transform: Callable = None):
        """Loads matrices into tensors that are in the form:
        - Individual folders
        - Files in the folder with number seq N from 0-anything
        - Files prefix with given `mats`, have `[N].hdf5` as suffix

        Args:
            mat_dirs (Sequence[str]): The dirs in which the hdf5 files are located. All are used across the dataset loading
            mats (Sequence[str], optional): The dataset(s) to load. Defaults to ["f"].
            return_imname (bool, optional): Whether to also return the path+filename of the matrix
            transform (Callable, optional): A transform fcn to be applied to an matrix after loading. Defaults to None.
        """
        self.mat_dirs = mat_dirs
        self.mat_names = mats
        self.transform = transform
        self.return_imname = return_imname

        self._num_folders = len(mat_dirs)
        # Count the number of occurences of one type of matrix in each folder
        self._folder_lens = [len([f for f in os.listdir(i) if f.endswith("hdf5")])
                                    for i in mat_dirs]
        self._folder_idxs = np.cumsum(self._folder_lens) - 1
        self._len = sum(self._folder_lens)


    def __len__(self):
        return self._len


    def __getitem__(self, idx):
        folder_num = np.searchsorted(self._folder_idxs, idx)
        mat_num = idx if folder_num == 0 else idx - self._folder_idxs[folder_num - 1] - 1
        mat_path = os.path.join(self.mat_dirs[folder_num], f"{mat_num}.hdf5")

        with h5py.File(mat_path, 'r') as file:
            mats = tuple(torch.Tensor(file[m][()]) for m in self.mat_names)
        if self.transform is not None:
            mats = tuple(self.transform(m) for m in mats)

        if self.return_imname:
            mats = mats + (os.path.join(f"{self.mat_dirs[folder_num]}", f"{mat_num}.hdf5"), idx)

        return mats[0] if len(mats) == 1 else mats
