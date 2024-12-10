import os
from collections.abc import Sequence, Callable

from framework.params import *

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torch import nn

class Plasma4DDataset(Dataset):
    def __init__(self,
                 mat_dirs: Sequence[str],
                 return_imname = False,
                 transform: Callable = None):
        """Loads matrices into tensors that are in the form:
        - Individual folders
        - Files in the folder with number seq N from 0-anything
        - Files prefix path with given `mats`, have `[N].hdf5` as suffix/filename
        - HDF5 file has one 4d matrix called "f"

        Args:
            mat_dirs (Sequence[str]): The dirs in which the hdf5 files are located. All are used across the dataset loading
            return_imname (bool, optional): Whether to also return the path+filename, and index of the matrix
            transform (Callable, optional): A transform fcn to be applied to an matrix after loading. Defaults to None.
        """
        self.mat_dirs = mat_dirs
        self.transform = transform
        self.return_imname = return_imname

        self._matstr = "f"
        self._num_folders = len(mat_dirs)
        # Count the number of occurences of one type of matrix in each folder
        self._folder_lens = [len([f for f in os.listdir(i) if f.endswith("hdf5")])
                                    for i in mat_dirs]
        self._folder_idxs = np.cumsum(self._folder_lens) - 1
        self._len = sum(self._folder_lens)


    def __len__(self):
        return self._len


    def __getitem__(self, idx):
        """
        Returns:
            mat (Tensor(n, n, m, m)): 4d matrix
            retvals (mat, str, int): If return_imname, return also the str of filepath and the index of the file
        """
        folder_num = np.searchsorted(self._folder_idxs, idx)
        mat_num = idx if folder_num == 0 else idx - self._folder_idxs[folder_num - 1] - 1
        mat_path = os.path.join(self.mat_dirs[folder_num], f"{mat_num}.hdf5")

        with h5py.File(mat_path, 'r') as file:
            mat = torch.tensor(file[self._matstr][()].T, device=device, dtype=torch.float)
        if self.transform is not None:
            mat = self.transform(mat)

        retvals = None
        if self.return_imname:
            retvals = (mat, os.path.join(f"{self.mat_dirs[folder_num]}", f"{mat_num}.hdf5"), idx)

        return mat if not retvals else retvals
