import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torch import nn

class YtVidsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """For loading images in the form:
        - Folder with subfolders labeled "test_imgs0", "test_imgs1", "test_imgs2"
        - In each folder with images labeled "image{A, B, C}_n" where n >= 0 and a sequence
            starting at 0

        Args:
            img_dir (str | os.Path): The dir where the subfolders of images are located
            transform (Function(x), optional): A transform fcn to be applied to an image after loading. Defaults to None.
        """
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

        self._folder_name = "test_imgs"
        self._num_folders = 3
        self._image_names = ["imageA", "imageB", "imageC"]
        self._folder_lens = [len([f for f in
                                    os.listdir(os.path.join(img_dir, self._folder_name + str(i))) if f.endswith("jpg")])
                                    for i in range(self._num_folders)]
        self._folder_idxs = np.cumsum(self._folder_lens) - 1
        self._len = sum(self._folder_lens)


    def __len__(self):
        return self._len


    def __getitem__(self, idx):
        folder_num = np.searchsorted(self._folder_idxs, idx)
        img_num = idx if folder_num == 0 else idx - self._folder_idxs[folder_num - 1] - 1
        img_path = os.path.join(self.img_dir, self._folder_name + str(folder_num), f"{self._image_names[folder_num]}_{img_num}.jpg")
        # Load in as a 2d image (no channels)
        image = read_image(img_path, mode=ImageReadMode.GRAY)[0, :, :].to(torch.float32)
        # Normalize w.r.t. unit Frobenius norm
        image = nn.functional.normalize(image, dim=(0, 1))
        if self.transform is not None:
            image = self.transform(image)
        return image
