"""
author: Bhavana Jonnalagadda
"""
from typing import Tuple
import hashlib
from abc import abstractmethod

import torch
from torch import nn


format_name = dict(
    rank = ("r", lambda x: str(x)),
    img_size = ("img", lambda x: x[0]),
    stem_layer_dims = ("sdim", lambda x: f"{len(x)}-{hashlib.shake_256(str(x).encode()).hexdigest(2)}"),
    fork_layer_dims = ("fdim", lambda x: f"{len(x)}-{hashlib.shake_256(str(x).encode()).hexdigest(2)}"),
    layer_dims = ("dim", lambda x: f"{len(x)}-{hashlib.shake_256(str(x).encode()).hexdigest(2)}"),
    conv_dims = ("cdim", lambda x: f"{len(x)}-{hashlib.shake_256(str(x).encode()).hexdigest(2)}"),
)

class BaseMF(nn.Module):
    """Base MF model from which all others subclass.

    Args:
        rank (int, optional): The rank of the final UxV. Defaults to defaults["rank"].
        img_size (Tuple[int, int], optional): Input image size. Defaults to defaults["img_size"].
    """
    defaults = dict(rank= 12, img_size = (54, 96))


    def __init__(self, rank: int = defaults["rank"], img_size: Tuple[int, int] = defaults["img_size"]):
        super().__init__()
        self.rank = rank
        self.img_size = img_size


    def get_hyperparameters(self):
        return dict(
            rank = self.rank,
            img_size = torch.Tensor(self.img_size),
            desc = "Base MF model",
        )


    def get_name(self):
        """The model name = the class, plus any hyperparameters different from
        the defaults. Use a method instead of defining self.name in init because this needs to
        occur *after* subclass __init__ occurs
        """
        #
        name = self.__class__.__name__
        for k, v in self.defaults.items():
            if v != getattr(self, k):
                name += "_" + format_name[k][0] + format_name[k][1](getattr(self, k))
        return name


    @abstractmethod
    def forward(self, x):
        pass
