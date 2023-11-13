"""
author: Bhavana Jonnalagadda
"""
from .base import BaseMF

import torch
from torch import nn
import numpy as np

class Fork(BaseMF):
    """Linear tanh layers that fork into two separate linear tanh channels for U, V

    Args:
        stem_layer_dims (Tuple[], optional): Tuple of length > 0, for the seq layers before fork
        fork_layer_dims (Tuple[], optional): Tuple len >=0, last layer is always the rank dim
    """
    defaults = dict(rank = 6,
                    img_size = (54, 96),
                    stem_layer_dims = [500, 400, 300, 200],
                    fork_layer_dims = [300, 200, 100, 100],
                    )


    def __init__(self,
                 rank = defaults["rank"],
                 img_size = defaults["img_size"],
                 stem_layer_dims = defaults["stem_layer_dims"],
                 fork_layer_dims = defaults["fork_layer_dims"],
                 ):
        super().__init__(rank, img_size)
        assert len(stem_layer_dims) > 0
        self.stem_layer_dims = stem_layer_dims
        self.fork_layer_dims = fork_layer_dims

        self.input = nn.Flatten(start_dim=1)

        self.seq = nn.Sequential(nn.Sequential(nn.Linear(np.prod(img_size), stem_layer_dims[0]), nn.Tanh()))
        for i, r in enumerate(stem_layer_dims[1:]):
            self.seq = self.seq.append(nn.Sequential(nn.Linear(stem_layer_dims[i], r), nn.Tanh()))

        fork_dims = [stem_layer_dims[-1]] + fork_layer_dims
        self.U = nn.Sequential()
        self.V = nn.Sequential()

        for i, r in enumerate(fork_dims[1:]):
            self.U.append(nn.Sequential(nn.Linear(fork_dims[i], r), nn.Tanh()))
            self.V.append(nn.Sequential(nn.Linear(fork_dims[i], r), nn.Tanh()))

        # The last layer must be U: batch x m x r,  V: batch x r x n
        self.U.append(nn.Sequential(nn.Linear(fork_dims[-1], img_size[0] * rank),
                                    nn.Unflatten(1, (img_size[0], rank)),
                                    nn.Tanh()))
        self.V.append(nn.Sequential(nn.Linear(fork_dims[-1], rank * img_size[1]),
                                    nn.Unflatten(1, (rank, img_size[1])),
                                    nn.Tanh()))


    def forward(self, x):
        x = self.input(x)
        seq_output = self.seq(x)

        U = self.U(seq_output)
        V = self.V(seq_output)

        return U, V


    def get_hyperparameters(self):
        return dict(
            stem_layer_dims = torch.Tensor(self.stem_layer_dims),
            fork_layer_dims = torch.Tensor(self.fork_layer_dims),
            rank = self.rank,
            img_size = torch.Tensor(self.img_size),
            desc = "Linear layers that fork into two separate channels for U, V"
        )


class Simple(BaseMF):
    """ Basic model with ReLU layers decreasing in size, fork at output into U and V
    """
    defaults = dict(rank= 12,
                    img_size = (54, 96),
                    layer_dims = (500, 400, 300, 200, 100, 100),
                    )


    def __init__(self,
                 rank = defaults["rank"],
                 img_size = defaults["img_size"],
                 layer_dims = defaults["layer_dims"]
                 ):
        super().__init__(rank, img_size)
        self.layer_dims = layer_dims

        self.input = nn.Flatten(start_dim=1)

        self.seq = nn.Sequential(nn.Sequential(nn.Linear(np.prod(img_size), layer_dims[0]), nn.ReLU()))
        for i, r in enumerate(layer_dims[1:]):
            self.seq = self.seq.append(nn.Sequential(nn.Linear(layer_dims[i], r), nn.ReLU()))

        self.U = nn.Sequential(nn.Linear(layer_dims[-1], img_size[0] * rank),
                               nn.Unflatten(1, (img_size[0], rank)),
                               nn.ReLU())
        self.V = nn.Sequential(nn.Linear(layer_dims[-1], rank * img_size[1] ),
                               nn.Unflatten(1, (rank, img_size[1])),
                               nn.ReLU())

    def get_hyperparameters(self):
        return dict(
            layer_dims = self.layer_dims,
            rank = self.rank,
            img_size = self.img_size,
            desc = "ReLU layers decreasing in size, fork at output into U and V"
        )

    def forward(self, x):
        input = self.input(x)
        seq_output = self.seq(input)

        U = self.U(seq_output)
        V = self.V(seq_output)

        return U, V
