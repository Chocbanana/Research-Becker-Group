"""
author: Bhavana Jonnalagadda
"""
from .simpleFork import Fork

import torch
from torch import nn
import numpy as np

class ConvMF(Fork):
    """Convolutional layers before feeding into Fork model

    Args:
        stem_layer_dims (Tuple[], optional): Tuple of length > 0, for the seq layers before fork
        fork_layer_dims (Tuple[], optional): Tuple len >=0, last layer is always the rank dim
    """
    defaults = dict(rank = 6,
                    img_size = (54, 96),
                    stem_layer_dims = [500, 200],
                    fork_layer_dims = [200, 300],
                    conv_dims = [],
                    )


    def __init__(self,
                 rank = defaults["rank"],
                 img_size = defaults["img_size"],
                 stem_layer_dims = defaults["stem_layer_dims"],
                 fork_layer_dims = defaults["fork_layer_dims"],
                 conv_dims = defaults["conv_dims"],
                 ):
        super().__init__(rank, img_size, stem_layer_dims, fork_layer_dims)
        assert len(conv_dims) > 0

        assert all([len(x) == 4 for x in conv_dims])

        for c in conv_dims:
            pass

        # The rest of the network is initialized in Fork

    def forward(self, x):



        x = self.flatten(x)
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
