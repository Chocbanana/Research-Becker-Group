"""
author: Bhavana Jonnalagadda
"""
from .simpleFork import Fork

from typing import Iterable

import torch
from torch import nn
import numpy as np

def unpack(x):
    return (x[0], x[1]) if isinstance(x, Iterable) else (x, x)

def conv_dims_out(H, W, kernel, stride, padding, dilation):
    kernel1, kernel2 = unpack(kernel)
    stride1, stride2 = unpack(stride)
    padding1, padding2 = unpack(padding)
    dilation1, dilation2 = unpack(dilation)

    H_out = (H + 2*padding1 - dilation1*(kernel1 - 1) - 1) / stride1 + 1
    W_out = (W + 2*padding2 - dilation2*(kernel2 - 1) - 1) / stride2 + 1

    return (int(H_out), int(W_out))


class ConvMF(Fork):
    """Convolutional layers before feeding into Fork model

    Args:
        stem_layer_dims (List[], optional): Tuple of length > 0, for the seq layers before fork
        fork_layer_dims (List[], optional): Tuple len >=0, last layer is always the rank dim
        conv_dims (List[List[]], optional): List of (kernel_sisze, stride, padding, dilation) lists
    """
    defaults = dict(rank = 6,
                    img_size = [54, 96],
                    conv_dims = [[3, 1, 0, 1]],
                    stem_layer_dims = [500, 200],
                    fork_layer_dims = [200, 300],
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

        self.conv_dims = conv_dims
        self.convs = nn.Sequential()
        outdims = img_size

        for c in conv_dims:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=c[0], stride=c[1], padding=c[2], dilation=c[3]),
                    nn.Tanh()))
            outdims = conv_dims_out(outdims[0], outdims[1], *c)

        # Replace first linear layer with correct dims in (instead of img_size dims input in)
        self.seq[0] = nn.Sequential(nn.Linear(np.prod(outdims), stem_layer_dims[0]),
                                    nn.Tanh())

        # The rest of the network is initialized in Fork

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.convs(x)

        x = self.flatten(x)
        seq_output = self.seq(x)

        U = self.U(seq_output)
        V = self.V(seq_output)

        return U, V


    def get_hyperparameters(self, for_tb=False):
        if for_tb:
            h = dict(
                conv_dims_len = torch.Tensor(list(torch.Tensor(self.conv_dims).shape)),
                stem_layer_dims = torch.Tensor(self.stem_layer_dims),
                fork_layer_dims = torch.Tensor(self.fork_layer_dims),
                rank = self.rank,
                img_size = torch.Tensor(self.img_size),
                desc = "Conv layer(s) into Fork model"
            )
        else:
            h = dict(
                conv_dims = self.conv_dims,
                stem_layer_dims = self.stem_layer_dims,
                fork_layer_dims = self.fork_layer_dims,
                rank = self.rank,
                img_size = self.img_size,
                desc = "Conv layer(s) into Fork model"
            )
        return h
