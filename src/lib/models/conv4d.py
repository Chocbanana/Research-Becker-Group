"""
author: Bhavana Jonnalagadda
"""
from .base import BaseMF
from .simpleFork import Fork
from .convmf import ConvMF

from typing import Iterable

import torch
from torch import nn
import numpy as np

def conv_dims_out_4d(D, H, W, kernel, stride, padding, dilation):
    D_out = (D + 2*padding - dilation*(kernel - 1) - 1) / stride + 1
    H_out = (H + 2*padding - dilation*(kernel - 1) - 1) / stride + 1
    W_out = (W + 2*padding - dilation*(kernel - 1) - 1) / stride + 1

    return [int(D_out), int(H_out), int(W_out)]


class Conv4D(BaseMF):
    """Convolutional layers before feeding into Fork model, for 4d input (enforced orthogonality on all U output). Need to use different conv layer than 2d.

    Args:
        stem_layer_dims (List[], optional): Tuple of length > 0, for the seq layers before fork
        fork_layer_dims (List[], optional): Tuple len >=0, last layer is always the rank dim
        conv_dims (List[List[]], optional): List of (kernel_sisze, stride, padding, dilation) lists
    """
    defaults = dict(rank = 6,
                    img_size = [32, 32, 64, 64],
                    conv_dims = [[5, 1, 3, 1], [3, 1, 0, 1]],
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
        super().__init__(rank, img_size)

        # assert len(conv_dims) > 0
        assert all([len(x) == 4 for x in conv_dims])
        assert len(stem_layer_dims) > 0
        assert len(img_size) == 4

        self.stem_layer_dims = stem_layer_dims
        self.fork_layer_dims = fork_layer_dims
        self.conv_dims = conv_dims
        self.convs = nn.Sequential()

        ## Init the conv layers before they feed into the seq linear layers
        outdims = self.img_size[1:] # img_size[0] is treated as channel C
        for c in conv_dims:
            self.convs.append(
                nn.Sequential(
                    nn.Conv3d(img_size[0], img_size[0], kernel_size=c[0], stride=c[1], padding=c[2], dilation=c[3]),
                    nn.Tanh()))
            outdims = conv_dims_out_4d(outdims[0], outdims[1], outdims[2], *c)

        ## Init the sequential linear layers
        self.flatten = nn.Flatten(start_dim=1)
        # 1st hidden layer: flattened conv output x 1st dim_0
        self.seq = nn.Sequential(
                        nn.Sequential(
                            nn.Linear(np.prod([img_size[0]] + outdims), stem_layer_dims[0]),
                            nn.Tanh()))
        # Other layers: dim_{n-1} x dim_{n}
        for i, r in enumerate(stem_layer_dims[1:]):
            self.seq = self.seq.append(nn.Sequential(
                            nn.Linear(stem_layer_dims[i], r),
                            nn.Tanh()))

        ## Init the fork seq layers, for S, U_1, U_2, U_3, U_4
        fork_dims = [stem_layer_dims[-1]] + fork_layer_dims
        self.fork_seqs = nn.ModuleDict({
                "S": nn.Sequential(),
                "U_1": nn.Sequential(),
                "U_2": nn.Sequential(),
                "U_3": nn.Sequential(),
                "U_4": nn.Sequential()
        })
        for seq in self.fork_seqs.values():
            for i, r in enumerate(fork_dims[1:]):
                seq.append(nn.Sequential(nn.Linear(fork_dims[i], r), nn.Tanh()))
        # The last layer for S must be of shape (rank, rank, rank, rank)
        self.fork_seqs["S"].append(nn.Sequential(
                nn.Linear(fork_dims[-1], rank**4),
                nn.Unflatten(1, [rank] * 4)))
        # The last layers of U_n must be matrices of shape (img_size[n-1], rank)
        for i, seq in enumerate(list(self.fork_seqs.values())[1:]):
            seq.append(nn.Sequential(
                nn.Linear(fork_dims[-1], img_size[i] * rank),
                nn.Unflatten(1, (img_size[i], rank))))


    def forward(self, x):
        x = self.convs(x)

        x = self.flatten(x)
        x = self.seq(x)

        S, U_1, U_2, U_3, U_4 = (s(x) for s in self.fork_seqs.values())

        return S, (U_1, U_2, U_3, U_4)


    def get_hyperparameters(self, for_tb=False):
        if for_tb:
            h = dict(
                conv_dims_len = torch.Tensor(list(torch.Tensor(self.conv_dims).shape)),
                stem_layer_dims = torch.Tensor(self.stem_layer_dims),
                fork_layer_dims = torch.Tensor(self.fork_layer_dims),
                rank = self.rank,
                img_size = torch.Tensor(self.img_size),
                desc = "Conv layer(s) into Fork for 4D model"
            )
        else:
            h = dict(
                conv_dims = self.conv_dims,
                stem_layer_dims = self.stem_layer_dims,
                fork_layer_dims = self.fork_layer_dims,
                rank = self.rank,
                img_size = self.img_size,
                desc = "Conv layer(s) into Fork for 4D model"
            )
        return h
