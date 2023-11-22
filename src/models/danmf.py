"""
author: Bhavana Jonnalagadda
"""
from .base import BaseMF

import torch
from torch import nn
import numpy as np


class DANMF(BaseMF):

    defaults = dict(rank = 6,
                img_size = (54, 96),
                # Must be even number of dims
                layer_dims = [150, 100, 70, 50, 30, 10],
                )


    def __init__(self,
                 rank = defaults["rank"],
                 img_size = defaults["img_size"],
                 layer_dims = defaults["layer_dims"],
                 ):
        """Implementation of DA-NMF

        Args:
            rank (_type_, optional): Defaults to 6.
            img_size (_type_, optional): Defaults to (54, 96).
            layer_dims (_type_, optional): Defaults to 6 layers
        """
        super().__init__(rank, img_size)

        assert len(layer_dims) % 2 == 0 and len(layer_dims) > 0, "Must be even number of dims"

        self.layer_dims = layer_dims
        self.input = nn.Flatten(start_dim=1)

        self.batched_mm = torch.vmap(torch.linalg.multi_dot)

        self.U = nn.ModuleDict()
        self.V = nn.ModuleDict()
        self.seq = nn.ModuleList()

        for i, r in enumerate(layer_dims):
            if (i % 2 == 0): # Even layers
                if (i == 0):
                    prev = img_size
                    new_V = (r, img_size[1])
                    new_U = (img_size[0], r)
                else:
                    prev = (layer_dims[i-2], layer_dims[i-1])
                    new_V = (r, layer_dims[i-1])
                    new_U = (layer_dims[i-2], r)
                self.seq = self.seq.append(nn.Sequential(nn.Linear(np.prod(prev), np.prod(new_V)), nn.ReLU()))
                self.U.update({str(i): nn.Sequential(nn.Linear(np.prod(prev), np.prod(new_U)),
                                                 nn.Unflatten(1, new_U),
                                                 nn.ReLU()
                                                )})
            else:
                if (i == len(layer_dims) - 1):
                    self.U.update({str(i): nn.Sequential(nn.Linear(layer_dims[i-1] * layer_dims[i-2], layer_dims[i-1] * r),
                                                     nn.Unflatten(1, (layer_dims[i-1], r)),
                                                    )})
                    self.V.update({str(i): nn.Sequential(nn.Linear(layer_dims[i-1] * layer_dims[i-2], r * layer_dims[i-2]),
                                                     nn.Unflatten(1, (r, layer_dims[i-2])),
                                                    )})
                else:
                    if (i == 1):
                        prev = (layer_dims[i-1], img_size[1])
                        new_U = (layer_dims[i-1], r)
                        new_V = (r, img_size[1])
                    else:
                        prev = (layer_dims[i-1], layer_dims[i-2])
                        new_U = (layer_dims[i-1], r)
                        new_V = (r, layer_dims[i-2])

                    self.seq = self.seq.append(nn.Sequential(nn.Linear(np.prod(prev), np.prod(new_U)), nn.ReLU()))
                    self.V.update({str(i): nn.Sequential(nn.Linear(np.prod(prev), np.prod(new_V)),
                                                     nn.Unflatten(1, new_V),
                                                     nn.ReLU()
                                                    )})

    def get_hyperparameters(self):
        return dict(
            layer_dims = torch.Tensor(self.layer_dims),
            rank = self.rank,
            img_size = torch.Tensor(self.img_size),
            desc = "Implementation of DA-NMF"
        )

    def forward(self, x):
        input = self.input(x)
        x = input

        seq_outputs = {}
        for i, layer in enumerate(self.seq):
            x = layer(x)
            seq_outputs[i] = x

        U = self.U["0"](input)
        U_outputs = [U] # Input for U_0 is the original img
        for i in self.U:
            if i == "0":
                continue
            # Uy = self.U[i](seq_outputs[int(i) - 1])
            # U = torch.bmm(U, Uy)
            U_outputs.append(self.U[i](seq_outputs[int(i) - 1]))

        V_outputs = []
        for i in self.V:
            V_outputs.append(self.V[i](seq_outputs[int(i) - 1]))
        # We multiply Vs in reverse
        V_outputs = V_outputs[-1::-1]

        U = self.batched_mm(U_outputs)
        V = self.batched_mm(V_outputs)

        # V = V_outputs[0]
        # for Vy in V_outputs[1:]:
        #     V = torch.bmm(V, Vy)

        return U, V
