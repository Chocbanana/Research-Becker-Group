#!/usr/bin/env python
# coding: utf-8
"""Script for running NNs.

For running several NN training/validation, MF on plasma 2D data.

Author: Bhavana Jonnalagadda
"""

## Allow import from our custom lib python files
import sys
import os

module_path = os.path.abspath(os.path.join('..', "lib"))
if module_path not in sys.path:
    sys.path.append(module_path)
    # print(f"appended {module_path}, {sys.path}")


from models.simpleFork import Simple, Fork
from models.danmf import DANMF
from models.convmf import ConvMF
from datasets.gen_plasma_1d import GenPlasma1DDataset
from framework.params import * # device, use_cuda, Checkpoint, various saving strs
from framework.train import train_model

import json

import torch
import onnx
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
# from torch import profiler


#### Per-run user defined variables ####
# Define the machine being used
machine = "PC" # Macbook
# Paths for files
data_dir = os.path.normpath("../../data")
output_dir = os.path.join(data_dir, "output")
tensorboard_dir = os.path.join(output_dir, "tensorboard")
# Dataset params
batch_size = 25
mat_size = [64, 128]
mat_dirs = [os.path.join(data_dir, "gen_plasma_n64", f"mat_{i}") for i in range(3)]
####################

# Device comes from framework.params
print(f"Using {device} device")

# TODO: Move into function
#### Dataset ####
# Load and split the data, and prep for being fed into the NN
data = GenPlasma1DDataset(mat_dirs)
# Divide data into train, validation, test
train_data, validation_data, test_data = random_split(data, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), drop_last=True, num_workers=4)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), drop_last=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)
####################


plt.imshow(data[2669])
print(len(data), type(data[0]), data[244].shape)
print(data[2000])



#### NN run params ####
ranks = [6, 12]

# Get flattened img size, try comparing that
imsize = np.prod(mat_size)
stem_layer_dims = [[500, 200],
                   [int(imsize / (8 * np.power(2, i))) for i in range(3)],
                   [int(imsize / (4 * np.power(2, i))) for i in range(4)],
                   ]
# Use largest possible U/V final size for comparison for fork dims
rsize = ranks[1] * mat_size[1]
fork_layer_dims = [[200, 300],
                   [int(rsize / (9 - 2*i)) for i in range(3)],
                   ]

# kernel_size, stride, padding, dilation
conv_dims = [[5, 1, 3, 1], [3, 1, 0, 1]]

run_params = dict(
                epochs=120,
                checkpoint_at=60,
                batch_pr=200,
                batch_size=batch_size,
                runname="plasma_mf_dims",
                machine=machine,
                )


# Save details
run_details = {"run_params": run_params}
runname = run_details["run_params"]["runname"]
output_run_dir = os.path.join(output_dir, f"run_{runname}")
####################


#### Construct models from params ####
models = []

for r in ranks:
    for sl in stem_layer_dims:
        for fl in fork_layer_dims:
            m = Fork(r, mat_size, sl, fl).to(device)
            print(f"{m.get_name()} \trank = {r} \t sl={sl} \t fl={fl}")
            models.append(m)
            run_details[m.get_name()] = m.get_hyperparameters()

            m = ConvMF(r, mat_size, sl, fl, conv_dims).to(device)
            print(f"{m.get_name()} \trank = {r} \t sl={sl} \t fl={fl} \tcl={conv_dims}")
            models.append(m)
            run_details[m.get_name()] = m.get_hyperparameters()

# Try out DANMF because why not
m = DANMF(ranks[0], mat_size, layer_dims=[150, 100, 50, 30]).to(device)
print(f"{m.get_name()} \trank = {r} \t l={sl}")
models.append(m)
run_details[m.get_name()] = m.get_hyperparameters()

print(run_details)

if not os.path.exists(output_run_dir):
    os.mkdir(output_run_dir)

# Save details
with open(os.path.join(output_run_dir, f"details_{runname}.json"), "w" ) as write:
    json.dump(run_details, write, indent=2 )
####################


#### Run models ####
for model in models:
    writer = SummaryWriter(os.path.join(tensorboard_dir, f'{machine}_{model.get_name()}_{runname}'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_model(model=model,
                optimizer=optimizer,
                train_data=train_dataloader,
                validate_data=validation_dataloader,
                output_run_dir=output_run_dir,
                **run_details["run_params"],
                writer=writer,
                load=False
               )
    writer.close()
####################


#### Winners ####
# In general, all rank 12 and rank 6 were clustered together
# with loss difference very tiny, about +-0.04 at most!?

# Fork_r12_img64_sdim2-3ebc_fdim3-042d
# ConvMF_img64_cdim2-c00f_sdim4-4d8f






