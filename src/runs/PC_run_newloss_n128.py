#!/usr/bin/env python
# coding: utf-8
"""Script for running NNs.

For running several NN training/validation, MF on plasma 2D data.
In this run, we use no nonnegativity in the loss function (aka do not do rectified
linear on U*V, since we want to allow negative values), along with training on
the bigger image dataset.

Author: Bhavana Jonnalagadda
"""

## Allow import from our custom lib python files
import sys
import os

module_path = os.path.abspath(os.path.join('..', "lib"))
if module_path not in sys.path:
    sys.path.append(module_path)
    # print(f"appended {module_path}, {sys.path}")


from models.simpleFork import Fork
from models.danmf import DANMF
from models.convmf import ConvMF
from datasets.gen_plasma_1d import GenPlasma1DDataset
from framework.params import * # device, use_cuda, Checkpoint, various saving strs
from framework.train import loss_fcn_neg, train_model

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
mat_size_1 = [64, 128]
mat_dirs_1 = [os.path.join(data_dir, "gen_plasma_n64", f"mat_{i}") for i in range(3)]
mat_size_2 = [128, 256]
mat_dirs_2 = [os.path.normpath(data_dir + "/gen_plasma_n128/mat_hdf5/")]
####################

# Device comes from framework.params
print(f"Using {device} device")

# TODO: Move into function
#### Dataset ####
# Load and split the data, and prep for being fed into the NN
data_n64 = GenPlasma1DDataset(mat_dirs_1)
# Divide data into train, validation, test
train_n64, validation_n64, _ = random_split(data_n64, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))

train_dataloader_n64 = DataLoader(train_n64, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), drop_last=True, num_workers=4)
validation_dataloader_n64 = DataLoader(validation_n64, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), drop_last=True, num_workers=4)

data_n128 = GenPlasma1DDataset(mat_dirs_2)
train_n128, validation_n128, _ = random_split(data_n128, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))
train_dataloader_n128 = DataLoader(train_n128, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), drop_last=True, num_workers=4)
validation_dataloader_n128 = DataLoader(validation_n128, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), drop_last=True, num_workers=4)
####################

# Output validate
plt.imshow(data_n64[2669])
print(len(data_n64), type(data_n64[0]), data_n64[244].shape)
print(data_n64[2000])

plt.imshow(data_n128[2669])
print(len(data_n128), type(data_n128[0]), data_n128[244].shape)
print(data_n128[2000])


#### Run params ####
run_params = dict(
                epochs=120,
                checkpoint_at=60,
                batch_pr=200,
                batch_size=batch_size,
                runname="plasma_newloss_n128",
                machine=machine,
                )


# Save details
run_details = {"run_params": run_params}
runname = run_details["run_params"]["runname"]
output_run_dir = os.path.join(output_dir, f"run_{runname}")
####################


#### NN params ####
ranks_n64 = range(6, 13)
ranks_n128 = [6, 12, 30]

sl_n64 = [500, 200]
fl_n64 = [200, 300]


# Get flattened img size, try comparing that
imsize = np.prod(mat_size_2)
sls_n128 = [[500, 200],
                   [int(imsize / (4 * np.power(2, i))) for i in range(4)],
                   ]
# Use largest possible U/V final size for comparison for fork dims
rsize = ranks_n128[2] * mat_size_2[1]
fls_n128 = [[200, 300],
                   [int(rsize / (9 - 2*i)) for i in range(3)],
                   ]

# kernel_size, stride, padding, dilation
conv_dims = [[5, 1, 3, 1], [3, 1, 0, 1]]
####################


#### Construct models from params ####
# n64
models_n64 = []
for r in ranks_n64:
    m = ConvMF(r, mat_size_1, sl_n64, fl_n64, conv_dims).to(device)
    print(f"{m.get_name()} \trank = {r} \t sl={sl_n64} \t fl={fl_n64} \tcl={conv_dims}")
    models_n64.append(m)
    run_details[m.get_name()] = m.get_hyperparameters()

# n128
models_n128 = []
for r in ranks_n128:
    for sl in sls_n128:
        for fl in fls_n128:
            # m = Fork(r, mat_size, sl, fl).to(device)
            # print(f"{m.get_name()} \trank = {r} \t sl={sl} \t fl={fl}")
            # models.append(m)
            # run_details[m.get_name()] = m.get_hyperparameters()

            m = ConvMF(r, mat_size_2, sl, fl, conv_dims).to(device)
            print(f"{m.get_name()} \trank = {r} \t sl={sl} \t fl={fl} \tcl={conv_dims}")
            models_n128.append(m)
            run_details[m.get_name()] = m.get_hyperparameters()

print(run_details)

if not os.path.exists(output_run_dir):
    os.mkdir(output_run_dir)

# Save details
with open(os.path.join(output_run_dir, f"details_{runname}.json"), "w" ) as write:
    json.dump(run_details, write, indent=2 )
####################


#### Run models ####
for model in models_n128:
    writer = SummaryWriter(os.path.join(tensorboard_dir, f'{machine}_{model.get_name()}_{runname}'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_model(model=model,
                optimizer=optimizer,
                train_data=train_dataloader_n128,
                validate_data=validation_dataloader_n128,
                loss_fcn=loss_fcn_neg,
                output_run_dir=output_run_dir,
                **run_details["run_params"],
                writer=writer,
                load=False
               )
    writer.close()

for model in models_n64:
    writer = SummaryWriter(os.path.join(tensorboard_dir, f'{machine}_{model.get_name()}_{runname}'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_model(model=model,
                optimizer=optimizer,
                train_data=train_dataloader_n64,
                validate_data=validation_dataloader_n64,
                loss_fcn=loss_fcn_neg,
                output_run_dir=output_run_dir,
                **run_details["run_params"],
                writer=writer,
                load=False
               )
    writer.close()
####################


#### Winners ####
# TODO






