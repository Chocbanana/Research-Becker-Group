#!/usr/bin/env python
# coding: utf-8
"""Script for running NNs.

Run on n 128 x 256 2D matrices, with the training data being only the first 70% of the data
AND randomly generated 1000-length sequences of data.
Thus the validation + test data will be the "hardest" last 30% matrices that have not been seen
by the NN. The first 70% of data (and randomly generated seqs) is still fed shuffled during training in order to prevent
bias in gradient descent.

Author: Bhavana Jonnalagadda
"""

## Allow import from our custom lib python files
from ast import Sub
import sys
import os

# Stupid super annoying tensorflow warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

module_path = os.path.abspath(os.path.join('..', "lib"))
if module_path not in sys.path:
    sys.path.append(module_path)
    # print(f"appended {module_path}, {sys.path}")


# module_path = os.path.abspath("D:\OneDrive - UCB-O365\Work\Research-Becker-Group\src\lib")
# if module_path not in sys.path:
#     sys.path.append(module_path)
#     # print(f"appended {module_path}, {sys.path}")
#     # "D:\OneDrive - UCB-O365\Work\Research-Becker-Group\src\lib"

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
from torch.utils.data import DataLoader, random_split, Subset
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
# from torch import profiler

def run():
    #### Per-run user defined variables ####
    # Define the machine being used
    machine = "PC" # Macbook
    # Paths for files
    # data_dir = os.path.normpath("D:\OneDrive - UCB-O365\Work\Research-Becker-Group\data")
    data_dir = os.path.normpath("../../data")
    output_dir = os.path.join(data_dir, "output")
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    # Dataset params
    batch_size = 50
    mat_size = [128, 256]
    mat_dirs = [os.path.normpath(data_dir + "/gen_plasma_n128/mat_hdf5/")] + [
                os.path.normpath(data_dir + f"/gen_plasma_n128/rand_{i}/") for i in range(1, 16)
                ]
    ####################

    # Device comes from framework.params
    print(f"Using {device} device")

    # TODO: Move into function
    #### Dataset ####
    # Load and split the data, and prep for being fed into the NN
    data_n128 = GenPlasma1DDataset(mat_dirs)

    # Get last 30% of "real" data for validation/test, rest as train
    data_len = 16294
    test_inds = [int(data_len * 0.7), data_len]

    # Divide data into train, validation, test
    # train_ind = int(len(data_n128) * 0.7)
    train_n128 = Subset(data_n128, list(range(test_inds[0])) + list(range(test_inds[1] + 1, len(data_n128))))
    validation_n128 = Subset(data_n128, range(test_inds[0], test_inds[1]))

    train_dataloader_n128 = DataLoader(train_n128, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), drop_last=True, num_workers=4)
    validation_dataloader_n128 = DataLoader(validation_n128, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), drop_last=True, num_workers=4)
    ####################

    # Output validate
    plt.imshow(data_n128[2669])
    print(len(data_n128), len(train_n128), type(data_n128[0]), data_n128[244].shape)
    print(len(validation_n128), type(validation_n128[0]), validation_n128[244].shape)
    print(data_n128[2000])


    #### Run params ####
    run_params = dict(
                    epochs=120,
                    checkpoint_at=30,
                    batch_pr=200,
                    batch_size=batch_size,
                    runname="plasma_data_gen_seq_less",
                    machine=machine,
                    )


    # Save details
    run_details = {"run_params": run_params}
    runname = run_details["run_params"]["runname"]
    output_run_dir = os.path.join(output_dir, f"run_{runname}")
    ####################


    #### NN params ####
    ranks_n128 = [6, 12, 30]

    # kernel_size, stride, padding, dilation
    conv_dims = [[5, 1, 3, 1], [3, 1, 0, 1]]
    ####################


    #### Construct models from params ####
    # n128
    models_n128 = []
    for r in ranks_n128:
        m = ConvMF(r, mat_size, conv_dims=conv_dims).to(device)
        print(f"{m.get_name()} \trank = {r} \tcl={conv_dims}")
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
####################



if __name__ == '__main__':
    run()


