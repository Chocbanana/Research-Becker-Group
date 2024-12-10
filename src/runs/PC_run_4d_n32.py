#!/usr/bin/env python
# coding: utf-8
"""Script for running NNs.

For running several NN training/validation, MF on plasma 4D data.
In this run, just testing various ranks for output of the NNs with only
one input size, shape (32, 32, 64, 64) and with the default hyperparameters

Author: Bhavana Jonnalagadda
"""

## Allow import from our custom lib python files
import sys
import os

# Stupid super annoying tensorflow warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

module_path = os.path.abspath(os.path.join('..', "lib"))
if module_path not in sys.path:
    sys.path.append(module_path)
    # print(f"appended {module_path}, {sys.path}")


from models.conv4d import Conv4D
from datasets.gen_plasma_4d import Plasma4DDataset
from framework.params import * # device, use_cuda, Checkpoint, various saving strs
from framework.train import loss_4d, loss_fcn_neg, train_model

import json

import torch
import onnx
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
# from torch import profiler

# Get rid of super annoying deprecation warning
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def run():
    #### Per-run user defined variables ####
    # Define the machine being used
    machine = "PC" # or Macbook or RC
    # Paths for files
    data_dir = os.path.normpath("../../data")
    output_dir = os.path.join(data_dir, "output")
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    # Dataset params
    batch_size = 2 # Batch size impacting OOM errors
    mat_size = [32, 32, 64, 64]
    mat_dirs = [os.path.normpath(data_dir + "/gen_plasma_4d/n32/")]
    ####################

    # Device comes from framework.params
    print(f"Using {device} device")

    # TODO: Move into function
    #### Dataset ####
    # Load and split the data, and prep for being fed into the NN
    data_n32 = Plasma4DDataset(mat_dirs)
    # Divide data into train, validation, test
    train_n32, validation_n32, _ = random_split(data_n32,
                                                [0.7, 0.2, 0.1],
                                                generator=torch.Generator().manual_seed(42))

    train_dataloader_n32 = DataLoader(train_n32,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    # pin_memory=(torch.cuda.is_available()),
                                    drop_last=True,
                                    num_workers=2)
    validation_dataloader_n32 = DataLoader(validation_n32,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        # pin_memory=(torch.cuda.is_available()),
                                        drop_last=True,
                                        num_workers=2)
    ####################

    # Output validate
    print(len(data_n32), type(data_n32[0]), data_n32[244].shape)
    # print(data_n32[1000])


    #### Run params ####
    run_params = dict(
                    epochs=120,
                    checkpoint_at=60,
                    batch_pr=30,
                    batch_size=batch_size,
                    runname="tst_4d",
                    machine=machine,
                    )


    # Save details
    run_details = {"run_params": run_params}
    runname = run_details["run_params"]["runname"]
    output_run_dir = os.path.join(output_dir, f"run_{runname}")
    ####################


    #### NN params ####
    ranks_n32 = [3, 6, 12]
    ####################


    #### Construct models from params ####
    models = []
    for r in ranks_n32:
        m = Conv4D(r).to(device)
        print(f"{m.get_name()} \trank = {r}")
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
        # Set foreach=False to avoid OOM
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, foreach=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, foreach=False)
        train_model(model=model,
                    optimizer=optimizer,
                    train_data=train_dataloader_n32,
                    validate_data=validation_dataloader_n32,
                    loss_fcn=loss_4d,
                    output_run_dir=output_run_dir,
                    **run_details["run_params"],
                    writer=writer,
                    load=False,
                    output_onnx=False
                )
        writer.close()
    ####################

if __name__ == '__main__':
    run()







