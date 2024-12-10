
from models.base import BaseMF
from .params import *
from framework.saveload import load_checkpoint, save_checkpoint

import os
from timeit import default_timer
from datetime import datetime

import torch
import onnx
import numpy as np
from torch import nn
import tensorly as tl

tl.set_backend('pytorch')
tl.use_static_dispatch()

# Types
from collections.abc import Callable
from torch.utils.data import DataLoader
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from torch import profiler



def loss_fcn_nonneg(X, U, V):
    return torch.mean(
            torch.square(
                torch.linalg.matrix_norm(X - nn.functional.relu(torch.bmm(U, V)),
                                        ord='fro')))

# DO NOT ENFORCE NONNEG
def loss_fcn_neg(X, U, V):
    # Don't enforce non-negativity on UV
    return torch.mean(
        torch.square(
            torch.linalg.matrix_norm(X - torch.bmm(U, V), ord='fro')))


# Go from tucker decomp to approximated full
# TODO: Enforce orthogonality on U_n through loss? sum(square(U_n*U_n^T - I))
# Problem: not square matrices
def loss_4d(X, S, U):
    tucker_batched = torch.stack(tuple(tl.tucker_to_tensor(x) for x in zip(S, zip(*U))))
    return torch.mean(
        torch.square(
            torch.linalg.matrix_norm(X - tucker_batched, ord='fro')))


# loop over the dataset multiple times
def train_model(model: BaseMF,
                optimizer: torch.optim.Optimizer,
                train_data: DataLoader,
                validate_data: DataLoader,
                output_run_dir: str,
                machine: str,
                epochs = 15,
                checkpoint_at = -1,
                load = True,
                batch_pr = 200,
                writer: SummaryWriter = None,
                profiler: profiler.profile = None,
                loss_fcn: Callable = loss_fcn_neg,
                output_onnx=True,
                **kwargs):
    mname = model.get_name()
    start_epoch = -1

    print(f"Training {mname}")

    # Attempt to load the previous checkpoint
    if load:
        checkpoint, statedict = load_checkpoint(mname, machine, output_run_dir)
        if checkpoint and statedict:
            start_epoch = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["opt_state_dict"])
            model.load_state_dict(statedict)
        else:
            print("No checkpoint found to load. Using base model")


    # Save basic hyperparams
    if writer:
        writer.add_hparams(model.get_hyperparameters(True), {})
        writer.flush()

    # Helps somehow? https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    torch.backends.cudnn.benchmark = True

    loss_arr = []
    validation_arr = []
    time_arr = []
    for e in range(start_epoch+1, start_epoch+1 + epochs ):
        model.train()
        running_loss = 0.0
        running_time = 0.0

        for i, data in enumerate(train_data, 0):
            data = data.to(device) # Should already be on the device?

            # Write out a view of the NN graph, just once
            if writer and e == 0 and i == 0 and output_onnx:
                writer.add_graph(model, data)
                writer.flush()
                torch.onnx.export(model, data, os.path.join(output_run_dir, f'{mname}_model.onnx'), input_names=["matrix"], output_names=["V", "U"])

            start_time = default_timer()
            # zero the parameter gradients
            # Supposedly less memory operations with set_to_none=True?
            optimizer.zero_grad(set_to_none=True)
            # forward + backward + optimize
            U, V = model(data)
            # Loss function
            loss = loss_fcn(data, U, V)
            loss.backward()
            optimizer.step()
            running_time = default_timer() - start_time

            running_loss += loss.item()
            # Print and save statistics, every batch_pr amt of data
            if i % batch_pr == batch_pr - 1:
                avg_loss = running_loss / batch_pr
                loss_arr.append(avg_loss)
                avg_time = running_time / batch_pr
                time_arr.append(avg_time)

                # Determine validation loss
                model.eval()
                model.train(False)
                v_arr = []
                with torch.no_grad():
                    for v_data in validate_data:
                        v_data = v_data.to(device)
                        U_v, V_v = model(v_data)
                        v_arr.append(loss_fcn(v_data, U_v, V_v).item())
                validation_arr.append(np.mean(v_arr))
                model.train(True)

                # Write out stats
                print(f"[{e}, {i+1}] loss: {avg_loss}, validation loss: {validation_arr[-1]}, average train time (sec): {avg_time}")
                if writer:
                    writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : validation_arr[-1] },
                            e * len(train_data) + i)
                    writer.add_scalar('Average Train Time (s)', avg_time, e * len(train_data) + i)
                    writer.flush()

                running_loss = 0.0
                running_time = 0.0

            if profiler:
                profiler.step()

        # Save output to checkpoint dict
        if e % checkpoint_at == checkpoint_at - 1:
            save = Checkpoint(mname, e, loss_arr, validation_arr, optimizer.state_dict(), time_arr)
            save_checkpoint(model, machine, output_run_dir, save)
            loss_arr = []
            validation_arr = []
            time_arr = []

    # Always save output at end
    save = Checkpoint(mname, e, loss_arr, validation_arr, optimizer.state_dict(), time_arr)
    save_checkpoint(model, machine, output_run_dir, save)

    print('Finished Training')



def trace_handler(prof):
    table = prof.key_averages().table(sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total", row_limit=10)
    print(table)
    # ff = prof.key_averages()
    # df_table = pd.DataFrame(ff)
    # print(df_table)
    # df_table.to_csv("tst.csv")

    ## TODO: save to file, and tensorboard
