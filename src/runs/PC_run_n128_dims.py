#!/usr/bin/env python
# coding: utf-8

# # Initial Work for Test Network
# 

# ## Notebook Setup

# <div hidden>
# $\usepackage{cancel}$
# $\usepackage{amssymb}$
# </div>

# In[1]:


# Required packages
#!pip install "../requirements.txt"


# In[1]:


# Visualize tensorboard in the notebook
get_ipython().run_line_magic('load_ext', 'tensorboard')
# In order to force reload any changes done to the models package files
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


## Allow import from our custom lib python files
import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(os.path.join(module_path, "lib"))


# In[3]:


from models.simpleFork import Simple, Fork
from models.danmf import DANMF
from models.convmf import ConvMF
from datasets.gen_plasma_1d import GenPlasma1DDataset
from framework.saveload import load_checkpoint, load_trained_model
from framework.params import * # device, use_cuda, Checkpoint, various saving strs
from framework.train import train_model

from timeit import default_timer
import re
from datetime import datetime
import json

import torch
import onnx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, diagsvd
from torch import nn
from torch.utils.data import DataLoader, random_split
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from torch import profiler


# ### Per-run user defined variables

# In[4]:


# Define the machine being used
machine = "PC"
# machine = "Macbook"

# Paths for files
data_dir = os.path.normpath("../../data")
output_dir = os.path.join(data_dir, "output")
tensorboard_dir = os.path.join(output_dir, "tensorboard")

# Dataset params
batch_size = 25
mat_size = [64, 128]
mat_dirs = [os.path.join(data_dir, "gen_plasma_n64", f"mat_{i}") for i in range(3)]

# Device comes from framework.params
print(f"Using {device} device")


# ## Data 
# 
# We remove the channels in the images (convert to grayscale) to more closely mimic the actual plasma dataset
# 
# * [Helpful Link](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
# 
# 
# #### Normalization
# * Normalize each image with respect to its unit Frobenius norm

# ### Dataset

# In[5]:


# Load and split the data, and prep for being fed into the NN
data = GenPlasma1DDataset(mat_dirs)
# Divide data into train, validation, test
train_data, validation_data, test_data = random_split(data, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), drop_last=True, num_workers=4)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), drop_last=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)


# In[6]:


plt.imshow(data[2669])
print(len(data), type(data[0]), data[244].shape)
print(data[2000])


# ## Hyperparameters
# 
# 
# ### Structure
# 
# #### DANMF Implementation from papers
# 
# * **Non-alternating/nonnegative Matrix Factorization (NMF)**: Classical deep NMF for a non-negative matrix X ∈ Rm×n + can be described as follows: first, X is decomposed as X ≈ U0V 0 where U 0 ∈ Rm×r0 +  and V 0 ∈ Rr0×n + . The coding matrix V 0 is then further decomposed as V 0 ≈ U 1V 1 where U 1 ∈ Rr0×r1 +  and V 1 ∈ Rr1×n +. The procedure is repeated until a pre-fixed number of layers is reached. [[2]](#r2)
# * **deep alternating non-negative matrix factorisation (DA-NMF)**: we propose to factorise the basis and coding matrices in an alternating order along the layers. [[2]](#r2)
#     * Uses approx 6-8 layers
# 
# <img src="images/r2_fig1.png" alt="Drawing" style="width: 400px;"/>
# <!-- ![image.png](images/r2_fig1.png) -->
# 
# 
# ### L-System
# * Using L (Lindenmayer) Systems to define strucure?
# * Node types:
#     * Terminating (1 -> 0)
#     * Splitting (1 -> 2)
#     * Unifying (2 ->1)
#     * Direct (1 -> 1)
#     * Starting (0 -> 1)
# 
# 
# ### Loss Function
# We use methods defined in [[1]](#r1) to define our loss function <!-- $\mathcal{L}$ \\ -->
# 
# $$
# \min_{\substack{U \in \real^{m\times r} \\ V \in \real^{r\times n}}} ||X - ReLU(UV)||^2_F ,
# $$
# Where we are finding the square Frobenius norm of the difference between the original matrix $X$ and the rectified linear low rank representation matrices $UV$
# 
# #### From paper
# 
# Eq. 10-11 [[2]](#r2)
# 
# 
# 
# ### Optimizer
# 
# Adam works fine / is standard
# 
# 

# ## Model Definition

# ### Multiple models hyperparams

# ### Models Creation

# In[7]:


## Params
ranks = [5, 7, 8, 9, 10, 11]

stem_layer_dims = [500, 200]

fork_layer_dims = [200, 300]

conv_dims = [[5, 1, 3, 1], [3, 1, 0, 1]]

run_params = dict(
                epochs=120,
                checkpoint_at=40,
                batch_pr=40,
                batch_size=batch_size,
                runname="plasma_mf_2",
                machine=machine,
                )


# Save details
run_details = {"run_params": run_params}
runname = run_details["run_params"]["runname"]
output_run_dir = os.path.join(output_dir, f"run_{runname}")


# In[8]:


models = []

for r in ranks:
    m = ConvMF(r, mat_size, stem_layer_dims, fork_layer_dims, conv_dims).to(device)
    print(f"{m.get_name()} \trank = {r} \t sl={stem_layer_dims} \t fl={fork_layer_dims} \tcl={conv_dims}")
    models.append(m)
    run_details[m.get_name()] = m.get_hyperparameters()

# m = Fork(nn_rank, img_size, stem_layer_dims[0], fork_layer_dims[0]).to(device)
# print(f"{m.get_name()} \trank = {nn_rank}, \tcl={0} \t sl={stem_layer_dims[0]}, \t fl={fork_layer_dims[0]}")
# models.append(m)
# run_details[m.get_name()] = m.get_hyperparameters()
# for i, cl in enumerate(conv_dims):
#     if i == 0:
#         for sl in stem_layer_dims:
#             for fl in fork_layer_dims:
#                 m = ConvMF(nn_rank, img_size, sl, fl, cl).to(device)
#                 print(f"{m.get_name()} \trank = {nn_rank}, \tcl={cl} \t sl={sl}, \t fl={fl}")
#                 models.append(m)
#                 run_details[m.get_name()] = m.get_hyperparameters()
#     else:
#         m = ConvMF(nn_rank, img_size, stem_layer_dims[0], fork_layer_dims[0], cl).to(device)
#         print(f"{m.get_name()} \trank = {nn_rank}, \tcl={cl} \t sl={stem_layer_dims[0]}, \t fl={fork_layer_dims[0]}")
#         models.append(m)
#         run_details[m.get_name()] = m.get_hyperparameters()


# ## Run models

# In[9]:


run_details


# In[10]:


if not os.path.exists(output_run_dir):
    os.mkdir(output_run_dir)

# Save details
with open(os.path.join(output_run_dir, f"details_{runname}.json"), "w" ) as write:
    json.dump(run_details, write, indent=2 )


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


# ### Single model running

# In[8]:


# model = Fork(nn_rank, img_size, [500, 200], [200, 300]).to(device)
# model = model_danmf
# runname = "Better_opt"
# output_run_dir = output_dir
# writer = SummaryWriter(os.path.join(tensorboard_dir, f'{machine}_{model.get_name()}_{runname}'))

# # Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# train_model(model, optimizer, output_run_dir, epochs=120, checkpoint_at=30, batch_pr=40, writer=writer, load=False)

## Profiling code
# with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA] if use_cuda else [profiler.ProfilerActivity.CPU],
#                          record_shapes=False,
#                          profile_memory=True,
#                          # use_cuda=use_cuda,
#                          schedule=torch.profiler.schedule(
#                             wait=1,
#                             warmup=1,
#                             active=2,
#                             repeat=1),
#                          on_trace_ready=trace_handler
#                          ) as prof:
#         with profiler.record_function("train_model"):
#             train_model(15, model, optimizer, checkpoint_at=5, writer=writer, load=False, profiler=prof)






# #### Winners
# 
# * Mac_Fork_sdim2-3ebc_fdim2-37e7_Conv_1
# * Mac_ConvMF_cdim2-c00f_Conv_1
# * Mac_ConvMF_cdim2-2d8c_Conv_1
# * Mac_ConvMF_cdim2-f2cd_Conv_1

# ### Setup to load and run against SVD

# In[ ]:


## Params
# runname = "Conv_1"
# model_str_name = "ConvMF_cdim2-f2cd"

# model = load_trained_model(ConvMF, "Mac", runname, model_str_name, output_run_dir)
# print(model, model.get_name())

# writer = SummaryWriter(os.path.join(tensorboard_dir, f'{machine}_{model.get_name()}_{runname}'))

# def save_prof(prof):
#     prof_events = prof.key_averages()

# with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA] if use_cuda else [profiler.ProfilerActivity.CPU],
#                          record_shapes=False,
#                          profile_memory=True,
#                       on_trace_ready=save_prof
#                          ) as prof:
#     U, V = model(next(iter(validation_dataloader)))
#     prof.step()
#     prof_events = prof.key_averages()



# In[ ]:


# U_svd, S, V_svd = svd(imgs[i, :, :])
# # plt.semilogy(S)
# plt.semilogy(np.sqrt(np.linalg.norm(S)**2 - np.cumsum(S**2)) / np.linalg.norm(S))


# # TODO
# 
# * Try saving all CSVs as HDF5 files, load that instead (using h5py)
#     * Or try saving as .mat file, then load with `scipy.io.loadmat`
# * ~~Save hyperparameters, run params to dict (json?)~~
# * ~~Run against bigger networks~~
# * ~~Compare rank vs err (loss) for SVD vs NN, and see how many more ranks it is to get comparable err~~
#     * ~~compare against SVD 4, using rank 6 for NN (or bigger)~~
#     * <img src="images/rank_vs_err.png" alt="Rank vs Error Plot" style="width: 300px;"/>
# * ~~Use a different algo where we pre-give the rank, better alg: `scipy.sparse.linalg.svds`, tol=0.001~~
# * ~~Impose nonneg on U, V BEFORE multiplying~~
# * ~~Generate the data~~
#     * ~~Make a movie of hte fs, to sanity check~~
# * Try supervised learning with `torch.nn.KLDivLoss(reduction = "batchmean")`
#     * Remember needs log input from nn!
#     * Figure out S?
#     * Use the square root of S, then multiply into U and V
# * Research how big 1st layer should be in comparison to dims of  input
# * ~~Go as big for imgs as I can, log space increase in size~~
# * Try running at bigger network and input sizes, (dont even train) and just test timings
# * Try transfer learning: train on smaller input, transfer learn to bigger input (1st and last layers)
# * Log clock time
# * Use a faster loss function
#     * $||X - UV^T||^2$ =  
# * Try timing the loading of the data, maybe use HDF5 for storing and loading
# * Look into DataLoader and see if it reloads the data each time or is inst
#     * manually add time outputs
# * Check/analyze loss based on which initial condition of the dataset, and where in the time sequence in the dataset
# 
# ## Fast Kernel Approx
# * Look into using fast kernel approx on the exponential form (in emails?)
#     * Pairwise mult slow, but pairwise * vector can be approx faster
#     * Look into where you build the exponential form, and how that would fit into the Kernel Transform
# * https://github.com/jpryan1/FastKernelTransform.jl
#     * Using that 
# 
# 
# ## Model: Variations
# * ~~Simple with no nonnegativity constraints (sigmoid activation)~~
#     * ~~Allow negativity at every step except after UV, aka ReLU(UV) or |UV| (relu better)~~
# * ~~Different activation layers~~
# * ~~Different structure~~
#     * ~~Try forking at beginning~~
# * Define new layer
#     * Instead of flattening and using a linear layer, doing a either side multiply of the matrix input: A\*X\*B where A: n_1 x m, X: m x n, B: n x n_2
# * ~~try 2d convolutional layers~~
# 
# ### Structure
# * Try dropout
# * Try regularization
# * Try different measured weight matrix sizes
# 
# ## Optional
# * Randomized Numerical Linear Algebra: Foundations & Algorithms (Per-Gunnar Martinsson, University of Texas at Austin Joel A. Tropp, California Institute of Technology)l
# *  Pad input when not enough imgs for batch
# *  Get access to NERSC?
# 
# ### Data Augmentation
# *  Cropping, rotating, scaling, reflecting

# # Visualization

# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir "../data/output/tensorboard"')


# # References
# 
# 1. <a id='r1'></a> De Handschutter, P., Gillis, N., & Siebert, X. (2021). A survey on deep matrix factorizations. Computer Science Review, 42, 100423. https://doi.org/10.1016/j.cosrev.2021.100423
# 2. <a id='r2'></a> Sun, J., Kong, Q., & Xu, Z. (2022). Deep alternating non-negative matrix factorisation. Knowledge-Based Systems, 251, 109210. https://doi.org/10.1016/j.knosys.2022.109210
# 
# 
# 

# In[ ]:




