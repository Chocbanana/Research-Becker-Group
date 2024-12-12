# Evaluation of data driven low-rank matrix factorization for accelerated solutions of the Vlasov equation

By Bhavana Jonnalagadda

## Quickstart

### Run evaluation notebooks

If you'd just like to be able to run the code that does model analysis, comparison, or plot generation, the data is available and saved as various `*/df_losses.csv` in `data/output/runs/*`. The following notebook should be able to be run:

- [`src/notebooks/paper_figures.ipynb`](https://github.com/Chocbanana/Research-Becker-Group/blob/main/src/notebooks/paper_figures.ipynb): Generates interactive versions of the plots in the paper

### Generate data

The data used in the paper should be stored in `data/gen_plasma*`, unfortunately it was too much and too large data to upload all of it to Github. To generate it, run the matlab code `src/gen_data_matlab/SSPML_CWENO_ht1d.m`, and modfiy as needed to generate the different image sizes and initial conditions. Some data is present already for testing purposes, in `/data/gem_plasma_n64/mat_0`.

### Train Models

The files to train the models are located in [`src/runs/`](https://github.com/Chocbanana/Research-Becker-Group/tree/main/src/runs). Each file is self-contained to train and run a set of tests that were tried, and to train models. Their output model files (the `.tar` and `.pt` files that store the model parameter weights and hyperparameters) are stored in `data/output/runs/`.

### Run/evaluate models

Once there exists the model `.tar` and `.pt` files and the `.hdf5` source data, you can run the following notebooks:

- [`main/src/notebooks/EDA_3_bigger.ipynb`](https://github.com/Chocbanana/Research-Becker-Group/blob/main/src/notebooks/EDA_3_bigger.ipynb): Loads the model files and runs them against SVD, along with useful plot generation and general EDA of the model running
- [`main/src/notebooks/EDA_3_data_seq.ipynb`](https://github.com/Chocbanana/Research-Becker-Group/blob/main/src/notebooks/EDA_3_data_seq.ipynb): Same as above but for the sequentially-fed data/extrapolation, generalization testing

## Files and directory structure

**`/data`**: The generated data that was trained/tested on, the output from the models, the model parameter weights, and misc logs.

- **`/data/output`**: Output from training models, and storing of analysis loss values.
  - **`/data/output/run_*`**: Output from running a training suite (and from loss exctraction/saving) as described above using the `/src/runs/` files or from running the analysis notebooks in `/src/notebooks/EDA_*.ipynb`.
  - **`/data/output/tensorboard`**: The tensorboard-formatted data output from running trainings.
- **`/data/gem_plasma_n64`**: Generated data of size 64x128. `mat_0` is present, and has the generated data for the strong1d problem (order 5). The other mats (`mat_1`, etc) can be generated from the matlab code as described above.
- **`/data/gen_plasma_n128`**: Generated data of size 128x256.
- **`/data/*{.mp4, .webm}`**: The generated data as a video.
- **`/data/tb_data`**: Misc data taken from the tensorboard-formatted data and stored as json files instead.
- **`/data/images_96x54`**: Old data used to test out initial configurations of the models, just to validate learning was happening. Not used at all for the paper.


**`/src`**: All the source code used.

- **`/src/gen_data_matlab/SSPML_CWENO_ht1d.m`**: The matlab code used to generate the simulated plasma data.
- **`/src/lib`**: All the python code that defines: the NN models, the datasets, training code, saving/loading the models, and more. Built using Pytorch.
- **`/src/notebooks`**: Jupyter notebooks for model analysis, along with some initial notebooks that were used for training the initial tests of the models.
- **`/src/runs`**: The python files which are self-contained scripts to train a specific suite of models for a specific test, whose purpose is listed as a comment at the top of the file.
- **`/src/rc_scripts`**: Scripts to run the training files on the CU Boulder research computing cluters, CURC.


**`/rc_files`**: Supplementary files needed to run the code on the CU Boulder research computing cluters, CURC.


**`/paper`**: Supplementary files and information used for the paper, mostly just figures.


