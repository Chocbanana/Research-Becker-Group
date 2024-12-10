# Evaluation of data driven low-rank matrix factorization for accelerated solutions of the Vlasov equation

By Bhavana Jonnalagadda

## Quickstart

### Run evaluation notebooks

If you'd just like to be able to run the code that does model analysis, comparison, or plot generation, the data is available and saved as various `*/df_losses.csv` in `data/output/runs/*`. The following notebook should be able to be run:

- [`src/notebooks/paper_figures.ipynb`](https://github.com/Chocbanana/Research-Becker-Group/blob/main/src/notebooks/paper_figures.ipynb): Generates interactive versions of the plots in the paper

### Generate data

The data used in the paper should be stored in `data/gen_plasma*`, unfortunately it was too much and too large data to upload to Github. To generate it, run the matlab code `src/gen_data_matlab/SSPML_CWENO_ht1d.m`, and modfiy as needed to generate the different image sizes and initial conditions.

### Train Models

The files to train the models are located in [`src/runs/`](https://github.com/Chocbanana/Research-Becker-Group/tree/main/src/runs). Each file is self-contained to train and run a set of tests that were tried, and to train models. Their output model files (the `.tar` and `.pt` files that store the model parameter weights and hyperparameters) are stored in `data/output/runs/`.

### Run/evaluate models

Once there exists the model `.tar` and `.pt` files and the `.hdf5` source data, you can run the following notebooks:

- [`main/src/notebooks/EDA_3_bigger.ipynb`](https://github.com/Chocbanana/Research-Becker-Group/blob/main/src/notebooks/EDA_3_bigger.ipynb): Loads the model files and runs them against SVD, along with useful plot generation and general EDA of the model running
- [`main/src/notebooks/EDA_3_data_seq.ipynb`](https://github.com/Chocbanana/Research-Becker-Group/blob/main/src/notebooks/EDA_3_data_seq.ipynb): Same as above but for the sequentially-fed data/extrapolation, generalization testing

## Files and directory structure
