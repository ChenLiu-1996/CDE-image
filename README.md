# CDE-image
Chen Liu (chen.liu.cl2482@yale.edu) at Krishnaswamy Lab

## Background
In this work, we aim to extend [Neural CDE](https://arxiv.org/abs/2005.08926) to images!

Neural Controlled Differential Equations (Neural CDE) are a great extension on Neural Ordinary Differential Equations (Neural ODEs). While the solution of Neural ODEs are determined by its initial condition, the solution of Neural CDEs are adjusted based on subsequent observations. By modeling with Neural CDEs, we can predict the future observation at an arbitrary time $t$ based on the initial value as well as the intermediate observations that may be both partially observed and irregularly sampled.

Neural CDEs open a lot of doors to modeling the time-dependent dynamics in multivariate time series data. While the authors provided decent implementations, both for research prototype: [torchcde](https://github.com/patrick-kidger/torchcde) as well as for production-level usage: [diffrax](https://github.com/patrick-kidger/diffrax), so far we found no implementation that allows us to apply Neural CDE on longitudinal image data. Hence we created this repo.


## Usage


## Dependencies
We developed the codebase in a miniconda environment.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name cde-image pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia -c anaconda -c conda-forge
conda activate cde-image
conda install scikit-learn scikit-image pillow matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge
python -m pip install -U albumentations
python -m pip install timm
python -m pip install opencv-python
python -m pip install torchdiffeq
python -m pip install torch-ema
python -m pip install torchcde
```

