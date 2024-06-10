<h1 align="center">Graph State Space Convolution (GSSC)</h1>
<p align="center">
    <a href=""><img src="https://img.shields.io/badge/-arXiv-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
    <a href="https://github.com/Graph-COM/GSSC"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
</p>


This repository contains the official implementation of GSSC as described in the paper: [What Can We Learn from State Space Models for Machine Learning on Graphs?]() by Yinan Huang*, Siqi Miao*, and Pan Li.

(*Equal contribution, listed in alphabetical order)

## Installation
All required packages are listed in `environment.yml`.

## Running the code
Replace `--cfg` with the path to the configuration file and `--device` with the GPU device number like below:
```
python main.py --cfg configs/GSSC/peptides-func-GSSC.yaml --device 0 wandb.use False
```
This command will train the model on the `peptides-func` dataset using the GSSC method with default hyperparameters.

## Reproducing the results
We use wandb to log and sweep the results. To reproduce the reported results, one needs to create and login to a wandb account. Then, one can launch the sweep using the configuration files in the `configs` directory.
For example, to reproduce the tuned results of GSSC on the `peptides-func` dataset, one can launch the sweep using `configs/GSSC/peptides-func-GSSC-tune.yaml`.

## Acknowledgement
This repository is built upon [GraphGPS (Rampasek et al., 2022)](https://github.com/rampasek/GraphGPS).
