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
