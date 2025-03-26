# LSTM for wear prediction of dental materials

This repository provides code to train LSTM models to predict material wear.


## Installation (on NeSI)

From a terminal, clone this repository:

```
git clone https://github.com/Alexanderyhx/LSTM-ML-for-teeth.git
```

Then change directory

```
cd LSTM-ML-for-teeth
```

and create a Python virtual environment to install all dependencies

```
module purge && module load Python/3.10.5-gimkl-2022a
python3 -m venv venv
venv/bin/pip3 install -r requirements.lock.txt
```

Finally configure it to be a jupyter kernel

```
module purge && module load JupyterLab
nesi-add-kernel -v ./venv ml_tooth Python/3.10.5-gimkl-2022a
```

*Note: to generate `requirements.lock.txt`, create the virtual environment using `requirements.txt` then freeze dependencies using*

```
venv/bin/pip3 list --format freeze > requirements.lock.txt
```


## Getting started (on NeSI)

First, make sure to run notebook [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) to create a curated training dataset.

You can then run one of the provided notebooks to *explore* how the LSTM model works:

- [`notebooks/02_darts_lstm.ipynb`](notebooks/02_darts_lstm.ipynb) illustrates how a simple [Darts](https://unit8co.github.io/darts/) based LSTM model can be trained to extrapolate an inidividual trace,
- [`notebooks/03_tune_darts_lstm.ipynb`](notebooks/03_tune_darts_lstm.ipynb) shows how to tune a Darts LSTM model using [Optuna](https://optuna.org) and [BoTorch](https://botorch.org) to select the best hyperparameters.

In addition, a multiple sequence LSTM model is implemented in the [`src/rnn_model.py`](src/rnn_model.py).
You can use it to train a model from the command line using

```
module purge && module load Python/3.10.5-gimkl-2022a
venv/bin/python3 src/rnn_model.py fit --verbosity 1 --use-gpu results/dataset_minutes.csv RESULTSDIR
```

where `RESULTSDIR` is a folder of your choice to store the results (including the trained model).

It is also possible to use this script to evaluate and tune models using the `evaluate` and `tune` commands.
List all available options (and their default value) for a given command using the `--help` option, for example:

```
venv/bin/python3 src/rnn_model.py fit --help
```

We recommend that you use the provided Slurm scripts to run model tuning on the NeSI HPC platform:

- [`slurm/tune_darts_lstm.sl`](slurm/tune_darts_lstm.sl) uses [`notebooks/03_tune_darts_lstm.ipynb`](notebooks/03_tune_darts_lstm.ipynb) to tune a LSTM model for one material sample,
- [`slurm/tune_darts_lstm_all.sl`](slurm/tune_darts_lstm_all.sl) uses [`notebooks/03_tune_darts_lstm.ipynb`](notebooks/03_tune_darts_lstm.ipynb) to tune a LSTM model for each material sample listed in the file [`slurm/samples.csv`](slurm/samples.csv), as a Slurm job array,
- [`slurm/tune_darts_lstm_multiple.sl`](slurm/tune_darts_lstm_multiple.sl) uses [`src/rnn_model.py`](src/rnn_model.py) to tune a multiple sequence LSTM model for a set of materials.
- [`slurm/tune_darts_lstm_covariates.sl`](slurm/tune_darts_lstm_covariates.sl) also tunes a multiple sequence LSTM model, with an additional set of covariates.
- [`slurm/fit_darts_lstm_multiple.sl`](slurm/fit_darts_lstm_multiple.sl) fits a multiple sequence LSTM model for a set of materials.

Before using one of these script, make sure to edit it to set parameters you want to test and save.

Next open a terminal, navigate to your folder and submit the Slurm job script to tune the model, for example:

```
sbatch slurm/tune_darts_lstm_multiple.sl
```

You can then monitor the job progress in the queue using

```
squeue --me
```

When the job is running, outputs are recorded in the `logs` folder, as `logs/JOBID-tune_lstm.sl.out` where `JOBID` is the job ID number. All results (model, figures, etc.) are saved a subfolder of the `results` folder, named `JOBID-JOBNAME`.
