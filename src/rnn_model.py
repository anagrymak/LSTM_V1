import sys
import json
import logging
import warnings
import argparse
import typing as T
from pathlib import Path
from functools import partial

import defopt
import joblib
import numpy as np
import pandas as pd
import torch
import optuna
from darts import metrics
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from dask.distributed import Client, LocalCluster, as_completed
from darts.models import RNNModel

import optuna.visualization as vis

from resrnn import ResRNNModel
from utils import historical_forecasts, plot_forecasts, prepare_col, append_covariates


parse_known_args_orig = argparse.ArgumentParser.parse_known_args


# adapted from https://gist.github.com/kgaughan/b659d6c173b5a2203dfb3ae225135cce
def parse_known_args(self, args=None, namespace=None):
    """monkey-patched Argumentparser method to add a configuration file option"""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config", type=argparse.FileType("r"))
    args, remaining = parse_known_args_orig(parser, args)

    # use configuration file as defaults for the parser
    if args.config:
        try:
            defaults = json.load(args.config)
        except ValueError:
            print(
                f"ERROR: Could not parse configuration file '{args.config}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        finally:
            args.config.close()
        self.set_defaults(**defaults)

        for action in self._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for subparser in action.choices.values():
                subparser.set_defaults(**defaults)

    self.add_argument(
        "-c", "--config", type=argparse.FileType("r"), help=".json configuration file"
    )
    args, remaining = parse_known_args_orig(self, remaining)

    return args, remaining


argparse.ArgumentParser.parse_known_args = parse_known_args


class MissingMaterial(ValueError):
    pass


def check_material(dset, materials):
    """check if all materials are in the dataset"""
    all_materials = set(dset["material"])
    missing_materials = [mat for mat in materials if mat not in all_materials]
    if missing_materials:
        err_msg = ", ".join(missing_materials)
        raise MissingMaterial(f"Missing materials in the input dataset ({err_msg})!")


class MissingCovariates(ValueError):
    pass


def check_covariates(covariates, materials):
    missing_materials = [mat for mat in materials if mat not in covariates.index]
    if missing_materials:
        err_msg = ", ".join(missing_materials)
        raise MissingCovariates(
            f"Missing materials in the covariates dataset ({err_msg})!"
        )


def fit(
    dataset_path: Path,
    results_path: Path,
    *,
    training_materials: T.Sequence[str] | None = None,
    warmup: int = 500,
    hidden_dim: int = 64,
    n_rnn_layers: int = 1,
    dropout: float = 0.0,
    batch_size: int = 32,
    n_epochs: int = 2,
    lr: float = 1e-3,
    seed: int = 42,
    use_proba: bool = False,
    use_gpu: bool = False,
    verbosity: int = 0,
    float32_precision: str = "high",
    path_covariates: Path | None = None,
    positive: bool = True,
    gradient_clip: float | None = None,
    training_length: int | None = None,
):
    """Fit an LSTM model to input wear timeseries

    :raises MissingMaterial: missing training material in the input dataset
    :raises MissingCovariates: missing training material in the covariates dataset
    """
    inputs = locals()

    torch.set_float32_matmul_precision(float32_precision)

    results_path.mkdir(parents=True, exist_ok=True)
    with (results_path / "params.json").open("w") as fd:
        json.dump(inputs, fd, indent=4, default=str)

    if verbosity <= 1:
        # disable warnings from PyTorch Lightning and lower level of logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=PossibleUserWarning)

    dset_train = pd.read_csv(dataset_path)

    if training_materials is not None:
        check_material(dset_train, training_materials)
        dset_train = dset_train[dset_train["material"].isin(training_materials)]

    groups_train = dset_train.groupby(["material", "sample"])
    series_train = [prepare_col(group, "Wear Loss [mm]") for _, group in groups_train]
    series_train_idx = [prepare_col(group, "Time [min]") for _, group in groups_train]

    if path_covariates is not None:
        covariates = pd.read_csv(path_covariates).set_index("material")
        check_covariates(covariates, dset_train["material"].unique())
        str_cols = covariates.columns[covariates.dtypes == "object"]
        covariates = pd.get_dummies(covariates, columns=str_cols, drop_first=True)

        series_train_idx = [
            append_covariates(train_idx, covariates.loc[material])
            for train_idx, ((material, _), _) in zip(series_train_idx, groups_train)
        ]

    # TODO fit on one trace? or max of all traces?
    scaler = Scaler().fit([series_train[0], series_train_idx[0]])
    series_train_scaled, series_train_idx_scaled = zip(
        *[scaler.transform(series) for series in zip(series_train, series_train_idx)]
    )

    likelihood = GaussianLikelihood() if use_proba else None
    pl_trainer_kwargs = {"enable_progress_bar": verbosity >= 1}

    if gradient_clip is not None:
        pl_trainer_kwargs["gradient_clip_val"] = gradient_clip

    if use_gpu:
        pl_trainer_kwargs["accelerator"] = "gpu"
        pl_trainer_kwargs["devices"] = 1

    if training_length is None:
        training_length = warmup
    else:
        series_train_scaled = [
            serie for serie in series_train_scaled if len(serie) > training_length
        ]
        series_train_idx_scaled = [
            serie for serie in series_train_idx_scaled if len(serie) > training_length
        ]

    model_class = ResRNNModel if positive else RNNModel
    model = model_class(
        model="LSTM",
        input_chunk_length=warmup,
        training_length=training_length,
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer_kwargs={"lr": lr},
        likelihood=likelihood,
        random_state=seed,
        force_reset=True,
        log_tensorboard=True,
        show_warnings=True,
        pl_trainer_kwargs=pl_trainer_kwargs,
        work_dir=results_path,
        model_name="lightning",
    )

    model.fit(series_train_scaled, future_covariates=series_train_idx_scaled)

    model.save(str(results_path / "model.pt"))
    joblib.dump(scaler, results_path / "scaler.pkl.gz", compress=9)


def score_forecasts(forecasts, n_warmup, scores_functions):
    scores = []
    for (material, sample), (serie, serie_forecasts) in forecasts.items():
        for warmup, serie_forecast in zip(n_warmup, serie_forecasts):
            serie_wear = serie["Wear Loss [mm]"]
            context = {"material": material, "sample": sample, "warmup": warmup}
            for name, func in scores_functions.items():
                score_end = {
                    **context,
                    "metric": f"{name}_end",
                    "value": func(serie_wear[-1], serie_forecast[-1]),
                }
                score_avg = {
                    **context,
                    "metric": f"{name}_avg",
                    "value": func(serie_wear, serie_forecast),
                }
                scores.extend([score_end, score_avg])
    scores = pd.DataFrame.from_dict(scores)
    return scores


def forecasts_to_frame(serie, series_forecast, n_warmup):
    serie = serie.pd_dataframe().rename(columns={"Wear Loss [mm]": "observations"})
    series_forecast = [
        forecast.pd_dataframe().iloc[:, 0].rename(f"warmup = {n}")
        for forecast, n in zip(series_forecast, n_warmup)
    ]
    return pd.concat([serie] + series_forecast, axis=1)


def evaluate(
    dataset_path: Path,
    model_path: Path,
    results_path: Path,
    *,
    materials: T.Iterable[str] | None = None,
    n_warmup: T.Iterable[int] = (1000,),
    n_col: int = 3,
    path_covariates: Path | None = None,
    positive: bool = True,
):
    # disable warnings from PyTorch Lightning and lower level of logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=PossibleUserWarning)

    scaler = joblib.load(model_path / "scaler.pkl.gz")
    model_class = ResRNNModel if positive else RNNModel
    model = model_class.load(str(model_path / "model.pt"))
    model.trainer_params["enable_progress_bar"] = False

    results_path.mkdir(parents=True, exist_ok=True)

    dset = pd.read_csv(dataset_path)
    if materials is not None:
        check_material(dset, materials)
        dset = dset[dset["material"].isin(materials)]
    groups = dset.groupby(["material", "sample"])

    if path_covariates is None:
        covariates = None
    else:
        covariates = pd.read_csv(path_covariates).set_index("material")
        check_covariates(covariates, dset["material"].unique())
        str_cols = covariates.columns[covariates.dtypes == "object"]
        covariates = pd.get_dummies(covariates, columns=str_cols, drop_first=True)

    n_samples = 1 if model.likelihood is None else 50
    forecasts = {
        key: historical_forecasts(
            model,
            scaler,
            group,
            n_warmup,
            n_samples=n_samples,
            covariates=None if covariates is None else covariates.loc[key[0]],
        )
        for key, group in groups
    }

    fig = plot_forecasts(forecasts, n_warmup, n_col=n_col)
    fig.savefig(results_path / "predictions.png", bbox_inches="tight")

    scores_functions = {
        "MAE": metrics.mae,
        "RMSE": metrics.rmse,
        "MAPE": metrics.mape,
    }
    scores = score_forecasts(forecasts, n_warmup, scores_functions)
    scores.to_csv(results_path / "scores.csv", index=False)

    dset_forecast = [
        forecasts_to_frame(*series, n_warmup).assign(material=material, sample=sample)
        for (material, sample), series in forecasts.items()
    ]
    dset_forecast = pd.concat(dset_forecast)
    dset_forecast.to_csv(results_path / "predictions.csv")


def gather_n_remove(futures, loop):
    """return Dask futures results when ready and cancel them"""
    for future, result in as_completed(futures, loop=loop, with_results=True):
        future.cancel()
        yield result


def objective_(
    trial,
    dataset_path,
    results_path,
    *,
    materials,
    n_epochs_max,
    warmup_max,
    hidden_dim_max,
    n_warmup,
    metric,
    cluster_addr,
    path_covariates,
    positive,
    **fit_kwargs,
):
    workdir = results_path / f"trial_{trial.number}"
    workdir.mkdir(exist_ok=True, parents=True)

    if materials is None:
        dset = pd.read_csv(dataset_path)
        materials = dset["material"].unique()

    if (warmup := fit_kwargs.pop("warmup")) is None:
        warmup = trial.suggest_int("warmup", 10, warmup_max)

    if (hidden_dim := fit_kwargs.pop("hidden_dim")) is None:
        hidden_dim = trial.suggest_int("hidden_dim", 16, hidden_dim_max)

    if (n_rnn_layers := fit_kwargs.pop("n_rnn_layers")) is None:
        n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 3)

    if (dropout := fit_kwargs.pop("dropout")) is None:
        dropout = 0.0 if n_rnn_layers == 1 else trial.suggest_float("dropout", 0, 0.5)

    if (batch_size := fit_kwargs.pop("batch_size")) is None:
        batch_size = trial.suggest_int("batch_size", 1, 64)

    if (n_epochs := fit_kwargs.pop("n_epochs")) is None:
        n_epochs = trial.suggest_int("n_epochs", 1, n_epochs_max)

    if (lr := fit_kwargs.pop("lr")) is None:
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    if (gradient_clip := fit_kwargs.pop("gradient_clip")) is None:
        gradient_clip = trial.suggest_float("gradient_clip", 0.1, 10, log=True)

    fit_trial = partial(
        fit,
        dataset_path,
        warmup=warmup,
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        lr=lr,
        gradient_clip=gradient_clip,
        verbosity=0,
        **fit_kwargs,
    )

    def fit_n_evaluate(train_materials, val_material):
        fold_path = workdir / val_material
        fit_trial(results_path=fold_path, training_materials=train_materials)
        evaluate(
            dataset_path,
            fold_path,
            fold_path,
            materials=[val_material],
            n_warmup=n_warmup,
            path_covariates=path_covariates,
            positive=positive,
        )
        scores = pd.read_csv(fold_path / "scores.csv")
        avg_metric = scores[scores["metric"] == metric]["value"].mean()
        return avg_metric

    materials = set(materials)

    with Client(cluster_addr) as client:
        futures = client.map(lambda x: fit_n_evaluate(materials - {x}, x), materials)
        fold_metrics = list(gather_n_remove(futures, loop=client.loop))

    avg_metric = np.mean(fold_metrics)

    return avg_metric


def tune(
    dataset_path: Path,
    results_path: Path,
    *,
    materials: T.Iterable[str] | None = None,
    n_warmup: T.Iterable[int] = (1000,),
    n_trials: int = 1,
    metric: str = "MAE_end",
    use_gpu: bool = False,
    use_proba: bool = False,
    sampler: str = "botorch",
    seed: int = 42,
    n_jobs: int = 1,
    n_workers: int = 1,
    memory_limit: str = "2GB",
    scheduler_file: Path | None = None,
    float32_precision: str = "high",
    path_covariates: Path | None = None,
    positive: bool = True,
    training_length: int | None = None,
    warmup_max: int = 100,
    warmup: int | None = None,
    hidden_dim_max: int = 256,
    hidden_dim: int | None = None,
    n_rnn_layers: int | None = None,
    dropout: float | None = None,
    batch_size: int | None = None,
    n_epochs_max: int = 5,
    n_epochs: int | None = None,
    lr: float | None = None,
    gradient_clip: float | None = None,
):
    inputs = locals()

    results_path.mkdir(parents=True, exist_ok=True)
    with (results_path / "params.json").open("w") as fd:
        json.dump(inputs, fd, indent=4, default=str)

    if scheduler_file is not None:
        with scheduler_file.open("r") as fp:
            scheduler_infos = json.load(fp)
        cluster_addr = scheduler_infos["address"]

    elif n_workers > 1:
        cluster = LocalCluster(
            processes=True,
            n_workers=n_workers,
            memory_limit=memory_limit,
            threads_per_worker=1,
            local_directory=results_path,
        )
        cluster_addr = cluster.scheduler_address
        print(f"Dask dashboard address: {cluster.dashboard_link}", flush=True)

    else:
        cluster = LocalCluster(
            processes=False,
            n_workers=1,
            memory_limit=memory_limit,
            threads_per_worker=1,
            local_directory=results_path,
        )
        cluster_addr = cluster.scheduler_address
        print(f"Dask dashboard address: {cluster.dashboard_link}", flush=True)

    objective = partial(
        objective_,
        dataset_path=dataset_path,
        results_path=results_path,
        materials=materials,
        n_epochs_max=n_epochs_max,
        warmup_max=warmup_max,
        hidden_dim_max=hidden_dim_max,
        n_warmup=n_warmup,
        metric=metric,
        cluster_addr=cluster_addr,
        path_covariates=path_covariates,
        positive=positive,
        use_gpu=use_gpu,
        use_proba=use_proba,
        seed=seed,
        float32_precision=float32_precision,
        training_length=training_length,
        warmup=warmup,
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        lr=lr,
        gradient_clip=gradient_clip,
    )

    if sampler == "botorch":
        sampler = optuna.integration.BoTorchSampler(seed=seed)
    elif sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler '{sampler}'.")

    storage = f"sqlite:///{results_path.resolve() / 'study.db'}"
    study = optuna.create_study(
        direction="minimize",
        study_name="darts_lstm_multiple",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )
    n_runs = max(0, n_trials - len(study.get_trials()))
    study.optimize(objective, n_trials=n_runs, n_jobs=n_jobs, catch=RuntimeError)

    Path(results_path / "best_trial").symlink_to(f"trial_{study.best_trial.number}")

    with (results_path / "best_hyperparams.json").open("w") as fd:
        json.dump(study.best_params, fd, indent=4, default=str)

    fig = vis.plot_parallel_coordinate(study)
    fig.write_html(results_path / "parallel_coordinates.html")


# TODO predict function


if __name__ == "__main__":
    defopt.run([fit, evaluate, tune], no_negated_flags=True)
