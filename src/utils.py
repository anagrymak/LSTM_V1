from itertools import zip_longest

import numpy as np
from darts import TimeSeries

import matplotlib.pyplot as plt


def prepare_col(dset, col):
    serie = TimeSeries.from_dataframe(dset, time_col="Time [min]", value_cols=col)
    return serie.astype(np.float32)


def append_covariates(series, covariates):
    covariates_series = TimeSeries.from_values(
        np.repeat(covariates.values[None, :], len(series), axis=0)
    )
    return series.stack(covariates_series.astype(np.float32))


def historical_forecasts(
    model, scaler, dset, n_warmup, n_samples=1, covariates=None, n_lags=0
):
    serie = prepare_col(dset, ["Line Distance [m]", "Wear Loss [mm]"])
    serie_wear = serie["Wear Loss [mm]"]
    serie_idx = prepare_col(dset, "Time [min]")

    if covariates is not None:
        serie_idx = append_covariates(serie_idx, covariates)

    serie_scaled, serie_scaled_idx = scaler.transform([serie_wear, serie_idx])

    series_forecast = []
    for n in n_warmup:
        if n > len(serie):
            break

        serie_warmup, serie_obs = serie_scaled.split_after(n)
        serie_forecast_scaled = model.predict(
            len(serie_obs) - n_lags,
            series=serie_warmup,
            future_covariates=serie_scaled_idx,
            num_samples=n_samples,
        )
        serie_forecast = scaler.inverse_transform(serie_forecast_scaled)
        series_forecast.append(serie_forecast)

    return serie, series_forecast


def plot_forecasts(forecasts, n_warmup, n_col=3, sharey=True):
    n_series = len(forecasts)
    n_row = int(np.ceil(n_series / n_col))

    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(3.5 * n_col, 3 * n_row),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    for ax, forecast in zip_longest(axes.flat, forecasts.items()):
        if forecast is None:
            ax.axis("off")
        else:
            (material, sample), (serie, serie_forecasts) = forecast
            plt.sca(ax)
            serie["Wear Loss [mm]"].plot(label="observations")
            for warmup, serie_forecast in zip(n_warmup, serie_forecasts):
                serie_forecast.plot(label=f"warmup = {warmup}")
            ax.set_ylabel("Wear Loss [mm]")
            ax.set_title(f"{material} - {sample}")
            ax.legend().remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="center right")

    # from https://github.com/matplotlib/matplotlib/issues/15257#issuecomment-530988010
    bbox = legend.get_window_extent(fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.transFigure.inverted())
    fig.tight_layout(rect=(0, 0, bbox.x0, 1), h_pad=0.5, w_pad=0.5)

    return fig
