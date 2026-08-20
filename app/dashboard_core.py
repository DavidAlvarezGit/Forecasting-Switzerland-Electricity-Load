from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DashboardResults:
    forecast: pd.DataFrame
    recent_actual: pd.Series
    backtest_history: pd.DataFrame
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    notices: list[str] = field(default_factory=list)


def _datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index, utc=True)
    return result.sort_index()


def _latest_observed_history(samples: pd.DataFrame, artifact: Any) -> pd.Series:
    """Reconstruct observations known at the latest forecast origin."""
    origin = samples.index[-1]
    latest = samples.iloc[-1]
    values: dict[pd.Timestamp, float] = {}
    for column in artifact.history_columns:
        if column not in latest or pd.isna(latest[column]):
            continue
        lag = int(column.rsplit("_", maxsplit=1)[-1])
        values[origin - pd.Timedelta(hours=lag)] = float(latest[column])
    return pd.Series(values, name="true_load_mw", dtype=float).sort_index()


def _run_backtest(
    function: Callable[..., dict[str, Any]] | None,
    samples: pd.DataFrame,
    artifact: Any,
    eval_origins: int,
    history_origins: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if function is None:
        return None, "The hourly backtest is unavailable in the loaded inference module."
    try:
        return (
            function(
                samples,
                artifact,
                eval_windows=eval_origins,
                history_windows=history_origins,
            ),
            None,
        )
    except Exception as exc:
        return None, f"The hourly backtest could not be calculated: {exc}"


def _run_forecast(
    point_function: Callable[..., pd.Series],
    interval_function: Callable[..., pd.DataFrame] | None,
    samples: pd.DataFrame,
    artifact: Any,
    include_intervals: bool,
) -> tuple[pd.DataFrame, str | None]:
    if include_intervals and interval_function is not None:
        try:
            return _datetime_index(interval_function(samples, artifact)), None
        except Exception as exc:
            point = point_function(samples, artifact)
            return point.to_frame(), f"Forecast ranges failed; point values are shown: {exc}"
    point = point_function(samples, artifact)
    notice = None if not include_intervals else "Forecast ranges are unavailable."
    return point.to_frame(), notice


def build_dashboard_results(
    samples: pd.DataFrame,
    artifact: Any,
    *,
    forecast_next_horizon: Callable[..., pd.Series],
    forecast_next_horizon_with_intervals: Callable[..., pd.DataFrame] | None,
    evaluate_recent_backtest: Callable[..., dict[str, Any]] | None,
    eval_origins: int = 120,
    history_origins: int = 30,
    include_intervals: bool = True,
) -> DashboardResults:
    samples = _datetime_index(samples)
    if samples.empty:
        raise ValueError("The hourly forecast sample table is empty")
    missing = [column for column in artifact.feature_cols if column not in samples.columns]
    if missing:
        raise ValueError(f"The sample table is missing model inputs: {', '.join(missing[:5])}")

    notices: list[str] = []
    backtest, backtest_notice = _run_backtest(
        evaluate_recent_backtest,
        samples,
        artifact,
        eval_origins,
        history_origins,
    )
    if backtest_notice:
        notices.append(backtest_notice)
    forecast, forecast_notice = _run_forecast(
        forecast_next_horizon,
        forecast_next_horizon_with_intervals,
        samples,
        artifact,
        include_intervals,
    )
    if forecast_notice:
        notices.append(forecast_notice)

    history = pd.DataFrame()
    recent_actual = _latest_observed_history(samples, artifact)
    metrics: dict[str, Any] = {}
    if backtest:
        history = backtest.get("forecast_history_df", pd.DataFrame())
        if isinstance(history, pd.DataFrame) and not history.empty:
            history = _datetime_index(history)
        metrics = {
            "mae_model": backtest["mae_model"],
            "rmse_model": backtest["rmse_model"],
            "model_metrics": backtest["model_metrics"],
            "baseline_metrics": backtest["baseline_metrics"],
            "coverage": backtest["coverage"],
            "target_coverage": backtest["target_coverage"],
            "mean_interval_width": backtest["mean_interval_width"],
            "coverage_by_lead": backtest["coverage_by_lead"],
            "width_by_lead": backtest["width_by_lead"],
            "n_eval_windows": backtest["n_eval_windows"],
            "n_calibration_windows": backtest["n_calibration_windows"],
            "aci_config": backtest["aci_config"],
        }

    latest = samples.iloc[-1]
    split_counts = samples["split"].value_counts().to_dict() if "split" in samples else {}
    metadata = {
        "dataset_rows": len(samples),
        "dataset_columns": samples.shape[1],
        "dataset_start": samples.index.min(),
        "dataset_end": samples.index.max(),
        "missing_share": float(samples[artifact.feature_cols].isna().mean().mean()),
        "feature_count": len(artifact.context_columns),
        "forecast_weather_feature_count": sum(
            column.startswith("weather_") for column in artifact.context_columns
        ),
        "model_version": artifact.model_version,
        "lookback": artifact.lookback,
        "horizon": artifact.horizon,
        "device": artifact.device,
        "nominal_coverage": artifact.forecast_config["nominal_coverage"],
        "forecast_frequency": artifact.forecast_config.get("forecast_frequency", "daily"),
        "timezone": artifact.forecast_config["timezone"],
        "latest_origin": samples.index[-1],
        "latest_weather_run": latest.get("weather_run_utc"),
        "latest_weather_available": latest.get("weather_available_utc"),
        "latest_weather_model": latest.get("weather_model"),
        "latest_weather_retrieved": latest.get("weather_retrieved_at_utc"),
        "split_counts": split_counts,
        "interval_method": "Adaptive conformal inference (ACI), selected on validation",
    }
    return DashboardResults(
        forecast=forecast,
        recent_actual=recent_actual,
        backtest_history=history,
        metrics=metrics,
        metadata=metadata,
        notices=notices,
    )