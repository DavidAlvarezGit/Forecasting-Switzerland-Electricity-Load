from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass(slots=True)
class DashboardResults:
    """Serializable output consumed by the Streamlit presentation layer."""

    forecast: pd.DataFrame
    recent_actual: pd.Series
    backtest_history: pd.DataFrame
    metrics: dict[str, float | int]
    metadata: dict[str, Any]
    notices: list[str] = field(default_factory=list)


def _normalise_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index, utc=True)
    return result.sort_index()


def _collapse_backtest_history(backtest: dict[str, Any] | None) -> pd.DataFrame:
    """Return one most-recent prediction per timestamp for readable charts."""

    if not backtest:
        return pd.DataFrame()

    history = backtest.get("forecast_history_df")
    if not isinstance(history, pd.DataFrame) or history.empty:
        history = backtest.get("latest_window_df")
    if not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame()

    history = _normalise_datetime_index(history)
    if history.index.has_duplicates:
        history = history.groupby(level=0).last()
    return history.sort_index()


def _run_backtest(
    evaluate_recent_backtest: Callable[..., dict[str, Any]] | None,
    df: pd.DataFrame,
    artifact: Any,
    *,
    alpha: float,
    calibration_windows: int,
    eval_windows: int,
    history_windows: int,
    per_horizon: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    if evaluate_recent_backtest is None:
        return None, "Backtest diagnostics are unavailable in the loaded inference module."

    kwargs = {
        "alpha": alpha,
        "calibration_windows": calibration_windows,
        "eval_windows": eval_windows,
        "history_windows": history_windows,
        "per_horizon": per_horizon,
    }
    try:
        try:
            return evaluate_recent_backtest(df, artifact, **kwargs), None
        except TypeError as exc:
            # Compatibility with checkpoints served alongside an older module.
            if "history_windows" not in str(exc):
                raise
            kwargs.pop("history_windows")
            return evaluate_recent_backtest(df, artifact, **kwargs), None
    except Exception as exc:  # The point forecast can still be useful.
        return None, f"Backtest diagnostics could not be calculated: {exc}"


def _run_forecast(
    forecast_next_horizon: Callable[..., pd.Series],
    forecast_next_horizon_with_intervals: Callable[..., pd.DataFrame] | None,
    df: pd.DataFrame,
    artifact: Any,
    *,
    alpha: float,
    calibration_windows: int,
    per_horizon: bool,
    include_intervals: bool,
) -> tuple[pd.DataFrame, str | None]:
    if include_intervals and forecast_next_horizon_with_intervals is not None:
        try:
            forecast = forecast_next_horizon_with_intervals(
                df,
                artifact,
                alpha=alpha,
                calibration_windows=calibration_windows,
                per_horizon=per_horizon,
            )
            return _normalise_datetime_index(forecast), None
        except Exception as exc:
            notice = f"Prediction intervals failed; the point forecast is shown instead: {exc}"
            point = forecast_next_horizon(df, artifact)
            return point.to_frame(name="forecast_load_mw"), notice

    notice = None
    if include_intervals:
        notice = "Prediction intervals are unavailable in the loaded inference module."
    point = forecast_next_horizon(df, artifact)
    return point.to_frame(name="forecast_load_mw"), notice


def build_dashboard_results(
    df: pd.DataFrame,
    artifact: Any,
    *,
    forecast_next_horizon: Callable[..., pd.Series],
    forecast_next_horizon_with_intervals: Callable[..., pd.DataFrame] | None,
    evaluate_recent_backtest: Callable[..., dict[str, Any]] | None,
    confidence: float = 0.90,
    calibration_windows: int = 120,
    eval_windows: int = 120,
    history_windows: int = 336,
    per_horizon: bool = True,
    include_intervals: bool = True,
) -> DashboardResults:
    """Run validation, forecasting, and diagnostics for the dashboard."""

    df = _normalise_datetime_index(df)
    if len(df) < artifact.lookback:
        raise ValueError(
            f"At least {artifact.lookback:,} feature rows are required; found {len(df):,}."
        )

    missing_features = [column for column in artifact.feature_cols if column not in df.columns]
    if missing_features:
        preview = ", ".join(missing_features[:5])
        raise ValueError(f"The feature table is missing model inputs: {preview}.")

    alpha = 1.0 - float(confidence)
    notices: list[str] = []

    backtest, backtest_notice = _run_backtest(
        evaluate_recent_backtest,
        df,
        artifact,
        alpha=alpha,
        calibration_windows=calibration_windows,
        eval_windows=eval_windows,
        history_windows=history_windows,
        per_horizon=per_horizon,
    )
    if backtest_notice:
        notices.append(backtest_notice)

    forecast, forecast_notice = _run_forecast(
        forecast_next_horizon,
        forecast_next_horizon_with_intervals,
        df,
        artifact,
        alpha=alpha,
        calibration_windows=calibration_windows,
        per_horizon=per_horizon,
        include_intervals=include_intervals,
    )
    if forecast_notice:
        notices.append(forecast_notice)

    target = pd.Series(dtype=float, name=artifact.target_col)
    if artifact.target_col in df.columns:
        target = df[artifact.target_col].dropna().astype(float)

    metrics: dict[str, float | int] = {}
    if backtest:
        for key in (
            "mae_model",
            "mae_baseline",
            "mae_improvement_abs",
            "mae_improvement_pct",
            "rmse_model",
            "rmse_baseline",
            "coverage",
            "target_coverage",
            "coverage_gap",
            "mean_interval_width",
            "n_eval_windows",
            "n_calibration_windows",
            "aci_eta",
            "aci_alpha_t_mean",
        ):
            value = backtest.get(key)
            if value is not None:
                metrics[key] = value

    missing_share = float(df[artifact.feature_cols].isna().mean().mean())
    metadata = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
        "dataset_start": df.index.min(),
        "dataset_end": df.index.max(),
        "missing_share": missing_share,
        "feature_count": int(len(artifact.feature_cols)),
        "forecast_weather_feature_count": sum(
            column.startswith("forecast_ch_mean_") for column in artifact.feature_cols
        ),
        "model_version": int(getattr(artifact, "model_version", 1)),
        "target_column": artifact.target_col,
        "lookback": int(artifact.lookback),
        "horizon": int(artifact.horizon),
        "device": str(artifact.device),
        "confidence": float(confidence),
        "interval_method": (
            "Adaptive ranges · separate for each forecast hour"
            if per_horizon
            else "Adaptive ranges · shared across all forecast hours"
        ),
        "baseline_name": backtest.get("baseline_name", "unavailable") if backtest else "unavailable",
    }

    return DashboardResults(
        forecast=forecast,
        recent_actual=target.tail(history_windows),
        backtest_history=_collapse_backtest_history(backtest),
        metrics=metrics,
        metadata=metadata,
        notices=notices,
    )
