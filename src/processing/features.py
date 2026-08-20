from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast_config import DEFAULT_CONFIG, ForecastConfig
from src.ingestion.openmeteo import load_weather_runs


def impute_load_causally(load: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Fill isolated gaps from older load only and return an audit flag."""
    result = load.astype(float).copy()
    was_missing = result.isna()
    for lag in (168, 24, 1):
        missing = result.isna()
        if not missing.any():
            break
        result.loc[missing] = result.shift(lag).loc[missing]
    return result, was_missing.astype("int8")


def load_entsoe_series(project_root: Path) -> tuple[pd.Series, pd.Series]:
    paths = sorted((project_root / "data" / "raw" / "entsoe").glob("swiss_load_*.parquet"))
    if not paths:
        raise FileNotFoundError("No ENTSO-E load partitions found")
    raw = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)
    series = (
        raw.sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc", keep="last")
        .set_index("timestamp_utc")["load_mw"]
        .sort_index()
        .asfreq("h")
    )
    filled, imputed = impute_load_causally(series)
    if filled.isna().any():
        first_valid = filled.first_valid_index()
        filled = filled.loc[first_valid:]
        imputed = imputed.loc[first_valid:]
    if filled.isna().any():
        raise ValueError("Load still contains gaps after causal imputation")
    return filled, imputed


def validate_weather_vintages(
    weather: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> None:
    required = {
        "weather_run_utc",
        "weather_available_utc",
        "target_timestamp_utc",
        "forecast_lead_hour",
        "city",
        "weather_model",
        "weather_run_is_fallback",
        "source",
        "retrieved_at_utc",
        *config.weather_variables,
    }
    missing = sorted(required.difference(weather.columns))
    if missing:
        raise ValueError(f"Weather-run archive is missing columns: {missing}")
    if "forecast_origin_utc" in weather and (
        weather["weather_available_utc"] > weather["forecast_origin_utc"]
    ).any():
        raise ValueError("Weather issued after the archive collection origin was detected")
    expected_target = weather["weather_run_utc"] + pd.to_timedelta(
        weather["forecast_lead_hour"], unit="h"
    )
    if not (expected_target == weather["target_timestamp_utc"]).all():
        raise ValueError("Weather model-run lead and target timestamp disagree")


def _calendar_context(origin: pd.Timestamp, config: ForecastConfig) -> dict[str, float | int]:
    local = origin.tz_convert(config.timezone)
    dow = local.dayofweek
    doy = local.dayofyear
    hour = local.hour
    return {
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "is_weekend": int(dow >= 5),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
    }


def _split_name(origin: pd.Timestamp, config: ForecastConfig) -> str:
    if origin <= pd.Timestamp(config.train_end):
        return "train"
    if origin <= pd.Timestamp(config.validation_end):
        return "validation"
    if origin <= pd.Timestamp(config.final_test_end):
        return "final_test"
    return "future"


def build_hourly_forecast_samples(
    load: pd.Series,
    load_was_imputed: pd.Series,
    weather: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Create one point-in-time-correct 24-hour sample at every available hour."""
    if "weather_run_utc" not in weather.columns:
        return _build_hourly_samples_from_historical_forecast(load, load_was_imputed, weather, config)
    validate_weather_vintages(weather, config)
    weather = (
        weather.sort_values("retrieved_at_utc")
        .drop_duplicates(["weather_run_utc", "target_timestamp_utc", "city"], keep="last")
    )
    national = (
        weather.groupby(
            [
                "weather_run_utc",
                "weather_available_utc",
                "weather_model",
                "weather_run_is_fallback",
                "source",
                "retrieved_at_utc",
                "target_timestamp_utc",
            ],
            sort=True,
        )[list(config.weather_variables)]
        .mean()
        .reset_index()
    )

    run_frames = {
        pd.Timestamp(run).tz_convert("UTC"): frame.set_index("target_timestamp_utc").sort_index()
        for run, frame in national.groupby("weather_run_utc", sort=True)
    }
    available_runs = sorted(
        (pd.Timestamp(frame["weather_available_utc"].iloc[0]).tz_convert("UTC"), run)
        for run, frame in run_frames.items()
    )
    if not available_runs:
        raise ValueError("No weather model runs are available")

    first_origin = max(
        load.index.min() + pd.Timedelta(hours=config.lookback_hours - 1),
        available_runs[0][0],
    )
    last_origin = load.index.max() - pd.Timedelta(hours=config.horizon_hours)
    rows: list[dict[str, object]] = []
    run_position = -1
    for origin in pd.date_range(first_origin.ceil("h"), last_origin.floor("h"), freq="h", tz="UTC"):
        while (
            run_position + 1 < len(available_runs)
            and available_runs[run_position + 1][0] <= origin
        ):
            run_position += 1
        if run_position < 0:
            continue
        run_utc = available_runs[run_position][1]
        run_frame = run_frames[run_utc]

        history_index = pd.date_range(
            origin - pd.Timedelta(hours=config.lookback_hours - 1),
            origin,
            freq="h",
            tz="UTC",
        )
        target_index = pd.date_range(
            origin + pd.Timedelta(hours=1),
            periods=config.horizon_hours,
            freq="h",
            tz="UTC",
        )
        if not history_index.isin(load.index).all() or not target_index.isin(load.index).all():
            continue

        history = load.reindex(history_index)
        targets = load.reindex(target_index)
        if history.isna().any() or targets.isna().any():
            continue
        if not target_index.isin(run_frame.index).all():
            continue
        forecast_weather = run_frame.reindex(target_index)

        row: dict[str, object] = {
            "forecast_origin_local": origin.tz_convert(config.timezone).isoformat(),
            "weather_run_utc": run_utc,
            "weather_available_utc": forecast_weather["weather_available_utc"].iloc[0],
            "weather_model": forecast_weather["weather_model"].iloc[0],
            "weather_run_is_fallback": bool(forecast_weather["weather_run_is_fallback"].iloc[0]),
            "weather_source": forecast_weather["source"].iloc[0],
            "weather_retrieved_at_utc": forecast_weather["retrieved_at_utc"].iloc[0],
            "target_start_utc": target_index[0],
            "target_end_utc": target_index[-1],
            "split": _split_name(origin, config),
            "load_history_imputed_count": int(load_was_imputed.reindex(history_index).sum()),
            "load_lag_24": float(load.loc[origin - pd.Timedelta(hours=24)]),
            "load_lag_168": float(load.loc[origin - pd.Timedelta(hours=168)]),
            "load_roll_mean_24": float(history.iloc[-24:].mean()),
            "load_roll_std_24": float(history.iloc[-24:].std()),
            "load_roll_mean_168": float(history.iloc[-168:].mean()),
            "load_roll_std_168": float(history.iloc[-168:].std()),
            **_calendar_context(origin, config),
        }
        for lag, value in enumerate(history.iloc[::-1]):
            row[f"load_history_lag_{lag:03d}"] = float(value)
        for lead, (_, weather_row) in enumerate(forecast_weather.iterrows(), start=1):
            for variable in config.weather_variables:
                row[f"weather_{variable}_lead_{lead:02d}"] = float(weather_row[variable])
        for lead, value in enumerate(targets, start=1):
            row[f"target_load_lead_{lead:02d}"] = float(value)
        rows.append({"forecast_origin_utc": origin, **row})

    samples = pd.DataFrame(rows).set_index("forecast_origin_utc").sort_index()
    required_numeric = [*config.history_columns, *config.context_columns, *config.target_columns]
    if samples.empty:
        raise ValueError("No complete hourly forecast samples could be constructed")
    if samples[required_numeric].isna().any().any():
        raise ValueError("Hourly forecast samples contain missing model values")
    if samples.index.tz is None or samples.index.has_duplicates:
        raise ValueError("Forecast origins must be unique and timezone-aware")
    return samples


def _build_hourly_samples_from_historical_forecast(
    load: pd.Series,
    load_was_imputed: pd.Series,
    weather: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Build hourly samples from Open-Meteo's continuous historical forecast archive.

    This archive is operational forecast data suitable for ML training, but it does not
    preserve individual model-run timestamps. Its provenance is recorded explicitly.
    """
    required = {"timestamp_utc", "city", *config.weather_variables}
    missing = sorted(required.difference(weather.columns))
    if missing:
        raise ValueError(f"Historical forecast archive is missing columns: {missing}")
    weather = weather.copy()
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
    national = weather.groupby("timestamp_utc", sort=True)[list(config.weather_variables)].mean()
    first_origin = max(load.index.min() + pd.Timedelta(hours=config.lookback_hours - 1), national.index.min())
    last_origin = min(
        load.index.max() - pd.Timedelta(hours=config.horizon_hours),
        national.index.max() - pd.Timedelta(hours=config.horizon_hours),
    )
    rows: list[dict[str, object]] = []
    for origin in pd.date_range(first_origin.ceil("h"), last_origin.floor("h"), freq="h", tz="UTC"):
        history_index = pd.date_range(
            origin - pd.Timedelta(hours=config.lookback_hours - 1),
            origin,
            freq="h",
            tz="UTC",
        )
        target_index = pd.date_range(
            origin + pd.Timedelta(hours=1),
            periods=config.horizon_hours,
            freq="h",
            tz="UTC",
        )
        if not history_index.isin(load.index).all() or not target_index.isin(load.index).all():
            continue
        if not target_index.isin(national.index).all():
            continue
        history = load.reindex(history_index)
        targets = load.reindex(target_index)
        forecast_weather = national.reindex(target_index)
        if history.isna().any() or targets.isna().any() or forecast_weather.isna().any().any():
            continue
        row: dict[str, object] = {
            "forecast_origin_local": origin.tz_convert(config.timezone).isoformat(),
            "weather_run_utc": pd.NaT,
            "weather_available_utc": origin,
            "weather_model": "historical_forecast_timeseries",
            "weather_run_is_fallback": False,
            "weather_source": "open-meteo-historical-forecast",
            "weather_retrieved_at_utc": pd.NaT,
            "target_start_utc": target_index[0],
            "target_end_utc": target_index[-1],
            "split": _split_name(origin, config),
            "load_history_imputed_count": int(load_was_imputed.reindex(history_index).sum()),
            "load_lag_24": float(load.loc[origin - pd.Timedelta(hours=24)]),
            "load_lag_168": float(load.loc[origin - pd.Timedelta(hours=168)]),
            "load_roll_mean_24": float(history.iloc[-24:].mean()),
            "load_roll_std_24": float(history.iloc[-24:].std()),
            "load_roll_mean_168": float(history.iloc[-168:].mean()),
            "load_roll_std_168": float(history.iloc[-168:].std()),
            **_calendar_context(origin, config),
        }
        for lag, value in enumerate(history.iloc[::-1]):
            row[f"load_history_lag_{lag:03d}"] = float(value)
        for lead, (_, weather_row) in enumerate(forecast_weather.iterrows(), start=1):
            for variable in config.weather_variables:
                row[f"weather_{variable}_lead_{lead:02d}"] = float(weather_row[variable])
        for lead, value in enumerate(targets, start=1):
            row[f"target_load_lead_{lead:02d}"] = float(value)
        rows.append({"forecast_origin_utc": origin, **row})
    samples = pd.DataFrame(rows).set_index("forecast_origin_utc").sort_index()
    required_numeric = [*config.history_columns, *config.context_columns, *config.target_columns]
    if samples.empty:
        raise ValueError("No complete hourly forecast samples could be constructed")
    if samples[required_numeric].isna().any().any():
        raise ValueError("Historical forecast samples contain missing model values")
    return samples


build_daily_forecast_samples = build_hourly_forecast_samples


def run_feature_pipeline(
    output_path: str | Path | None = None,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    project_root = config.features_path.parents[2]
    load, imputed = load_entsoe_series(project_root)
    historical_paths = sorted(
        (project_root / "data" / "raw" / "weather_historical_forecast_hourly").glob(
            "year=*.parquet"
        )
    )
    if historical_paths:
        weather = pd.concat((pd.read_parquet(path) for path in historical_paths), ignore_index=True)
    else:
        weather = load_weather_runs(config)
    if weather.empty:
        raise FileNotFoundError(
            "No weather forecast runs found. Run `python -m src.ingestion.openmeteo` first."
        )
    samples = build_hourly_forecast_samples(load, imputed, weather, config)
    resolved = Path(output_path) if output_path is not None else config.features_path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(resolved)
    metadata: dict[str, int | str] = {
        "rows": len(samples),
        "columns": samples.shape[1],
        "history_hours": config.lookback_hours,
        "context_features": len(config.context_columns),
        "weather_features": len(config.weather_columns),
        "start": str(samples.index.min()),
        "end": str(samples.index.max()),
        "output_path": str(resolved),
    }
    return samples, metadata


if __name__ == "__main__":
    frame, summary = run_feature_pipeline()
    print(summary)
    print(frame[["forecast_origin_local", "weather_run_utc", "split"]].tail())
