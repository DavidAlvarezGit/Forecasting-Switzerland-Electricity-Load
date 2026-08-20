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
        "forecast_origin_utc",
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
    if (weather["weather_available_utc"] > weather["forecast_origin_utc"]).any():
        raise ValueError("Weather issued after the forecast origin was detected")
    run_counts = weather.groupby("forecast_origin_utc")["weather_run_utc"].nunique()
    if not run_counts.eq(1).all():
        raise ValueError("Every forecast origin must use exactly one weather run")
    expected_target = weather["forecast_origin_utc"] + pd.to_timedelta(
        weather["forecast_lead_hour"], unit="h"
    )
    if not (expected_target == weather["target_timestamp_utc"]).all():
        raise ValueError("Weather target timestamp and lead metadata disagree")


def _calendar_context(origin: pd.Timestamp, config: ForecastConfig) -> dict[str, float | int]:
    local = origin.tz_convert(config.timezone)
    dow = local.dayofweek
    doy = local.dayofyear
    return {
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "is_weekend": int(dow >= 5),
    }


def _split_name(origin: pd.Timestamp, config: ForecastConfig) -> str:
    if origin <= pd.Timestamp(config.train_end):
        return "train"
    if origin <= pd.Timestamp(config.validation_end):
        return "validation"
    if origin <= pd.Timestamp(config.final_test_end):
        return "final_test"
    return "future"


def build_daily_forecast_samples(
    load: pd.Series,
    load_was_imputed: pd.Series,
    weather: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Create one point-in-time-correct 24-hour sample per local day."""
    validate_weather_vintages(weather, config)
    national = (
        weather.groupby(
            [
                "forecast_origin_utc",
                "weather_run_utc",
                "weather_available_utc",
                "weather_model",
                "weather_run_is_fallback",
                "source",
                "retrieved_at_utc",
                "forecast_lead_hour",
                "target_timestamp_utc",
            ],
            sort=True,
        )[list(config.weather_variables)]
        .mean()
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for origin, run_frame in national.groupby("forecast_origin_utc", sort=True):
        origin = pd.Timestamp(origin).tz_convert("UTC")
        run_frame = run_frame.sort_values("forecast_lead_hour")
        expected_leads = np.arange(1, config.horizon_hours + 1)
        if not np.array_equal(run_frame["forecast_lead_hour"].to_numpy(), expected_leads):
            continue

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

        row: dict[str, object] = {
            "forecast_origin_local": origin.tz_convert(config.timezone).isoformat(),
            "weather_run_utc": run_frame["weather_run_utc"].iloc[0],
            "weather_available_utc": run_frame["weather_available_utc"].iloc[0],
            "weather_model": run_frame["weather_model"].iloc[0],
            "weather_run_is_fallback": bool(run_frame["weather_run_is_fallback"].iloc[0]),
            "weather_source": run_frame["source"].iloc[0],
            "weather_retrieved_at_utc": run_frame["retrieved_at_utc"].iloc[0],
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
        for _, weather_row in run_frame.iterrows():
            lead = int(weather_row["forecast_lead_hour"])
            for variable in config.weather_variables:
                row[f"weather_{variable}_lead_{lead:02d}"] = float(weather_row[variable])
        for lead, value in enumerate(targets, start=1):
            row[f"target_load_lead_{lead:02d}"] = float(value)
        rows.append({"forecast_origin_utc": origin, **row})

    samples = pd.DataFrame(rows).set_index("forecast_origin_utc").sort_index()
    required_numeric = [*config.history_columns, *config.context_columns, *config.target_columns]
    if samples.empty:
        raise ValueError("No complete daily forecast samples could be constructed")
    if samples[required_numeric].isna().any().any():
        raise ValueError("Daily forecast samples contain missing model values")
    if samples.index.tz is None or samples.index.has_duplicates:
        raise ValueError("Forecast origins must be unique and timezone-aware")
    local = samples.index.tz_convert(config.timezone)
    if not (local.hour == config.forecast_origin_hour_local).all():
        raise ValueError("A sample does not use the configured local forecast hour")
    return samples


def run_feature_pipeline(
    output_path: str | Path | None = None,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    project_root = config.features_path.parents[2]
    load, imputed = load_entsoe_series(project_root)
    weather = load_weather_runs(config)
    if weather.empty:
        raise FileNotFoundError(
            "No weather forecast runs found. Run `python -m src.ingestion.openmeteo` first."
        )
    samples = build_daily_forecast_samples(load, imputed, weather, config)
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
