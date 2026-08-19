from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if __package__ in (None, ""):
    from helpers import find_project_root
else:
    from .helpers import find_project_root


TARGET_COL = "load_load_mw"
AGGREGATED_DATASET_REL_PATH = Path("data") / "interim" / "aggregated.parquet"
LSTM_FEATURE_DATASET_REL_PATH = Path("data") / "processed" / "lstm_features.parquet"
FORECAST_WEATHER_DATASETS = (
    "weather_previous_runs",
    "weather_live_forecast",
)
FORECAST_WEATHER_FEATURES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "surface_pressure",
)


def _as_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    return df.index


def _load_aggregated(project_root: Path, input_path: str | Path | None = None) -> pd.DataFrame:
    resolved_input = Path(input_path) if input_path is not None else project_root / AGGREGATED_DATASET_REL_PATH
    if not resolved_input.exists():
        raise FileNotFoundError(f"Aggregated data not found: {resolved_input}")

    df = pd.read_parquet(resolved_input)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Aggregated data must have a DatetimeIndex")

    return df.sort_index()


def _load_forecast_weather(project_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_rank, dataset in enumerate(FORECAST_WEATHER_DATASETS):
        paths = sorted((project_root / "data" / "raw" / dataset).glob("year=*.parquet"))
        if dataset == "weather_previous_runs" and not paths:
            raise FileNotFoundError(
                "No fixed-vintage weather_previous_runs data found. "
                "Run src/ingestion/openmeteo.py before rebuilding features."
            )
        for path in paths:
            frame = pd.read_parquet(path)
            if dataset == "weather_previous_runs":
                rename = {
                    f"{feature}_previous_day1": feature
                    for feature in FORECAST_WEATHER_FEATURES
                    if f"{feature}_previous_day1" in frame.columns
                }
                frame = frame.rename(columns=rename)
            frame["_source_rank"] = source_rank
            frames.append(frame)

    if not frames:
        expected = ", ".join(FORECAST_WEATHER_DATASETS)
        raise FileNotFoundError(f"No forecast-weather files found in: {expected}")

    forecast = pd.concat(frames, ignore_index=True)
    required = {"timestamp_utc", "city", *FORECAST_WEATHER_FEATURES}
    missing = sorted(required.difference(forecast.columns))
    if missing:
        raise ValueError(f"Forecast-weather data is missing columns: {missing}")

    forecast["timestamp_utc"] = pd.to_datetime(forecast["timestamp_utc"], utc=True)
    forecast["city"] = forecast["city"].astype(str).str.lower().str.replace(" ", "_", regex=False)
    if "retrieved_at_utc" in forecast.columns:
        forecast["retrieved_at_utc"] = pd.to_datetime(forecast["retrieved_at_utc"], utc=True)
        sort_columns = ["timestamp_utc", "city", "_source_rank", "retrieved_at_utc"]
    else:
        sort_columns = ["timestamp_utc", "city", "_source_rank"]

    # Live forecasts supersede archived forecasts for the same target timestamp.
    return (
        forecast.sort_values(sort_columns)
        .drop_duplicates(["timestamp_utc", "city"], keep="last")
        .sort_values("timestamp_utc")
    )


def build_forecast_lead_features(
    forecast_weather: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    """Encode target-time forecast weather on each forecast-issue row.

    A row at time ``t`` receives weather forecasts for ``t + 1`` through
    ``t + horizon``. These are forecast products, not shifted observations, so
    future covariates can be used without looking ahead at realised weather.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    national = (
        forecast_weather.groupby("timestamp_utc", sort=True)[list(FORECAST_WEATHER_FEATURES)]
        .mean()
        .sort_index()
        .asfreq("h")
    )
    lead_columns: dict[str, pd.Series] = {}
    for feature in FORECAST_WEATHER_FEATURES:
        for lead in range(1, horizon + 1):
            name = f"forecast_ch_mean_{feature}_lead_{lead:02d}"
            lead_columns[name] = national[feature].shift(-lead).reindex(target_index)

    return pd.DataFrame(lead_columns, index=target_index)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = _as_datetime_index(out)

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["dayofyear"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return out


def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    return out


def _add_load_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def _add_load_rolling_features(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    shifted = out[target_col].shift(1)

    for window in windows:
        roll = shifted.rolling(window=window, min_periods=window)
        out[f"{target_col}_roll_mean_{window}"] = roll.mean()
        out[f"{target_col}_roll_std_{window}"] = roll.std()
    return out


def build_lstm_feature_table(
    aggregated_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    keep_raw_calendar: bool = False,
    forecast_weather: pd.DataFrame | None = None,
    forecast_horizon: int = 24,
) -> pd.DataFrame:
    if target_col not in aggregated_df.columns:
        raise ValueError(f"Target column not found in aggregated data: {target_col}")

    lag_values = lags or [1, 2, 3, 6, 12, 24, 48, 72, 168]
    window_values = rolling_windows or [24, 168]

    features = aggregated_df.copy()
    features = _add_calendar_features(features)
    features = _add_cyclical_features(features)
    features = _add_load_lag_features(features, target_col=target_col, lags=lag_values)
    features = _add_load_rolling_features(features, target_col=target_col, windows=window_values)

    if forecast_weather is not None:
        forecast_features = build_forecast_lead_features(
            forecast_weather,
            target_index=_as_datetime_index(features),
            horizon=forecast_horizon,
        )
        features = features.join(forecast_features, how="left")

    if not keep_raw_calendar:
        # Cyclical calendar features are enough for sequence models and reduce redundancy.
        features = features.drop(columns=["hour", "dayofweek", "month", "dayofyear"], errors="ignore")

    features = features.dropna().sort_index()
    return features


def run_feature_pipeline(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    project_root = find_project_root(Path.cwd().resolve())
    aggregated_df = _load_aggregated(project_root, input_path)
    forecast_weather = _load_forecast_weather(project_root)
    lstm_df = build_lstm_feature_table(
        aggregated_df,
        target_col=target_col,
        forecast_weather=forecast_weather,
    )

    resolved_output = (
        Path(output_path)
        if output_path is not None
        else project_root / LSTM_FEATURE_DATASET_REL_PATH
    )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    lstm_df.to_parquet(resolved_output)

    meta: dict[str, int | str] = {
        "model": "lstm",
        "rows": int(len(lstm_df)),
        "columns": int(lstm_df.shape[1]),
        "forecast_weather_columns": sum(
            column.startswith("forecast_ch_mean_") for column in lstm_df.columns
        ),
        "start": str(lstm_df.index.min()),
        "end": str(lstm_df.index.max()),
        "output_path": str(resolved_output),
    }
    return lstm_df, meta


def main() -> None:
    feature_df, meta = run_feature_pipeline()
    print(
        f"LSTM feature dataset: {meta['start']} -> {meta['end']} | "
        f"rows={meta['rows']} cols={meta['columns']}"
    )
    print(f"Feature output: {meta['output_path']}")
    print(feature_df.head(5))


if __name__ == "__main__":
    main()
