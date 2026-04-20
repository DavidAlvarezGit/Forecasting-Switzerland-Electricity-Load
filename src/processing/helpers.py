from pathlib import Path
from typing import Mapping

import pandas as pd

from validation.core import collect_checks
from validation import data_checks as dc

WEATHER_FEATURES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation_probability",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "wind_speed_10m",
    "surface_pressure",
    "is_day",
    "sunshine_duration",
]

ZERO_FILL_FEATURES = [
    "precipitation_probability",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "sunshine_duration",
]


def _as_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    return df.index


def aggregate_to_interim(
    dfs: Mapping[str, pd.DataFrame],
    output_path: str = "../../data/interim/aggregated.parquet",
    freq: str = "h",
) -> pd.DataFrame:
    """Aggregate multiple time series into one aligned dataframe."""
    if not dfs:
        raise ValueError("No input datasets provided")

    for _, df in dfs.items():
        collect_checks([
            lambda df=df: dc.check_sorted_index(df),
            lambda df=df: dc.check_no_duplicates(df),
            lambda df=df: dc.check_timezone(df),
        ])

    common_index: pd.DatetimeIndex | None = None
    for df in dfs.values():
        index = _as_datetime_index(df)
        if common_index is None:
            common_index = index
        else:
            common_index = common_index.intersection(index)

    if common_index is None or len(common_index) == 0:
        raise ValueError("No overlapping timestamps between datasets")

    aligned: list[pd.DataFrame] = []
    for name, df in dfs.items():
        df_aligned = df.loc[common_index].copy()
        df_aligned.columns = [f"{name}_{col}" for col in df_aligned.columns]
        aligned.append(df_aligned)

    df_final = pd.concat(aligned, axis=1)
    df_final = df_final.asfreq(freq)

    collect_checks([
        lambda: dc.check_sorted_index(df_final),
        lambda: dc.check_no_duplicates(df_final),
        lambda: dc.check_frequency(df_final, freq),
        lambda: dc.check_no_missing_timestamps(df_final, freq),
        lambda: dc.check_no_nan(df_final),
    ])

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_file)

    print(f"Saved aggregated data to {output_file}")
    return df_final


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "raw").exists():
            return candidate
    raise RuntimeError("Could not locate project root containing data/raw")


def load_raw_inputs(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather_files = sorted((project_root / "data" / "raw" / "weather_historical").glob("year=*.parquet"))
    load_files = sorted((project_root / "data" / "raw" / "entsoe").glob("swiss_load_*.parquet"))

    if not weather_files:
        raise FileNotFoundError("No weather_historical parquet files found")
    if not load_files:
        raise FileNotFoundError("No entsoe parquet files found")

    weather_raw = pd.concat((pd.read_parquet(path) for path in weather_files), ignore_index=True)
    load_raw = pd.concat((pd.read_parquet(path) for path in load_files), ignore_index=True)
    return weather_raw, load_raw


def normalize_raw_inputs(
    weather_raw: pd.DataFrame,
    load_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather = weather_raw.copy()
    load = load_raw.copy()

    missing_weather_features = [feature for feature in WEATHER_FEATURES if feature not in weather.columns]
    if missing_weather_features:
        raise ValueError(f"Missing fetched weather features: {missing_weather_features}")

    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
    load["timestamp_utc"] = pd.to_datetime(load["timestamp_utc"], utc=True)
    weather["city"] = weather["city"].astype(str).str.lower().str.replace(" ", "_", regex=False)
    return weather, load


def build_weather_wide(weather_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather_dedup = (
        weather_raw.sort_values(["timestamp_utc", "city", "retrieved_at_utc"])
        .drop_duplicates(subset=["timestamp_utc", "city"], keep="last")
    )

    weather_wide = (
        weather_dedup.pivot_table(
            index="timestamp_utc",
            columns="city",
            values=WEATHER_FEATURES,
            aggfunc="first",
        )
        .sort_index()
    )
    weather_wide.columns = [f"{city}__{feature}" for feature, city in weather_wide.columns]
    return weather_wide, weather_dedup


def build_load_series(load_raw: pd.DataFrame) -> pd.DataFrame:
    return (
        load_raw[["timestamp_utc", "load_mw"]]
        .drop_duplicates("timestamp_utc")
        .set_index("timestamp_utc")
        .sort_index()
    )


def build_hourly_index(df_load: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DatetimeIndex:
    start = max(df_load.index.min(), df_weather.index.min())
    end = min(df_load.index.max(), df_weather.index.max())
    return pd.date_range(start=start, end=end, freq="h", tz="UTC")


def feature_columns(df_weather: pd.DataFrame, feature: str) -> list[str]:
    suffix = f"__{feature}"
    return [col for col in df_weather.columns if col.endswith(suffix)]


def impute_weather(df_weather: pd.DataFrame) -> pd.DataFrame:
    weather = df_weather.copy()
    continuous_features = [
        feature
        for feature in WEATHER_FEATURES
        if feature not in ZERO_FILL_FEATURES and feature != "is_day"
    ]

    for feature in continuous_features:
        cols = feature_columns(weather, feature)
        if cols:
            weather[cols] = weather[cols].interpolate(method="time").ffill().bfill()

    for feature in ZERO_FILL_FEATURES:
        cols = feature_columns(weather, feature)
        if cols:
            weather[cols] = weather[cols].fillna(0.0)

    is_day_cols = feature_columns(weather, "is_day")
    if is_day_cols:
        weather[is_day_cols] = weather[is_day_cols].ffill().bfill().fillna(0.0).round().clip(0, 1)

    return weather


def assert_no_missing_values(df_load: pd.DataFrame, df_weather: pd.DataFrame) -> None:
    load_nan_cols = df_load.columns[df_load.isna().any()].tolist()
    weather_nan_cols = df_weather.columns[df_weather.isna().any()].tolist()
    if load_nan_cols or weather_nan_cols:
        raise ValueError(
            f"Residual NaNs after imputation. load: {load_nan_cols}, weather: {weather_nan_cols}"
        )


def run_processing_pipeline(
    output_path: str | Path | None = None,
    freq: str = "h",
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Run the full raw-to-interim processing pipeline and return data plus summary metadata."""
    project_root = find_project_root(Path.cwd().resolve())
    weather_raw, load_raw = load_raw_inputs(project_root)
    weather_raw, load_raw = normalize_raw_inputs(weather_raw, load_raw)

    df_weather, weather_dedup = build_weather_wide(weather_raw)
    df_load = build_load_series(load_raw)

    hourly_index = build_hourly_index(df_load, df_weather)
    df_load = df_load.reindex(hourly_index).interpolate(method="time").ffill().bfill()
    df_weather = impute_weather(df_weather.reindex(hourly_index))

    assert_no_missing_values(df_load, df_weather)

    resolved_output = Path(output_path) if output_path is not None else project_root / "data" / "interim" / "aggregated.parquet"
    df = aggregate_to_interim(
        dfs={"load": df_load, "weather": df_weather},
        output_path=str(resolved_output),
        freq=freq,
    )

    metadata: dict[str, int | str] = {
        "weather_features": len(WEATHER_FEATURES),
        "city_count": int(weather_dedup["city"].nunique()),
        "weather_columns": int(df_weather.shape[1]),
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "output_path": str(resolved_output),
    }
    return df, metadata
