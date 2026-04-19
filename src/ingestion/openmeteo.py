import time
from pathlib import Path
from typing import Any

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

if __package__ in (None, ""):
    from state_utils import (
        get_state_timestamp,
        load_json_state,
        save_json_state,
        set_state_timestamp,
    )
else:
    from .state_utils import (
        get_state_timestamp,
        load_json_state,
        save_json_state,
        set_state_timestamp,
    )


OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CITIES = [
    "Zurich", "Geneva", "Bern", "Basel", "Lausanne",
    "Lucerne", "St_Gallen", "Lugano", "Interlaken", "Central_CH",
]

LAT = [47.3769, 46.2044, 46.9480, 47.5596, 46.5197,
       47.0502, 47.4245, 46.0101, 46.6863, 46.8182]

LON = [8.5417, 6.1432, 7.4474, 7.5886, 6.6323,
       8.3093, 9.3767, 8.9600, 7.8632, 8.2275]

HOURLY_VARS = [
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

DAILY_VARS = [
    "temperature_2m_mean",
    "daylight_duration",
    "temperature_2m_min",
    "temperature_2m_max",
    "wind_gusts_10m_mean",
    "wind_speed_10m_mean",
]

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_ROOT / "raw"
STATE_FILE = DATA_ROOT / "state" / "openmeteo_state.json"

DATASET_HISTORICAL = "weather_historical"
DATASET_HISTORICAL_FORECAST_HOURLY = "weather_historical_forecast_hourly"
DATASET_HISTORICAL_FORECAST_DAILY = "weather_historical_forecast_daily"
DATASET_LIVE_FORECAST = "weather_live_forecast"


def load_state() -> dict[str, str]:
    return load_json_state(STATE_FILE)


def save_state(state: dict[str, str]):
    save_json_state(STATE_FILE, state)


def get_last_timestamp(state: dict[str, str], key: str) -> pd.Timestamp | None:
    return get_state_timestamp(state, key)


def update_state(state: dict[str, str], key: str, timestamp: pd.Timestamp):
    set_state_timestamp(state, key, timestamp)
    save_state(state)


def build_client(expire_after: int = 3600):
    cache_session = requests_cache.CachedSession(".cache", expire_after=expire_after)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_with_retries(
    client,
    url: str,
    params: dict[str, Any],
    retries: int = 5,
    timeout_backoff_base: int = 2,
):
    for attempt in range(retries):
        try:
            return client.weather_api(url, params=params)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            sleep_time = timeout_backoff_base ** attempt
            print(f"[Retry {attempt + 1}] error: {exc} -> sleeping {sleep_time}s")
            time.sleep(sleep_time)


def _build_common_params(hourly_vars: list[str], timezone: str = "Europe/Zurich") -> dict[str, Any]:
    return {
        "latitude": LAT,
        "longitude": LON,
        "hourly": hourly_vars,
        "timezone": timezone,
    }


def _build_date_range(
    start_date: str,
    end_date: str,
    last_ts: pd.Timestamp | None,
) -> tuple[str, str] | None:
    effective_start = start_date
    if last_ts is not None:
        inferred_start = (last_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        effective_start = max(start_date, inferred_start)

    if pd.Timestamp(effective_start) > pd.Timestamp(end_date):
        return None

    return effective_start, end_date


def parse_hourly_responses(responses, hourly_vars: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for i, response in enumerate(responses):
        hourly = response.Hourly()
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                ),
                "city": CITIES[i],
                "location_id": i,
                "latitude": response.Latitude(),
                "longitude": response.Longitude(),
                "elevation": response.Elevation(),
                "retrieved_at_utc": pd.Timestamp.utcnow(),
            }
        )

        for j, var in enumerate(hourly_vars):
            df[var] = hourly.Variables(j).ValuesAsNumpy()

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def parse_historical_forecast_responses(
    responses,
    hourly_vars: list[str],
    daily_vars: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []

    for i, response in enumerate(responses):
        hourly = response.Hourly()
        daily = response.Daily()

        hourly_df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                ),
                "city": CITIES[i],
                "location_id": i,
                "latitude": response.Latitude(),
                "longitude": response.Longitude(),
                "elevation": response.Elevation(),
                "retrieved_at_utc": pd.Timestamp.utcnow(),
            }
        )

        for j, col in enumerate(hourly_vars):
            hourly_df[col] = hourly.Variables(j).ValuesAsNumpy()

        daily_df = pd.DataFrame(
            {
                "date_utc": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left",
                ),
                "city": CITIES[i],
                "location_id": i,
                "latitude": response.Latitude(),
                "longitude": response.Longitude(),
                "elevation": response.Elevation(),
                "retrieved_at_utc": pd.Timestamp.utcnow(),
            }
        )

        for j, col in enumerate(daily_vars):
            daily_df[col] = daily.Variables(j).ValuesAsNumpy()

        hourly_rows.append(hourly_df)
        daily_rows.append(daily_df)

    if not hourly_rows:
        return pd.DataFrame(), pd.DataFrame()

    return (
        pd.concat(hourly_rows, ignore_index=True),
        pd.concat(daily_rows, ignore_index=True),
    )


def _dataset_dir(dataset: str) -> Path:
    return PROCESSED_DIR / dataset


def _year_path(dataset: str, year: int) -> Path:
    return _dataset_dir(dataset) / f"year={year}.parquet"


def save_partitioned(
    df: pd.DataFrame,
    dataset: str,
    time_col: str,
    dedupe_cols: list[str],
):
    if df.empty:
        return

    data = df.copy()
    data["year"] = data[time_col].dt.year

    for year, df_year in data.groupby("year"):
        path = _year_path(dataset, int(year))

        if path.exists():
            existing = pd.read_parquet(path)
            combined = (
                pd.concat([existing, df_year], ignore_index=True)
                .drop_duplicates(subset=dedupe_cols, keep="last")
                .sort_values(time_col)
                .reset_index(drop=True)
            )
        else:
            combined = df_year.sort_values(time_col).reset_index(drop=True)

        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False)
        print(f"Saved dataset={dataset} year={year} -> {path} | shape={combined.shape}")


def load_dataset(dataset: str, time_col: str) -> pd.DataFrame:
    files = sorted(_dataset_dir(dataset).glob("year=*.parquet"))
    if not files:
        return pd.DataFrame()

    return (
        pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        .sort_values(time_col)
        .reset_index(drop=True)
    )


def ingest_weather_historical(
    start_date: str,
    end_date: str,
    hourly_vars: list[str] | None = None,
) -> pd.DataFrame:
    vars_to_use = hourly_vars or HOURLY_VARS
    state = load_state()
    last_ts = get_last_timestamp(state, DATASET_HISTORICAL)
    date_range = _build_date_range(start_date, end_date, last_ts)
    if date_range is None:
        print("No new historical weather window to fetch")
        return load_dataset(DATASET_HISTORICAL, "timestamp_utc")

    effective_start, effective_end = date_range

    params = _build_common_params(vars_to_use)
    params["start_date"] = effective_start
    params["end_date"] = effective_end

    client = build_client()
    responses = fetch_with_retries(client, OPENMETEO_ARCHIVE_URL, params)
    df = parse_hourly_responses(responses, vars_to_use)

    if df.empty:
        print("No new historical weather data")
        return load_dataset(DATASET_HISTORICAL, "timestamp_utc")

    save_partitioned(
        df=df,
        dataset=DATASET_HISTORICAL,
        time_col="timestamp_utc",
        dedupe_cols=["timestamp_utc", "location_id"],
    )
    update_state(state, DATASET_HISTORICAL, df["timestamp_utc"].max())
    return load_dataset(DATASET_HISTORICAL, "timestamp_utc")


def ingest_weather_historical_forecast(
    start_date: str,
    end_date: str,
    model: str = "best_match",
    hourly_vars: list[str] | None = None,
    daily_vars: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_to_use = hourly_vars or HOURLY_VARS
    daily_to_use = daily_vars or DAILY_VARS

    state = load_state()
    last_ts = get_last_timestamp(state, DATASET_HISTORICAL_FORECAST_HOURLY)
    date_range = _build_date_range(start_date, end_date, last_ts)
    if date_range is None:
        print("No new historical forecast weather window to fetch")
        return (
            load_dataset(DATASET_HISTORICAL_FORECAST_HOURLY, "timestamp_utc"),
            load_dataset(DATASET_HISTORICAL_FORECAST_DAILY, "date_utc"),
        )

    effective_start, effective_end = date_range

    params = _build_common_params(hourly_to_use)
    params["daily"] = daily_to_use
    params["start_date"] = effective_start
    params["end_date"] = effective_end
    params["models"] = model

    client = build_client()
    responses = fetch_with_retries(client, OPENMETEO_HISTORICAL_FORECAST_URL, params)
    hourly_df, daily_df = parse_historical_forecast_responses(
        responses,
        hourly_vars=hourly_to_use,
        daily_vars=daily_to_use,
    )

    if hourly_df.empty:
        print("No new historical forecast weather data")
        return (
            load_dataset(DATASET_HISTORICAL_FORECAST_HOURLY, "timestamp_utc"),
            load_dataset(DATASET_HISTORICAL_FORECAST_DAILY, "date_utc"),
        )

    save_partitioned(
        df=hourly_df,
        dataset=DATASET_HISTORICAL_FORECAST_HOURLY,
        time_col="timestamp_utc",
        dedupe_cols=["timestamp_utc", "location_id"],
    )
    save_partitioned(
        df=daily_df,
        dataset=DATASET_HISTORICAL_FORECAST_DAILY,
        time_col="date_utc",
        dedupe_cols=["date_utc", "location_id"],
    )
    update_state(state, DATASET_HISTORICAL_FORECAST_HOURLY, hourly_df["timestamp_utc"].max())

    return (
        load_dataset(DATASET_HISTORICAL_FORECAST_HOURLY, "timestamp_utc"),
        load_dataset(DATASET_HISTORICAL_FORECAST_DAILY, "date_utc"),
    )


def ingest_weather_live_forecast(
    forecast_hours: int = 24,
    hourly_vars: list[str] | None = None,
) -> pd.DataFrame:
    vars_to_use = hourly_vars or HOURLY_VARS
    params = _build_common_params(vars_to_use)
    params["forecast_hours"] = forecast_hours

    client = build_client()
    responses = fetch_with_retries(client, OPENMETEO_FORECAST_URL, params)
    df = parse_hourly_responses(responses, vars_to_use)

    if df.empty:
        print("No new live forecast weather data")
        return load_dataset(DATASET_LIVE_FORECAST, "timestamp_utc")

    save_partitioned(
        df=df,
        dataset=DATASET_LIVE_FORECAST,
        time_col="timestamp_utc",
        dedupe_cols=["timestamp_utc", "location_id"],
    )

    state = load_state()
    update_state(state, DATASET_LIVE_FORECAST, df["timestamp_utc"].max())
    return load_dataset(DATASET_LIVE_FORECAST, "timestamp_utc")


if __name__ == "__main__":
    historical_df = ingest_weather_historical(
        start_date="2020-01-01",
        end_date="2026-04-19",
    )
    print("historical shape:", historical_df.shape)

    historical_forecast_hourly_df, historical_forecast_daily_df = ingest_weather_historical_forecast(
        start_date="2020-01-01",
        end_date="2026-04-19",
    )
    print("historical_forecast_hourly shape:", historical_forecast_hourly_df.shape)
    print("historical_forecast_daily shape:", historical_forecast_daily_df.shape)

    live_forecast_df = ingest_weather_live_forecast(forecast_hours=24)
    print("live_forecast shape:", live_forecast_df.shape)
