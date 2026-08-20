from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.forecast_config import DEFAULT_CONFIG, ForecastConfig, daily_origins

SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
SOURCE_NAME = "open-meteo-single-runs"

CITIES = (
    ("zurich", 47.3769, 8.5417),
    ("geneva", 46.2044, 6.1432),
    ("bern", 46.9480, 7.4474),
    ("basel", 47.5596, 7.5886),
    ("lausanne", 46.5197, 6.6323),
    ("lucerne", 47.0502, 8.3093),
    ("st_gallen", 47.4245, 9.3767),
    ("lugano", 46.0101, 8.9600),
    ("interlaken", 46.6863, 7.8632),
    ("central_ch", 46.8182, 8.2275),
)


class ModelRunUnavailable(RuntimeError):
    """Raised when the archive explicitly reports a missing model run."""


def _request_json(
    params: dict[str, Any],
    retries: int = 5,
    timeout_seconds: int = 60,
) -> list[dict[str, Any]]:
    for attempt in range(retries):
        try:
            response = requests.get(SINGLE_RUNS_URL, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            if "modelRunUnavailable" in response.text:
                raise ModelRunUnavailable(response.text)
            payload = response.json()
            return payload if isinstance(payload, list) else [payload]
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Open-Meteo request failed without an exception")


def _request_params(
    run_utc: pd.Timestamp,
    config: ForecastConfig,
    locations: tuple[tuple[str, float, float], ...] = CITIES,
) -> dict[str, Any]:
    return {
        "latitude": [latitude for _, latitude, _ in locations],
        "longitude": [longitude for _, _, longitude in locations],
        "hourly": list(config.weather_variables),
        "models": config.weather_model,
        "run": run_utc.strftime("%Y-%m-%dT%H:%M"),
        "forecast_hours": 192,
        "timezone": "UTC",
    }


def _request_run(run_utc: pd.Timestamp, config: ForecastConfig) -> list[dict[str, Any]]:
    """Request all locations together, then split only if archive streaming fails."""
    try:
        return _request_json(_request_params(run_utc, config))
    except (requests.RequestException, ValueError, ModelRunUnavailable) as combined_error:
        payload: list[dict[str, Any]] = []
        try:
            for location in CITIES:
                payload.extend(_request_json(_request_params(run_utc, config, (location,))))
            return payload
        except (requests.RequestException, ValueError, ModelRunUnavailable) as split_error:
            raise RuntimeError(f"Weather run {run_utc} could not be retrieved") from (
                split_error or combined_error
            )


def _payload_is_complete(
    payload: list[dict[str, Any]],
    target_index: pd.DatetimeIndex,
    config: ForecastConfig,
) -> bool:
    if len(payload) != len(CITIES):
        return False
    for location in payload:
        hourly = location.get("hourly", {})
        timestamps = pd.to_datetime(hourly.get("time", []), utc=True)
        positions = np.flatnonzero(timestamps.isin(target_index))
        if len(positions) != config.horizon_hours:
            return False
        for variable in config.weather_variables:
            values = pd.to_numeric(pd.Series(hourly.get(variable, [])), errors="coerce")
            if len(values) <= int(positions.max()) or values.iloc[positions].isna().any():
                return False
    return True


def fetch_weather_run(
    forecast_origin_utc: pd.Timestamp,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Fetch one complete weather vintage available at a daily forecast origin."""
    origin = pd.Timestamp(forecast_origin_utc).tz_convert("UTC")
    target_index = pd.date_range(
        origin + pd.Timedelta(hours=1),
        periods=config.horizon_hours,
        freq="h",
        tz="UTC",
    )
    scheduled_run = config.weather_run_for_origin(origin)
    candidate_runs = tuple(
        scheduled_run - pd.Timedelta(hours=offset)
        for offset in range(0, 121, 12)
    )
    payload: list[dict[str, Any]] | None = None
    last_error: Exception | None = None
    run_utc: pd.Timestamp | None = None
    for candidate in candidate_runs:
        try:
            payload = _request_run(candidate, config)
            if not _payload_is_complete(payload, target_index, config):
                split_payload: list[dict[str, Any]] = []
                for location in CITIES:
                    split_payload.extend(
                        _request_json(_request_params(candidate, config, (location,)))
                    )
                payload = split_payload
            if not _payload_is_complete(payload, target_index, config):
                raise ValueError(f"Weather run {candidate} has incomplete target values")
            run_utc = candidate
            break
        except (requests.RequestException, ValueError, RuntimeError) as error:
            last_error = error
    if payload is None or run_utc is None:
        raise RuntimeError(
            f"No complete weather run from the scheduled or ten preceding cycles is available for {origin}"
        ) from last_error

    available_utc = config.weather_available_at(run_utc)
    if available_utc > origin:
        raise ValueError(
            f"Weather run {run_utc} is not available by forecast origin {origin}."
        )

    if len(payload) != len(CITIES):
        raise ValueError(f"Expected {len(CITIES)} locations, received {len(payload)}")

    retrieved_at = pd.Timestamp.now(tz="UTC")
    frames: list[pd.DataFrame] = []
    for (city, requested_lat, requested_lon), location in zip(CITIES, payload, strict=True):
        hourly = location.get("hourly", {})
        timestamps = pd.to_datetime(hourly.get("time", []), utc=True)
        frame = pd.DataFrame({"target_timestamp_utc": timestamps})
        for variable in config.weather_variables:
            values = hourly.get(variable)
            if values is None:
                raise ValueError(f"Open-Meteo response is missing {variable} for {city}")
            frame[variable] = values
        frame = frame[frame["target_timestamp_utc"].isin(target_index)].copy()
        frame["city"] = city
        frame["requested_latitude"] = requested_lat
        frame["requested_longitude"] = requested_lon
        frame["grid_latitude"] = location.get("latitude")
        frame["grid_longitude"] = location.get("longitude")
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result["forecast_origin_utc"] = origin
    result["forecast_origin_local"] = origin.tz_convert(config.timezone)
    result["weather_run_utc"] = run_utc
    result["weather_run_is_fallback"] = run_utc != scheduled_run
    result["weather_available_utc"] = available_utc
    result["forecast_lead_hour"] = (
        (result["target_timestamp_utc"] - origin) / pd.Timedelta(hours=1)
    ).astype("int16")
    result["weather_model"] = config.weather_model
    result["source"] = SOURCE_NAME
    result["source_url"] = SINGLE_RUNS_URL
    result["retrieved_at_utc"] = retrieved_at

    expected_rows = len(CITIES) * config.horizon_hours
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} run rows, received {len(result)}")
    if not result["forecast_lead_hour"].between(1, config.horizon_hours).all():
        raise ValueError("Weather run contains targets outside the configured horizon")
    if not (result["weather_available_utc"] <= result["forecast_origin_utc"]).all():
        raise ValueError("A weather value was issued after its forecast origin")
    if result.groupby("forecast_origin_utc")["weather_run_utc"].nunique().max() != 1:
        raise ValueError("A forecast origin contains more than one weather run")
    return result.sort_values(["forecast_origin_utc", "target_timestamp_utc", "city"])


def _partition_path(year: int, config: ForecastConfig) -> Path:
    return config.weather_runs_path / f"year={year}.parquet"


def _save_partitions(data: pd.DataFrame, config: ForecastConfig) -> None:
    local_year = data["forecast_origin_utc"].dt.tz_convert(config.timezone).dt.year
    for year, frame in data.groupby(local_year):
        path = _partition_path(int(year), config)
        if path.exists():
            existing = pd.read_parquet(path)
            frame = pd.concat([existing, frame], ignore_index=True)
        frame = (
            frame.sort_values("retrieved_at_utc")
            .drop_duplicates(
                ["forecast_origin_utc", "target_timestamp_utc", "city"],
                keep="last",
            )
            .sort_values(["forecast_origin_utc", "target_timestamp_utc", "city"])
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def load_weather_runs(config: ForecastConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    paths = sorted(config.weather_runs_path.glob("year=*.parquet"))
    if not paths:
        return pd.DataFrame()
    data = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    for column in (
        "forecast_origin_utc",
        "weather_run_utc",
        "weather_available_utc",
        "target_timestamp_utc",
        "retrieved_at_utc",
    ):
        data[column] = pd.to_datetime(data[column], utc=True)
    scheduled_run = data["forecast_origin_utc"].dt.normalize() + pd.Timedelta(
        hours=config.weather_run_hour_utc
    )
    data["weather_run_is_fallback"] = data["weather_run_utc"] != scheduled_run
    return data.sort_values(["forecast_origin_utc", "target_timestamp_utc", "city"])


def ingest_weather_forecast_runs(
    start_date: str,
    end_date: str,
    config: ForecastConfig = DEFAULT_CONFIG,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Incrementally archive one auditable ECMWF run per local forecast day."""
    origins = daily_origins(start_date, end_date, config)
    existing = load_weather_runs(config)
    existing_origins: set[pd.Timestamp] = set()
    if not existing.empty:
        expected_rows = len(CITIES) * config.horizon_hours
        grouped = existing.groupby("forecast_origin_utc")
        complete = grouped.size().eq(expected_rows) & grouped[
            list(config.weather_variables)
        ].count().eq(expected_rows).all(axis=1)
        existing_origins = set(pd.DatetimeIndex(complete[complete].index))
    missing = [origin for origin in origins if origin not in existing_origins]
    if not missing:
        return existing

    pending = missing
    completed_total = 0
    for attempt in range(1, 4):
        failed: list[pd.Timestamp] = []
        checkpoint: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_weather_run, origin, config): origin for origin in pending
            }
            for future in as_completed(futures):
                origin = futures[future]
                try:
                    checkpoint.append(future.result())
                    completed_total += 1
                except (requests.RequestException, ValueError, RuntimeError) as error:
                    failed.append(origin)
                    print(f"Will retry weather run for {origin}: {type(error).__name__}", flush=True)
                if len(checkpoint) >= 25:
                    _save_partitions(pd.concat(checkpoint, ignore_index=True), config)
                    checkpoint.clear()
                    print(
                        f"Archived {completed_total}/{len(missing)} missing weather runs",
                        flush=True,
                    )
        if checkpoint:
            _save_partitions(pd.concat(checkpoint, ignore_index=True), config)
            print(
                f"Archived {completed_total}/{len(missing)} missing weather runs",
                flush=True,
            )
        if not failed:
            break
        pending = failed
        print(f"Retry pass {attempt}: {len(pending)} runs remain", flush=True)
    else:
        raise RuntimeError(f"Could not archive {len(pending)} weather runs after three passes")

    return load_weather_runs(config)


def latest_scheduled_origin(
    now: pd.Timestamp | None = None,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.Timestamp:
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    local_now = current.tz_convert(config.timezone)
    origin = config.origin_for_local_date(local_now)
    if origin > current:
        origin = config.origin_for_local_date(local_now - pd.Timedelta(days=1))
    return origin


if __name__ == "__main__":
    result = ingest_weather_forecast_runs("2024-03-15", "2026-04-18")
    print(
        f"Archived {result['forecast_origin_utc'].nunique():,} daily runs and "
        f"{len(result):,} location/lead rows."
    )
