from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TIMEZONE = "Europe/Zurich"
UTC = "UTC"


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """Single source of truth for the operational forecasting contract."""

    timezone: str = LOCAL_TIMEZONE
    forecast_origin_hour_local: int = 12
    horizon_hours: int = 24
    lookback_hours: int = 336
    random_seed: int = 42

    weather_model: str = "ecmwf_ifs"
    weather_run_hour_utc: int = 0
    weather_availability_delay_hours: int = 6
    weather_archive_hours: int = 54
    weather_variables: tuple[str, ...] = (
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "wind_speed_10m",
    )

    train_end: str = "2025-06-30 23:59:59+00:00"
    validation_end: str = "2025-12-31 23:59:59+00:00"
    final_test_end: str = "2026-04-18 23:59:59+00:00"

    nominal_coverage: float = 0.90
    aci_eta_candidates: tuple[float, ...] = (0.0005, 0.001, 0.002)
    aci_window_candidates: tuple[int, ...] = (7 * 24, 30 * 24, 60 * 24)
    aci_per_horizon_candidates: tuple[bool, ...] = (True, False)

    features_path: Path = PROJECT_ROOT / "data" / "processed" / "daily_forecast_samples.parquet"
    weather_runs_path: Path = PROJECT_ROOT / "data" / "raw" / "weather_forecast_runs"
    model_path: Path = PROJECT_ROOT / "data" / "processed" / "models" / "daily_lstm_24h.pt"
    metrics_path: Path = PROJECT_ROOT / "data" / "processed" / "models" / "daily_lstm_24h.metrics.json"

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(f"target_load_lead_{lead:02d}" for lead in range(1, self.horizon_hours + 1))

    @property
    def history_columns(self) -> tuple[str, ...]:
        return tuple(f"load_history_lag_{lag:03d}" for lag in range(self.lookback_hours))

    @property
    def weather_columns(self) -> tuple[str, ...]:
        return tuple(
            f"weather_{variable}_lead_{lead:02d}"
            for variable in self.weather_variables
            for lead in range(1, self.horizon_hours + 1)
        )

    @property
    def context_columns(self) -> tuple[str, ...]:
        return (
            "load_lag_24",
            "load_lag_168",
            "load_roll_mean_24",
            "load_roll_std_24",
            "load_roll_mean_168",
            "load_roll_std_168",
            "dow_sin",
            "dow_cos",
            "doy_sin",
            "doy_cos",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            *self.weather_columns,
        )

    def origin_for_local_date(self, value: date | str | pd.Timestamp) -> pd.Timestamp:
        local_date = pd.Timestamp(value).date()
        origin_local = pd.Timestamp(
            year=local_date.year,
            month=local_date.month,
            day=local_date.day,
            hour=self.forecast_origin_hour_local,
            tz=ZoneInfo(self.timezone),
        )
        return origin_local.tz_convert(UTC)

    def weather_run_for_origin(self, origin_utc: pd.Timestamp) -> pd.Timestamp:
        origin = pd.Timestamp(origin_utc).tz_convert(UTC)
        run = origin.normalize() + pd.Timedelta(hours=self.weather_run_hour_utc)
        if self.weather_available_at(run) > origin:
            run -= pd.Timedelta(days=1)
        return run

    def weather_available_at(self, run_utc: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(run_utc).tz_convert(UTC) + pd.Timedelta(
            hours=self.weather_availability_delay_hours
        )


DEFAULT_CONFIG = ForecastConfig()


def daily_origins(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DatetimeIndex:
    """Return one DST-safe UTC origin for each local calendar day."""
    start_local = pd.Timestamp(start)
    end_local = pd.Timestamp(end)
    if start_local.tzinfo is not None:
        start_local = start_local.tz_convert(config.timezone)
    if end_local.tzinfo is not None:
        end_local = end_local.tz_convert(config.timezone)
    dates = pd.date_range(start_local.date(), end_local.date(), freq="D")
    return pd.DatetimeIndex([config.origin_for_local_date(day) for day in dates], name="forecast_origin_utc")


def hourly_origins(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DatetimeIndex:
    """Return every real hourly origin across the requested local calendar dates."""
    start_local = pd.Timestamp(start)
    end_local = pd.Timestamp(end)
    if start_local.tzinfo is None:
        start_local = start_local.tz_localize(config.timezone)
    else:
        start_local = start_local.tz_convert(config.timezone)
    if end_local.tzinfo is None:
        end_local = end_local.tz_localize(config.timezone)
    else:
        end_local = end_local.tz_convert(config.timezone)
    first = start_local.normalize()
    last = end_local.normalize() + pd.DateOffset(days=1) - pd.Timedelta(hours=1)
    return pd.date_range(first, last, freq="h", name="forecast_origin_utc").tz_convert(UTC)
