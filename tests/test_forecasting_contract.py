from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from src.forecast_config import DEFAULT_CONFIG, daily_origins
from src.modeling.evaluation import (
    ACIConfig,
    AdaptiveConformalCalibrator,
    prediction_metrics,
    rolling_aci_intervals,
    seasonal_baselines,
)
from src.modeling.lstm_pipeline import DailyLoadLSTM
from src.processing.features import (
    build_daily_forecast_samples,
    impute_load_causally,
    validate_weather_vintages,
)


class ForecastOriginTests(unittest.TestCase):
    def test_daily_origins_remain_at_local_noon_across_dst(self) -> None:
        origins = daily_origins("2025-03-29", "2025-03-31")
        local = origins.tz_convert("Europe/Zurich")
        self.assertEqual(local.hour.tolist(), [12, 12, 12])
        self.assertEqual(len(origins), 3)
        self.assertEqual((origins[1] - origins[0]) / pd.Timedelta(hours=1), 23)

    def test_configured_weather_run_is_available_before_origin(self) -> None:
        for day in ("2025-01-10", "2025-07-10"):
            origin = DEFAULT_CONFIG.origin_for_local_date(day)
            run = DEFAULT_CONFIG.weather_run_for_origin(origin)
            self.assertLessEqual(DEFAULT_CONFIG.weather_available_at(run), origin)


class PointInTimeWeatherTests(unittest.TestCase):
    def _weather(self, origin: pd.Timestamp, config=DEFAULT_CONFIG) -> pd.DataFrame:
        rows = []
        run = config.weather_run_for_origin(origin)
        for lead in range(1, config.horizon_hours + 1):
            for city_number, city in enumerate(("zurich", "geneva")):
                rows.append(
                    {
                        "forecast_origin_utc": origin,
                        "weather_run_utc": run,
                        "weather_available_utc": config.weather_available_at(run),
                        "target_timestamp_utc": origin + pd.Timedelta(hours=lead),
                        "forecast_lead_hour": lead,
                        "city": city,
                        "weather_model": config.weather_model,
                        "weather_run_is_fallback": False,
                        "source": "synthetic-test",
                        "retrieved_at_utc": origin + pd.Timedelta(days=1),
                        **{
                            variable: float(lead + city_number)
                            for variable in config.weather_variables
                        },
                    }
                )
        return pd.DataFrame(rows)

    def test_rejects_weather_available_after_forecast_origin(self) -> None:
        origin = DEFAULT_CONFIG.origin_for_local_date("2025-05-01")
        weather = self._weather(origin)
        weather["weather_available_utc"] = origin + pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "issued after"):
            validate_weather_vintages(weather)

    def test_daily_sample_uses_one_run_and_targets_t_plus_1_to_24(self) -> None:
        config = replace(DEFAULT_CONFIG, lookback_hours=168)
        origin = config.origin_for_local_date("2025-05-01")
        index = pd.date_range(origin - pd.Timedelta(hours=200), periods=225, freq="h")
        load = pd.Series(np.arange(len(index), dtype=float), index=index)
        imputed = pd.Series(0, index=index, dtype="int8")
        weather = self._weather(origin, config)

        samples = build_daily_forecast_samples(load, imputed, weather, config)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples.index[0], origin)
        self.assertEqual(samples.iloc[0]["target_load_lead_01"], load.loc[origin + pd.Timedelta(hours=1)])
        self.assertEqual(samples.iloc[0]["target_load_lead_24"], load.loc[origin + pd.Timedelta(hours=24)])
        self.assertEqual(samples.iloc[0]["weather_temperature_2m_lead_24"], 24.5)


class CausalLoadTests(unittest.TestCase):
    def test_missing_load_uses_previous_week_not_future(self) -> None:
        values = pd.Series(np.arange(170, dtype=float))
        values.iloc[168] = np.nan
        filled, flag = impute_load_causally(values)
        self.assertEqual(filled.iloc[168], filled.iloc[0])
        self.assertEqual(flag.iloc[168], 1)


class EvaluationTests(unittest.TestCase):
    def test_seasonal_baselines_have_explicit_daily_and_weekly_alignment(self) -> None:
        row = {
            f"load_history_lag_{lag:03d}": float(1000 - lag)
            for lag in range(DEFAULT_CONFIG.lookback_hours)
        }
        samples = pd.DataFrame([row])
        predictions = seasonal_baselines(samples)
        self.assertEqual(predictions["previous_day"][0, 0], row["load_history_lag_023"])
        self.assertEqual(predictions["previous_day"][0, 23], row["load_history_lag_000"])
        self.assertEqual(predictions["previous_week"][0, 0], row["load_history_lag_167"])
        self.assertEqual(predictions["previous_week"][0, 23], row["load_history_lag_144"])

    def test_metrics_are_reported_by_lead(self) -> None:
        truth = np.zeros((3, 24), dtype=float)
        prediction = np.tile(np.arange(1, 25, dtype=float), (3, 1))
        metrics = prediction_metrics(truth, prediction)
        self.assertEqual(metrics["mae_by_lead"]["1"], 1.0)
        self.assertEqual(metrics["rmse_by_lead"]["24"], 24.0)
        self.assertEqual(set(metrics["headline_leads"]), {"1", "6", "12", "24"})

    def test_aci_adapts_per_horizon_after_misses(self) -> None:
        calibrator = AdaptiveConformalCalibrator(
            horizon=2,
            alpha=0.1,
            config=ACIConfig(eta=0.05, window_size=20, per_horizon=True),
        )
        for _ in range(10):
            calibrator.update(np.asarray([1.0, 1.0]))
        alpha_before = calibrator.alpha_t.copy()
        calibrator.update(np.asarray([10.0, 10.0]))
        self.assertTrue(np.all(calibrator.alpha_t < alpha_before))
        self.assertTrue(np.all(np.isfinite(calibrator.widths())))

    def test_aci_uses_a_completed_forecast_only_on_the_next_origin(self) -> None:
        calibration = np.ones((2, 2), dtype=float)
        truth = np.asarray([[100.0, 100.0], [0.0, 0.0]])
        prediction = np.zeros_like(truth)
        result = rolling_aci_intervals(
            calibration,
            truth,
            prediction,
            alpha=0.1,
            aci_config=ACIConfig(eta=0.05, window_size=2, per_horizon=True),
        )
        first_width = result["upper"][0] - result["lower"][0]
        second_width = result["upper"][1] - result["lower"][1]
        np.testing.assert_allclose(first_width, np.asarray([2.0, 2.0]))
        self.assertTrue(np.all(second_width > first_width))

    def test_daily_lstm_output_shape_is_24_hours(self) -> None:
        model = DailyLoadLSTM(context_size=5, horizon=24, hidden_size=8)
        output = model(
            history=torch.zeros((2, 336, 1)),
            context=torch.zeros((2, 5)),
        )
        self.assertEqual(tuple(output.shape), (2, 24))


if __name__ == "__main__":
    unittest.main()
