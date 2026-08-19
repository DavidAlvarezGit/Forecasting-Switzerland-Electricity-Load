from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.streamlit_app import _baseline_label
from src.modeling.inference import _AdaptiveConformalCalibrator, _previous_day_baseline
from src.modeling.lstm_pipeline import previous_day_baseline
from src.processing.features import FORECAST_WEATHER_FEATURES, build_forecast_lead_features
from src.processing.helpers import impute_load_causally, impute_weather


class CausalImputationTests(unittest.TestCase):
    def test_weather_fill_never_uses_a_later_observation(self) -> None:
        index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        weather = pd.DataFrame(
            {"zurich__temperature_2m": [1.0, np.nan, 9.0]},
            index=index,
        )

        result = impute_weather(weather)

        self.assertEqual(result.iloc[1, 0], 1.0)

    def test_leading_weather_gap_is_not_backward_filled(self) -> None:
        index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
        weather = pd.DataFrame(
            {"zurich__temperature_2m": [np.nan, 9.0]},
            index=index,
        )

        result = impute_weather(weather)

        self.assertTrue(pd.isna(result.iloc[0, 0]))

    def test_load_gap_uses_previous_week_and_sets_audit_flag(self) -> None:
        index = pd.date_range("2026-01-01", periods=170, freq="h", tz="UTC")
        values = np.arange(170, dtype=float)
        values[168] = np.nan
        load = pd.DataFrame({"load_mw": values}, index=index)

        result = impute_load_causally(load)

        self.assertEqual(result.iloc[168]["load_mw"], result.iloc[0]["load_mw"])
        self.assertEqual(result.iloc[168]["load_was_imputed"], 1)


class ForecastWeatherTests(unittest.TestCase):
    def test_lead_features_use_forecast_targets_after_issue_time(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        rows: list[dict[str, object]] = []
        for timestamp_idx, timestamp in enumerate(timestamps):
            for city_idx, city in enumerate(("zurich", "geneva")):
                row: dict[str, object] = {"timestamp_utc": timestamp, "city": city}
                for feature_idx, feature in enumerate(FORECAST_WEATHER_FEATURES):
                    row[feature] = float(timestamp_idx * 10 + city_idx + feature_idx)
                rows.append(row)
        forecast = pd.DataFrame(rows)

        result = build_forecast_lead_features(forecast, timestamps[:2], horizon=2)

        self.assertEqual(result.iloc[0]["forecast_ch_mean_temperature_2m_lead_01"], 10.5)
        self.assertEqual(result.iloc[0]["forecast_ch_mean_temperature_2m_lead_02"], 20.5)


class BenchmarkAndIntervalTests(unittest.TestCase):
    def test_dashboard_always_names_the_fixed_baseline(self) -> None:
        self.assertEqual(_baseline_label(None), "Same hour yesterday")
        self.assertEqual(_baseline_label("unavailable"), "Same hour yesterday")

    def test_previous_day_baseline_uses_only_known_history(self) -> None:
        values = np.arange(240, dtype=np.float32)
        anchors = np.asarray([200])

        prediction = _previous_day_baseline(values, anchors, horizon=4)

        np.testing.assert_array_equal(prediction[0], [176, 177, 178, 179])

        target = pd.Series(values, index=pd.date_range("2025-01-01", periods=240, freq="h"))
        actual, training_prediction, issue_times = previous_day_baseline(
            target, lookback=200, horizon=4
        )
        np.testing.assert_array_equal(training_prediction[0], [176, 177, 178, 179])
        np.testing.assert_array_equal(actual[0], [200, 201, 202, 203])
        self.assertEqual(issue_times[0], target.index[200])

    def test_aci_alpha_adapts_after_misses(self) -> None:
        calibrator = _AdaptiveConformalCalibrator(
            horizon=2,
            alpha=0.1,
            eta=0.05,
            window_size=20,
            per_horizon=True,
        )
        for _ in range(10):
            calibrator.update(np.asarray([1.0, 1.0]))
        alpha_before = calibrator.alpha_t.copy()
        calibrator.update(np.asarray([10.0, 10.0]))

        self.assertTrue(np.all(calibrator.alpha_t < alpha_before))
        self.assertTrue(np.all(np.isfinite(calibrator.widths())))


if __name__ == "__main__":
    unittest.main()
