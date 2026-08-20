from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.forecast_config import DEFAULT_CONFIG, ForecastConfig


@dataclass(frozen=True, slots=True)
class ACIConfig:
    eta: float = 0.01
    window_size: int = 120
    per_horizon: bool = True


def _finite_sample_quantile(scores: list[float], alpha: float) -> tuple[float, float]:
    if not scores:
        return float("nan"), float("nan")
    n = len(scores)
    level = float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))
    return float(np.quantile(np.asarray(scores), level, method="higher")), level


class AdaptiveConformalCalibrator:
    """Adaptive conformal state with either one buffer per lead or one pooled buffer."""

    def __init__(self, horizon: int, alpha: float, config: ACIConfig) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.horizon = horizon
        self.nominal_alpha = float(alpha)
        self.config = config
        self.group_count = horizon if config.per_horizon else 1
        self.alpha_t = np.full(self.group_count, alpha, dtype=np.float64)
        self.buffers: list[list[float]] = [[] for _ in range(self.group_count)]

    def _update_group(self, group: int, scores: np.ndarray) -> None:
        buffer = self.buffers[group]
        if buffer:
            width, _ = _finite_sample_quantile(buffer, float(self.alpha_t[group]))
            miss_rate = float(np.mean(scores > width))
            self.alpha_t[group] = float(
                np.clip(
                    self.alpha_t[group]
                    + self.config.eta * (self.nominal_alpha - miss_rate),
                    1e-4,
                    1 - 1e-4,
                )
            )
        buffer.extend(float(score) for score in scores)
        maximum = self.config.window_size if self.config.per_horizon else self.config.window_size * self.horizon
        if len(buffer) > maximum:
            del buffer[: len(buffer) - maximum]

    def update(self, residuals: np.ndarray) -> None:
        scores = np.abs(np.asarray(residuals, dtype=np.float64)).reshape(-1)
        if scores.size != self.horizon:
            raise ValueError("Residual vector does not match horizon")
        if self.config.per_horizon:
            for lead, score in enumerate(scores):
                self._update_group(lead, np.asarray([score]))
        else:
            self._update_group(0, scores)

    def widths(self) -> np.ndarray:
        if self.config.per_horizon:
            return np.asarray(
                [
                    _finite_sample_quantile(self.buffers[lead], float(self.alpha_t[lead]))[0]
                    for lead in range(self.horizon)
                ],
                dtype=np.float32,
            )
        width = _finite_sample_quantile(self.buffers[0], float(self.alpha_t[0]))[0]
        return np.full(self.horizon, width, dtype=np.float32)


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape or true.ndim != 2:
        raise ValueError("Expected matching [origins, horizon] arrays")
    errors = true - pred
    mae_by_lead = np.mean(np.abs(errors), axis=0)
    rmse_by_lead = np.sqrt(np.mean(errors**2, axis=0))
    return {
        "overall_mae": float(np.mean(np.abs(errors))),
        "overall_rmse": float(np.sqrt(np.mean(errors**2))),
        "mae_by_lead": {str(i + 1): float(value) for i, value in enumerate(mae_by_lead)},
        "rmse_by_lead": {str(i + 1): float(value) for i, value in enumerate(rmse_by_lead)},
        "headline_leads": {
            str(lead): {
                "mae": float(mae_by_lead[lead - 1]),
                "rmse": float(rmse_by_lead[lead - 1]),
            }
            for lead in (1, 6, 12, 24)
            if lead <= true.shape[1]
        },
    }


def seasonal_baselines(
    samples: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> dict[str, np.ndarray]:
    """Compute causal previous-day and previous-week forecasts from each origin row."""
    daily = np.column_stack(
        [
            samples[f"load_history_lag_{24 - lead:03d}"].to_numpy(dtype=np.float32)
            for lead in range(1, config.horizon_hours + 1)
        ]
    )
    weekly = np.column_stack(
        [
            samples[f"load_history_lag_{168 - lead:03d}"].to_numpy(dtype=np.float32)
            for lead in range(1, config.horizon_hours + 1)
        ]
    )
    return {"previous_day": daily, "previous_week": weekly}


def rolling_aci_intervals(
    calibration_residuals: np.ndarray,
    evaluation_true: np.ndarray,
    evaluation_pred: np.ndarray,
    alpha: float,
    aci_config: ACIConfig,
    *,
    calibration_target_ends: pd.DatetimeIndex,
    evaluation_origins: pd.DatetimeIndex,
    evaluation_target_ends: pd.DatetimeIndex,
) -> dict[str, Any]:
    calibration = np.asarray(calibration_residuals, dtype=np.float64)
    true = np.asarray(evaluation_true, dtype=np.float64)
    pred = np.asarray(evaluation_pred, dtype=np.float64)
    if calibration.ndim != 2 or true.shape != pred.shape:
        raise ValueError("ACI expects 2D residual, truth, and prediction arrays")
    calibration_ends = pd.DatetimeIndex(calibration_target_ends)
    origins = pd.DatetimeIndex(evaluation_origins)
    target_ends = pd.DatetimeIndex(evaluation_target_ends)
    if len(calibration_ends) != len(calibration):
        raise ValueError("Each calibration residual needs a target-end timestamp")
    if len(origins) != len(true) or len(target_ends) != len(true):
        raise ValueError("Each evaluated forecast needs origin and target-end timestamps")
    if not origins.is_monotonic_increasing:
        raise ValueError("ACI evaluation origins must be chronological")
    calibrator = AdaptiveConformalCalibrator(true.shape[1], alpha, aci_config)
    pending: list[tuple[pd.Timestamp, np.ndarray]] = sorted(
        zip(calibration_ends, calibration, strict=True), key=lambda event: event[0]
    )

    widths: list[np.ndarray] = []
    for row, origin in enumerate(origins):
        matured = [event for event in pending if event[0] <= origin]
        pending = [event for event in pending if event[0] > origin]
        for _, residual in matured:
            calibrator.update(residual)
        widths.append(calibrator.widths())
        pending.append((target_ends[row], true[row] - pred[row]))
    width_array = np.stack(widths)
    lower = pred - width_array
    upper = pred + width_array
    covered = (true >= lower) & (true <= upper)
    coverage_by_lead = covered.mean(axis=0)
    width_by_lead = (upper - lower).mean(axis=0)
    return {
        "lower": lower,
        "upper": upper,
        "overall_coverage": float(covered.mean()),
        "mean_interval_width": float((upper - lower).mean()),
        "coverage_by_lead": {
            str(i + 1): float(value) for i, value in enumerate(coverage_by_lead)
        },
        "width_by_lead": {
            str(i + 1): float(value) for i, value in enumerate(width_by_lead)
        },
        "final_alpha_t": calibrator.alpha_t.tolist(),
    }


def tune_aci_on_validation(
    validation_true: np.ndarray,
    validation_pred: np.ndarray,
    validation_origins: pd.DatetimeIndex,
    validation_target_ends: pd.DatetimeIndex,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> tuple[ACIConfig, pd.DataFrame]:
    """Tune ACI only inside validation, leaving final-test outcomes untouched."""
    true = np.asarray(validation_true, dtype=np.float64)
    pred = np.asarray(validation_pred, dtype=np.float64)
    minimum_initial = 60 * 24
    if len(true) < minimum_initial * 2:
        raise ValueError("At least 120 days of hourly validation origins are required to tune ACI")
    initial_count = min(minimum_initial, len(true) // 2)
    residuals = true - pred
    rows: list[dict[str, Any]] = []
    alpha = 1.0 - config.nominal_coverage
    for eta in config.aci_eta_candidates:
        for window in config.aci_window_candidates:
            for per_horizon in config.aci_per_horizon_candidates:
                candidate = ACIConfig(eta=eta, window_size=window, per_horizon=per_horizon)
                result = rolling_aci_intervals(
                    residuals[:initial_count],
                    true[initial_count:],
                    pred[initial_count:],
                    alpha,
                    candidate,
                    calibration_target_ends=validation_target_ends[:initial_count],
                    evaluation_origins=validation_origins[initial_count:],
                    evaluation_target_ends=validation_target_ends[initial_count:],
                )
                rows.append(
                    {
                        **asdict(candidate),
                        "coverage": result["overall_coverage"],
                        "coverage_gap": abs(
                            result["overall_coverage"] - config.nominal_coverage
                        ),
                        "mean_interval_width": result["mean_interval_width"],
                    }
                )
    table = pd.DataFrame(rows)
    acceptable = table[table["coverage"] >= config.nominal_coverage - 0.02]
    ranked = acceptable if not acceptable.empty else table
    chosen = ranked.sort_values(["mean_interval_width", "coverage_gap"]).iloc[0]
    return (
        ACIConfig(
            eta=float(chosen["eta"]),
            window_size=int(chosen["window_size"]),
            per_horizon=bool(chosen["per_horizon"]),
        ),
        table.sort_values(["coverage_gap", "mean_interval_width"]).reset_index(drop=True),
    )
