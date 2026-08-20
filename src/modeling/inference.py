from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.forecast_config import DEFAULT_CONFIG
from src.modeling.evaluation import (
    ACIConfig,
    AdaptiveConformalCalibrator,
    prediction_metrics,
    rolling_aci_intervals,
    seasonal_baselines,
)
from src.modeling.lstm_pipeline import DailyLoadLSTM


@dataclass(slots=True)
class LoadedLSTMArtifact:
    model: DailyLoadLSTM
    history_columns: list[str]
    context_columns: list[str]
    target_columns: list[str]
    history_mean: float
    history_scale: float
    context_mean: np.ndarray
    context_scale: np.ndarray
    target_mean: float
    target_scale: float
    horizon: int
    lookback: int
    device: str
    model_version: int
    forecast_config: dict[str, Any]
    aci_config: ACIConfig

    @property
    def feature_cols(self) -> list[str]:
        return [*self.history_columns, *self.context_columns]

    @property
    def target_col(self) -> str:
        return self.target_columns[0]


def load_lstm_artifact(
    model_path: str | Path,
    device: str | None = None,
) -> LoadedLSTMArtifact:
    resolved = Path(model_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved}")
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(resolved, map_location=run_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(resolved, map_location=run_device)
    if int(checkpoint.get("model_version", 0)) < 3:
        raise ValueError("This dashboard requires the daily-origin model artifact (version 3).")

    context_columns = list(checkpoint["context_columns"])
    target_columns = list(checkpoint["target_columns"])
    model = DailyLoadLSTM(
        context_size=len(context_columns),
        horizon=len(target_columns),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    ).to(run_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    aci = checkpoint["aci_config"]
    return LoadedLSTMArtifact(
        model=model,
        history_columns=list(checkpoint["history_columns"]),
        context_columns=context_columns,
        target_columns=target_columns,
        history_mean=float(np.asarray(checkpoint["history_mean"]).reshape(-1)[0]),
        history_scale=float(np.asarray(checkpoint["history_scale"]).reshape(-1)[0]),
        context_mean=np.asarray(checkpoint["context_mean"], dtype=np.float32),
        context_scale=np.asarray(checkpoint["context_scale"], dtype=np.float32),
        target_mean=float(np.asarray(checkpoint["target_mean"]).reshape(-1)[0]),
        target_scale=float(np.asarray(checkpoint["target_scale"]).reshape(-1)[0]),
        horizon=len(target_columns),
        lookback=len(checkpoint["history_columns"]),
        device=run_device,
        model_version=int(checkpoint["model_version"]),
        forecast_config=dict(checkpoint["forecast_config"]),
        aci_config=ACIConfig(
            eta=float(aci["eta"]),
            window_size=int(aci["window_size"]),
            per_horizon=bool(aci["per_horizon"]),
        ),
    )


def _validate_samples(samples: pd.DataFrame, artifact: LoadedLSTMArtifact) -> pd.DataFrame:
    if not isinstance(samples.index, pd.DatetimeIndex):
        raise ValueError("Daily samples must have a DatetimeIndex of forecast origins")
    missing = [column for column in artifact.feature_cols if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing model inputs: {missing[:10]}")
    local_hours = samples.index.tz_convert(
        artifact.forecast_config.get("timezone", "Europe/Zurich")
    ).hour
    expected_hour = int(artifact.forecast_config.get("forecast_origin_hour_local", 12))
    if not (local_hours == expected_hour).all():
        raise ValueError("Samples contain an origin outside the configured local issue hour")
    return samples.sort_index()


def _predict_rows(samples: pd.DataFrame, artifact: LoadedLSTMArtifact) -> np.ndarray:
    frame = _validate_samples(samples, artifact)
    history = frame[artifact.history_columns].to_numpy(dtype=np.float32)
    context = frame[artifact.context_columns].to_numpy(dtype=np.float32)
    history = (history - artifact.history_mean) / artifact.history_scale
    context = (context - artifact.context_mean) / artifact.context_scale
    outputs: list[np.ndarray] = []
    artifact.model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), 128):
            history_tensor = torch.tensor(
                history[start : start + 128, :, None], dtype=torch.float32, device=artifact.device
            )
            context_tensor = torch.tensor(
                context[start : start + 128], dtype=torch.float32, device=artifact.device
            )
            outputs.append(artifact.model(history_tensor, context_tensor).cpu().numpy())
    scaled = np.concatenate(outputs, axis=0)
    return scaled * artifact.target_scale + artifact.target_mean


def _forecast_index(origin: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")


def forecast_next_horizon(
    samples: pd.DataFrame,
    artifact: LoadedLSTMArtifact,
) -> pd.Series:
    frame = _validate_samples(samples, artifact)
    origin = frame.index[-1]
    prediction = _predict_rows(frame.iloc[[-1]], artifact)[0]
    return pd.Series(
        prediction,
        index=_forecast_index(origin, artifact.horizon),
        name="forecast_load_mw",
    )


def forecast_next_horizon_with_intervals(
    samples: pd.DataFrame,
    artifact: LoadedLSTMArtifact,
    alpha: float | None = None,
    **_: Any,
) -> pd.DataFrame:
    frame = _validate_samples(samples, artifact)
    alpha_value = (
        1.0 - float(artifact.forecast_config.get("nominal_coverage", 0.90))
        if alpha is None
        else float(alpha)
    )
    point = forecast_next_horizon(frame, artifact)
    calibration = frame.iloc[:-1]
    calibration = calibration[calibration[artifact.target_columns].notna().all(axis=1)]
    if calibration.empty:
        raise ValueError("No completed daily forecasts are available for ACI calibration")
    prediction = _predict_rows(calibration, artifact)
    actual = calibration[artifact.target_columns].to_numpy(dtype=np.float32)
    calibrator = AdaptiveConformalCalibrator(artifact.horizon, alpha_value, artifact.aci_config)
    for residual in (actual - prediction)[-artifact.aci_config.window_size :]:
        calibrator.update(residual)
    width = calibrator.widths()
    values = point.to_numpy(dtype=np.float32)
    return pd.DataFrame(
        {
            "forecast_load_mw": values,
            "lower_pi": values - width,
            "upper_pi": values + width,
            "interval_width": 2 * width,
        },
        index=point.index,
    )


def _long_history(
    samples: pd.DataFrame,
    true: np.ndarray,
    model: np.ndarray,
    baselines: dict[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row, origin in enumerate(samples.index):
        for lead, target_timestamp in enumerate(_forecast_index(origin, true.shape[1]), start=1):
            records.append(
                {
                    "forecast_origin_utc": origin,
                    "target_timestamp_utc": target_timestamp,
                    "lead_hour": lead,
                    "true_load_mw": float(true[row, lead - 1]),
                    "forecast_load_mw": float(model[row, lead - 1]),
                    "previous_day_load_mw": float(baselines["previous_day"][row, lead - 1]),
                    "previous_week_load_mw": float(baselines["previous_week"][row, lead - 1]),
                    "lower_pi": float(lower[row, lead - 1]),
                    "upper_pi": float(upper[row, lead - 1]),
                }
            )
    return pd.DataFrame(records).set_index("target_timestamp_utc").sort_index()


def evaluate_recent_backtest(
    samples: pd.DataFrame,
    artifact: LoadedLSTMArtifact,
    alpha: float | None = None,
    eval_windows: int = 120,
    history_windows: int = 120,
    **_: Any,
) -> dict[str, Any]:
    """Evaluate only daily origins, with validation residuals seeding final-test ACI."""
    frame = _validate_samples(samples, artifact)
    validation = frame[frame["split"] == "validation"]
    evaluation = frame[frame["split"] == "final_test"].tail(eval_windows)
    if validation.empty or evaluation.empty:
        raise ValueError("Validation and final-test daily origins are required")
    val_true = validation[artifact.target_columns].to_numpy(dtype=np.float32)
    eval_true = evaluation[artifact.target_columns].to_numpy(dtype=np.float32)
    val_pred = _predict_rows(validation, artifact)
    eval_pred = _predict_rows(evaluation, artifact)
    baselines = seasonal_baselines(evaluation, DEFAULT_CONFIG)
    alpha_value = (
        1.0 - float(artifact.forecast_config.get("nominal_coverage", 0.90))
        if alpha is None
        else float(alpha)
    )
    intervals = rolling_aci_intervals(
        val_true - val_pred,
        eval_true,
        eval_pred,
        alpha_value,
        artifact.aci_config,
    )
    model_metrics = prediction_metrics(eval_true, eval_pred)
    baseline_metrics = {
        name: prediction_metrics(eval_true, prediction) for name, prediction in baselines.items()
    }
    history_samples = evaluation.tail(history_windows)
    history_start = len(evaluation) - len(history_samples)
    history = _long_history(
        history_samples,
        eval_true[history_start:],
        eval_pred[history_start:],
        {name: values[history_start:] for name, values in baselines.items()},
        intervals["lower"][history_start:],
        intervals["upper"][history_start:],
    )
    return {
        "mae_model": model_metrics["overall_mae"],
        "rmse_model": model_metrics["overall_rmse"],
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "coverage": intervals["overall_coverage"],
        "target_coverage": 1.0 - alpha_value,
        "mean_interval_width": intervals["mean_interval_width"],
        "coverage_by_lead": intervals["coverage_by_lead"],
        "width_by_lead": intervals["width_by_lead"],
        "n_eval_windows": len(evaluation),
        "n_calibration_windows": len(validation),
        "aci_config": {
            "eta": artifact.aci_config.eta,
            "window_size": artifact.aci_config.window_size,
            "per_horizon": artifact.aci_config.per_horizon,
        },
        "interval_method": "adaptive_conformal_inference",
        "forecast_history_df": history,
        "latest_window_df": history[history["forecast_origin_utc"] == evaluation.index[-1]],
    }
