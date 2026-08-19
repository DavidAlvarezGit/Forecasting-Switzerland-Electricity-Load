from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(slots=True)
class LoadedLSTMArtifact:
    model: nn.Module
    feature_cols: list[str]
    target_col: str
    lookback: int
    horizon: int
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    device: str
    model_version: int


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_horizon: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out)


def _infer_architecture(state_dict: dict[str, torch.Tensor], horizon: int | None = None) -> tuple[int, int, int, int]:
    weight_ih_keys = sorted(
        key for key in state_dict.keys() if key.startswith("lstm.weight_ih_l") and "reverse" not in key
    )
    if not weight_ih_keys:
        raise ValueError("Invalid checkpoint: missing LSTM layer weights.")

    num_layers = len(weight_ih_keys)
    input_size = int(state_dict["lstm.weight_ih_l0"].shape[1])
    hidden_size = int(state_dict["head.weight"].shape[1])
    output_horizon = int(horizon) if horizon is not None else int(state_dict["head.weight"].shape[0])

    return input_size, hidden_size, num_layers, output_horizon


def _resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lstm_artifact(model_path: str | Path, device: str | None = None) -> LoadedLSTMArtifact:
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved_path}")

    run_device = _resolve_device(device)
    # PyTorch 2.6 changed torch.load default to weights_only=True, which cannot
    # deserialize our full training artifact dict (numpy arrays + metadata).
    try:
        checkpoint: dict[str, Any] = torch.load(
            resolved_path,
            map_location=run_device,
            weights_only=False,
        )
    except TypeError:
        # Backward compatibility for older torch versions that do not expose
        # the weights_only argument.
        checkpoint = torch.load(resolved_path, map_location=run_device)

    state_dict = checkpoint["model_state_dict"]
    horizon = int(checkpoint.get("horizon", 0)) or None
    input_size, hidden_size, num_layers, output_horizon = _infer_architecture(state_dict, horizon=horizon)

    model = LSTMRegressor(
        input_size=input_size,
        output_horizon=output_horizon,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
    ).to(run_device)
    model.load_state_dict(state_dict)
    model.eval()

    x_mean = np.asarray(checkpoint["x_scaler_mean"], dtype=np.float32)
    x_scale = np.asarray(checkpoint["x_scaler_scale"], dtype=np.float32)
    x_scale = np.where(x_scale == 0, 1.0, x_scale)

    y_mean = float(np.asarray(checkpoint["y_scaler_mean"], dtype=np.float32).reshape(-1)[0])
    y_scale = float(np.asarray(checkpoint["y_scaler_scale"], dtype=np.float32).reshape(-1)[0])
    if y_scale == 0:
        y_scale = 1.0

    feature_cols = list(checkpoint["feature_cols"])
    target_col = str(checkpoint.get("target_col", "load_load_mw"))
    lookback = int(checkpoint["lookback"])

    return LoadedLSTMArtifact(
        model=model,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=lookback,
        horizon=output_horizon,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        device=run_device,
        model_version=int(checkpoint.get("model_version", 1)),
    )


def _infer_frequency(index: pd.DatetimeIndex) -> str:
    if len(index) < 3:
        return "h"
    inferred = pd.infer_freq(index[-200:])
    return inferred or "h"


def _predict_scaled_batch(
    model: nn.Module,
    x_batch: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    tensor = torch.tensor(x_batch, dtype=torch.float32, device=device)
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            xb = tensor[start : start + batch_size]

            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(xb)
            else:
                pred = model(xb)

            outputs.append(pred.float().cpu().numpy())

    return np.concatenate(outputs, axis=0)


def forecast_next_horizon(df: pd.DataFrame, artifact: LoadedLSTMArtifact) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe must have a DatetimeIndex.")

    missing = [col for col in artifact.feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:10]}")

    if len(df) < artifact.lookback:
        raise ValueError(
            f"Need at least {artifact.lookback} rows to forecast, but got {len(df)}."
        )

    window = df[artifact.feature_cols].sort_index().iloc[-artifact.lookback :].to_numpy(dtype=np.float32)
    x_scaled = (window - artifact.x_mean) / artifact.x_scale
    x_tensor = torch.tensor(x_scaled[None, :, :], dtype=torch.float32, device=artifact.device)

    with torch.no_grad():
        if artifact.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y_pred_scaled = artifact.model(x_tensor).float().cpu().numpy()[0]
        else:
            y_pred_scaled = artifact.model(x_tensor).cpu().numpy()[0]

    y_pred = y_pred_scaled * artifact.y_scale + artifact.y_mean

    freq = _infer_frequency(df.index)
    last_ts = df.index.max()
    start = last_ts + pd.tseries.frequencies.to_offset(freq)
    forecast_index = pd.date_range(start=start, periods=artifact.horizon, freq=freq)

    return pd.Series(y_pred, index=forecast_index, name="forecast_load_mw")


def _finite_sample_quantile(scores: list[float], alpha: float) -> tuple[float, float]:
    if not scores:
        return float("nan"), float("nan")
    n_scores = len(scores)
    level = float(np.clip(np.ceil((n_scores + 1) * (1.0 - alpha)) / n_scores, 0.0, 1.0))
    return float(np.quantile(np.asarray(scores), level, method="higher")), level


class _AdaptiveConformalCalibrator:
    """Rolling ACI state for either per-lead or pooled residual scores."""

    def __init__(
        self,
        horizon: int,
        alpha: float,
        eta: float,
        window_size: int,
        per_horizon: bool,
    ) -> None:
        self.horizon = horizon
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.per_horizon = per_horizon
        self.group_count = horizon if per_horizon else 1
        self.max_scores = window_size if per_horizon else window_size * horizon
        self.alpha_t = np.full(self.group_count, self.alpha, dtype=np.float64)
        self.score_buffers: list[list[float]] = [[] for _ in range(self.group_count)]

    def _update_group(self, group: int, new_scores: np.ndarray) -> None:
        buffer = self.score_buffers[group]
        if buffer:
            q_hat, _ = _finite_sample_quantile(buffer, float(self.alpha_t[group]))
            error_rate = float(np.mean(new_scores > q_hat))
            self.alpha_t[group] = np.clip(
                self.alpha_t[group] + self.eta * (self.alpha - error_rate),
                1e-4,
                1.0 - 1e-4,
            )
        buffer.extend(float(score) for score in new_scores)
        if len(buffer) > self.max_scores:
            del buffer[: len(buffer) - self.max_scores]

    def update(self, absolute_residuals: np.ndarray) -> None:
        residuals = np.asarray(absolute_residuals, dtype=np.float64).reshape(-1)
        if residuals.size != self.horizon:
            raise ValueError("Residual vector does not match the forecast horizon.")
        if self.per_horizon:
            for step, score in enumerate(residuals):
                self._update_group(step, np.asarray([score]))
        else:
            self._update_group(0, residuals)

    def widths(self) -> np.ndarray:
        if self.per_horizon:
            return np.asarray(
                [
                    _finite_sample_quantile(self.score_buffers[step], float(self.alpha_t[step]))[0]
                    for step in range(self.horizon)
                ],
                dtype=np.float32,
            )
        width = _finite_sample_quantile(self.score_buffers[0], float(self.alpha_t[0]))[0]
        return np.full(self.horizon, width, dtype=np.float32)

    def quantile_levels(self) -> np.ndarray:
        if self.per_horizon:
            return np.asarray(
                [
                    _finite_sample_quantile(self.score_buffers[step], float(self.alpha_t[step]))[1]
                    for step in range(self.horizon)
                ],
                dtype=np.float32,
            )
        level = _finite_sample_quantile(self.score_buffers[0], float(self.alpha_t[0]))[1]
        return np.full(self.horizon, level, dtype=np.float32)

    def alpha_by_horizon(self) -> np.ndarray:
        if self.per_horizon:
            return self.alpha_t.astype(np.float32)
        return np.full(self.horizon, self.alpha_t[0], dtype=np.float32)


def _predict_anchors(
    x_all: np.ndarray,
    anchors: np.ndarray,
    artifact: LoadedLSTMArtifact,
    batch_size: int,
) -> np.ndarray:
    x_windows = np.stack(
        [x_all[anchor - artifact.lookback : anchor] for anchor in anchors],
        axis=0,
    )
    x_scaled = (x_windows - artifact.x_mean) / artifact.x_scale
    prediction_scaled = _predict_scaled_batch(
        model=artifact.model,
        x_batch=x_scaled,
        device=artifact.device,
        batch_size=batch_size,
    )
    return prediction_scaled * artifact.y_scale + artifact.y_mean


def _true_horizons(y_all: np.ndarray, anchors: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([y_all[anchor : anchor + horizon] for anchor in anchors], axis=0)


def _previous_day_baseline(
    y_all: np.ndarray,
    anchors: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Repeat the observed values from the same hours one day earlier."""
    offsets = np.arange(horizon)
    return np.stack(
        [y_all[anchor + offsets - 24] for anchor in anchors]
    ).astype(np.float32)


def forecast_next_horizon_with_intervals(
    df: pd.DataFrame,
    artifact: LoadedLSTMArtifact,
    alpha: float = 0.1,
    calibration_windows: int = 240,
    batch_size: int = 64,
    per_horizon: bool = True,
    aci_eta: float = 0.01,
) -> pd.DataFrame:
    if not (0 < float(alpha) < 1):
        raise ValueError("alpha must be in (0, 1).")
    if calibration_windows < 2:
        raise ValueError("calibration_windows must be >= 2 for ACI.")
    if not (0 < float(aci_eta) <= 1):
        raise ValueError("aci_eta must be in (0, 1].")

    point_forecast = forecast_next_horizon(df, artifact)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe must have a DatetimeIndex.")
    missing = [col for col in artifact.feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:10]}")
    if artifact.target_col not in df.columns:
        raise ValueError(f"Missing target column '{artifact.target_col}' for interval calibration.")

    df_sorted = df.sort_index()
    x_all = df_sorted[artifact.feature_cols].to_numpy(dtype=np.float32)
    y_all = df_sorted[artifact.target_col].to_numpy(dtype=np.float32)
    max_anchor = len(df_sorted) - artifact.horizon
    anchors = np.arange(artifact.lookback, max_anchor + 1, dtype=int)
    if anchors.size < 2:
        raise ValueError("Not enough fully observed windows for adaptive interval calibration.")
    anchors = anchors[-calibration_windows:]

    predictions = _predict_anchors(x_all, anchors, artifact, batch_size)
    actual = _true_horizons(y_all, anchors, artifact.horizon)
    calibrator = _AdaptiveConformalCalibrator(
        horizon=artifact.horizon,
        alpha=alpha,
        eta=aci_eta,
        window_size=calibration_windows,
        per_horizon=per_horizon,
    )
    for residual_row in np.abs(actual - predictions):
        calibrator.update(residual_row)

    q_hat = calibrator.widths()
    point_values = point_forecast.to_numpy(dtype=np.float32)
    lower = point_values - q_hat
    upper = point_values + q_hat
    return pd.DataFrame(
        {
            "forecast_load_mw": point_values,
            "lower_pi": lower,
            "upper_pi": upper,
            "interval_width": upper - lower,
            "q_hat": q_hat,
            "alpha": np.full(artifact.horizon, alpha, dtype=np.float32),
            "alpha_t": calibrator.alpha_by_horizon(),
            "aci_eta": np.full(artifact.horizon, aci_eta, dtype=np.float32),
            "calibration_windows_used": np.full(artifact.horizon, len(anchors), dtype=np.int32),
            "quantile_level": calibrator.quantile_levels(),
        },
        index=point_forecast.index,
    )


def evaluate_recent_backtest(
    df: pd.DataFrame,
    artifact: LoadedLSTMArtifact,
    alpha: float = 0.1,
    calibration_windows: int = 240,
    eval_windows: int = 120,
    history_windows: int = 336,
    batch_size: int = 64,
    per_horizon: bool = True,
    aci_eta: float = 0.01,
) -> dict[str, Any]:
    if not (0 < float(alpha) < 1):
        raise ValueError("alpha must be in (0, 1).")
    if calibration_windows < 2:
        raise ValueError("calibration_windows must be >= 2 for ACI.")
    if eval_windows < 1 or history_windows < 1:
        raise ValueError("eval_windows and history_windows must be >= 1.")
    if not (0 < float(aci_eta) <= 1):
        raise ValueError("aci_eta must be in (0, 1].")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe must have a DatetimeIndex.")

    missing = [col for col in artifact.feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:10]}")
    if artifact.target_col not in df.columns:
        raise ValueError(f"Missing target column '{artifact.target_col}' for backtest.")

    df_sorted = df.sort_index()
    x_all = df_sorted[artifact.feature_cols].to_numpy(dtype=np.float32)
    y_all = df_sorted[artifact.target_col].to_numpy(dtype=np.float32)
    max_anchor = len(df_sorted) - artifact.horizon
    minimum_anchor = max(artifact.lookback, 168)
    anchors = np.arange(minimum_anchor, max_anchor + 1, dtype=int)
    if anchors.size < 3:
        raise ValueError("Not enough rows for backtest evaluation.")

    eval_count = int(min(eval_windows, anchors.size))
    anchors_eval = anchors[-eval_count:]
    first_eval_anchor = int(anchors_eval[0])

    # Only residuals whose complete horizon was observable at the first issue
    # time initialize ACI. More recent forecasts are queued and incorporated
    # once their final target has been observed.
    anchors_cal = anchors[anchors + artifact.horizon <= first_eval_anchor]
    anchors_cal = anchors_cal[-calibration_windows:]
    if anchors_cal.size < 2:
        raise ValueError("Not enough causal calibration windows before evaluation.")
    last_cal_anchor = int(anchors_cal[-1])
    anchors_pending = anchors[
        (anchors > last_cal_anchor) & (anchors < first_eval_anchor)
    ]

    history_count = int(min(history_windows, anchors.size))
    anchors_history = anchors[-history_count:]
    anchors_needed = np.unique(
        np.concatenate([anchors_cal, anchors_pending, anchors_eval, anchors_history])
    )
    predictions_needed = _predict_anchors(x_all, anchors_needed, artifact, batch_size)
    prediction_by_anchor = {
        int(anchor): predictions_needed[row]
        for row, anchor in enumerate(anchors_needed)
    }

    y_cal_true = _true_horizons(y_all, anchors_cal, artifact.horizon)
    y_cal_pred = np.stack([prediction_by_anchor[int(anchor)] for anchor in anchors_cal])
    calibrator = _AdaptiveConformalCalibrator(
        horizon=artifact.horizon,
        alpha=alpha,
        eta=aci_eta,
        window_size=calibration_windows,
        per_horizon=per_horizon,
    )
    for residual_row in np.abs(y_cal_true - y_cal_pred):
        calibrator.update(residual_row)

    y_eval_true = _true_horizons(y_all, anchors_eval, artifact.horizon)
    y_eval_pred = np.stack([prediction_by_anchor[int(anchor)] for anchor in anchors_eval])
    y_eval_baseline = _previous_day_baseline(y_all, anchors_eval, artifact.horizon)

    update_queue = list(np.concatenate([anchors_pending, anchors_eval]))
    update_position = 0
    q_eval: list[np.ndarray] = []
    for anchor in anchors_eval:
        while (
            update_position < len(update_queue)
            and update_queue[update_position] + artifact.horizon <= anchor
        ):
            matured_anchor = int(update_queue[update_position])
            matured_true = y_all[matured_anchor : matured_anchor + artifact.horizon]
            matured_prediction = prediction_by_anchor[matured_anchor]
            calibrator.update(np.abs(matured_true - matured_prediction))
            update_position += 1
        q_eval.append(calibrator.widths())
    q_eval_array = np.stack(q_eval)

    lower_eval = y_eval_pred - q_eval_array
    upper_eval = y_eval_pred + q_eval_array
    coverage = float(np.mean((y_eval_true >= lower_eval) & (y_eval_true <= upper_eval)))
    mean_interval_width = float(np.mean(upper_eval - lower_eval))

    mae_model = float(np.mean(np.abs(y_eval_true - y_eval_pred)))
    mae_baseline = float(np.mean(np.abs(y_eval_true - y_eval_baseline)))
    rmse_model = float(np.sqrt(np.mean((y_eval_true - y_eval_pred) ** 2)))
    rmse_baseline = float(np.sqrt(np.mean((y_eval_true - y_eval_baseline) ** 2)))
    mae_improvement_abs = float(mae_baseline - mae_model)
    mae_improvement_pct = (
        float(100.0 * mae_improvement_abs / mae_baseline) if mae_baseline > 0 else float("nan")
    )
    y_history_true = _true_horizons(y_all, anchors_history, artifact.horizon)
    y_history_pred = np.stack([prediction_by_anchor[int(anchor)] for anchor in anchors_history])
    y_history_baseline = _previous_day_baseline(y_all, anchors_history, artifact.horizon)
    final_width = calibrator.widths()

    history_records: list[dict[str, Any]] = []
    history_index: list[pd.Timestamp] = []
    for row_idx, anchor in enumerate(anchors_history):
        horizon_index = df_sorted.index[int(anchor) : int(anchor) + artifact.horizon]
        for step_idx, timestamp in enumerate(horizon_index):
            history_records.append(
                {
                    "true_load_mw": float(y_history_true[row_idx, step_idx]),
                    "forecast_load_mw": float(y_history_pred[row_idx, step_idx]),
                    "baseline_load_mw": float(y_history_baseline[row_idx, step_idx]),
                    "lower_pi": float(y_history_pred[row_idx, step_idx] - final_width[step_idx]),
                    "upper_pi": float(y_history_pred[row_idx, step_idx] + final_width[step_idx]),
                    "anchor_timestamp": df_sorted.index[int(anchor)],
                }
            )
            history_index.append(timestamp)

    history_df = pd.DataFrame(history_records, index=pd.DatetimeIndex(history_index)).sort_index()
    latest_anchor = int(anchors_eval[-1])
    latest_index = df_sorted.index[latest_anchor : latest_anchor + artifact.horizon]
    latest_df = history_df.loc[latest_index].copy()

    return {
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "mae_improvement_abs": mae_improvement_abs,
        "mae_improvement_pct": mae_improvement_pct,
        "rmse_model": rmse_model,
        "rmse_baseline": rmse_baseline,
        "baseline_name": "previous_day",
        "coverage": coverage,
        "target_coverage": 1.0 - float(alpha),
        "coverage_gap": float(coverage - (1.0 - float(alpha))),
        "mean_interval_width": mean_interval_width,
        "n_eval_windows": eval_count,
        "n_calibration_windows": int(anchors_cal.size),
        "q_level": float(np.mean(calibrator.quantile_levels())),
        "aci_eta": float(aci_eta),
        "aci_alpha_t_mean": float(np.mean(calibrator.alpha_t)),
        "interval_method": "adaptive_conformal_inference",
        "latest_window_df": latest_df,
        "forecast_history_df": history_df,
    }
