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


def forecast_next_horizon_with_intervals(
    df: pd.DataFrame,
    artifact: LoadedLSTMArtifact,
    alpha: float = 0.1,
    calibration_windows: int = 240,
    batch_size: int = 64,
    per_horizon: bool = True,
) -> pd.DataFrame:
    if not (0 < float(alpha) < 1):
        raise ValueError("alpha must be in (0, 1).")
    if calibration_windows < 1:
        raise ValueError("calibration_windows must be >= 1.")

    point_forecast = forecast_next_horizon(df, artifact)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe must have a DatetimeIndex.")

    missing = [col for col in artifact.feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:10]}")
    if artifact.target_col not in df.columns:
        raise ValueError(f"Missing target column '{artifact.target_col}' for interval calibration.")

    df_sorted = df.sort_index()
    n_rows = len(df_sorted)
    max_anchor = n_rows - artifact.horizon
    anchors = np.arange(artifact.lookback, max_anchor + 1, dtype=int)

    if anchors.size < 1:
        raise ValueError(
            "Not enough rows for interval calibration. Need more history with observed targets."
        )

    if anchors.size > calibration_windows:
        anchors = anchors[-calibration_windows:]

    x_all = df_sorted[artifact.feature_cols].to_numpy(dtype=np.float32)
    y_all = df_sorted[artifact.target_col].to_numpy(dtype=np.float32)

    x_windows = np.stack([x_all[i - artifact.lookback : i] for i in anchors], axis=0)
    y_true = np.stack([y_all[i : i + artifact.horizon] for i in anchors], axis=0)

    x_scaled = (x_windows - artifact.x_mean) / artifact.x_scale
    y_pred_scaled = _predict_scaled_batch(
        model=artifact.model,
        x_batch=x_scaled,
        device=artifact.device,
        batch_size=batch_size,
    )
    y_pred = y_pred_scaled * artifact.y_scale + artifact.y_mean

    abs_residuals = np.abs(y_true - y_pred)
    n_cal = abs_residuals.shape[0]
    q_level = float(np.clip(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 0.0, 1.0))

    if per_horizon:
        q_hat = np.quantile(abs_residuals, q_level, axis=0, method="higher")
    else:
        q_scalar = float(np.quantile(abs_residuals.reshape(-1), q_level, method="higher"))
        q_hat = np.full(artifact.horizon, q_scalar, dtype=np.float32)

    lower = point_forecast.to_numpy(dtype=np.float32) - q_hat
    upper = point_forecast.to_numpy(dtype=np.float32) + q_hat

    return pd.DataFrame(
        {
            "forecast_load_mw": point_forecast.to_numpy(dtype=np.float32),
            "lower_pi": lower,
            "upper_pi": upper,
            "interval_width": upper - lower,
            "q_hat": q_hat,
            "alpha": np.full(artifact.horizon, alpha, dtype=np.float32),
            "calibration_windows_used": np.full(artifact.horizon, n_cal, dtype=np.int32),
            "quantile_level": np.full(artifact.horizon, q_level, dtype=np.float32),
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
) -> dict[str, Any]:
    if not (0 < float(alpha) < 1):
        raise ValueError("alpha must be in (0, 1).")
    if calibration_windows < 1:
        raise ValueError("calibration_windows must be >= 1.")
    if eval_windows < 1:
        raise ValueError("eval_windows must be >= 1.")
    if history_windows < 1:
        raise ValueError("history_windows must be >= 1.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe must have a DatetimeIndex.")

    missing = [col for col in artifact.feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:10]}")
    if artifact.target_col not in df.columns:
        raise ValueError(f"Missing target column '{artifact.target_col}' for backtest.")

    df_sorted = df.sort_index()
    n_rows = len(df_sorted)
    max_anchor = n_rows - artifact.horizon
    anchors = np.arange(artifact.lookback, max_anchor + 1, dtype=int)
    if anchors.size < 2:
        raise ValueError("Not enough rows for backtest evaluation.")

    eval_count = int(min(eval_windows, anchors.size))
    anchors_eval = anchors[-eval_count:]

    first_eval_anchor = int(anchors_eval[0])
    anchors_cal = anchors[anchors < first_eval_anchor]
    if anchors_cal.size > calibration_windows:
        anchors_cal = anchors_cal[-calibration_windows:]

    x_all = df_sorted[artifact.feature_cols].to_numpy(dtype=np.float32)
    y_all = df_sorted[artifact.target_col].to_numpy(dtype=np.float32)

    x_eval = np.stack([x_all[i - artifact.lookback : i] for i in anchors_eval], axis=0)
    y_eval_true = np.stack([y_all[i : i + artifact.horizon] for i in anchors_eval], axis=0)

    x_eval_scaled = (x_eval - artifact.x_mean) / artifact.x_scale
    y_eval_pred_scaled = _predict_scaled_batch(
        model=artifact.model,
        x_batch=x_eval_scaled,
        device=artifact.device,
        batch_size=batch_size,
    )
    y_eval_pred = y_eval_pred_scaled * artifact.y_scale + artifact.y_mean

    y_eval_baseline = np.stack(
        [np.full(artifact.horizon, y_all[i - 1], dtype=np.float32) for i in anchors_eval],
        axis=0,
    )

    mae_model = float(np.mean(np.abs(y_eval_true - y_eval_pred)))
    mae_baseline = float(np.mean(np.abs(y_eval_true - y_eval_baseline)))
    rmse_model = float(np.sqrt(np.mean((y_eval_true - y_eval_pred) ** 2)))
    rmse_baseline = float(np.sqrt(np.mean((y_eval_true - y_eval_baseline) ** 2)))
    mae_improvement_abs = float(mae_baseline - mae_model)
    mae_improvement_pct = (
        float(100.0 * mae_improvement_abs / mae_baseline) if mae_baseline > 0 else float("nan")
    )

    q_level = float("nan")
    q_hat = np.zeros(artifact.horizon, dtype=np.float32)
    coverage = float("nan")
    mean_interval_width = float("nan")
    n_cal = int(anchors_cal.size)

    if n_cal > 0:
        x_cal = np.stack([x_all[i - artifact.lookback : i] for i in anchors_cal], axis=0)
        y_cal_true = np.stack([y_all[i : i + artifact.horizon] for i in anchors_cal], axis=0)
        x_cal_scaled = (x_cal - artifact.x_mean) / artifact.x_scale
        y_cal_pred_scaled = _predict_scaled_batch(
            model=artifact.model,
            x_batch=x_cal_scaled,
            device=artifact.device,
            batch_size=batch_size,
        )
        y_cal_pred = y_cal_pred_scaled * artifact.y_scale + artifact.y_mean

        abs_residuals = np.abs(y_cal_true - y_cal_pred)
        q_level = float(np.clip(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 0.0, 1.0))

        if per_horizon:
            q_hat = np.quantile(abs_residuals, q_level, axis=0, method="higher")
        else:
            q_scalar = float(np.quantile(abs_residuals.reshape(-1), q_level, method="higher"))
            q_hat = np.full(artifact.horizon, q_scalar, dtype=np.float32)

        lower_eval = y_eval_pred - q_hat
        upper_eval = y_eval_pred + q_hat
        in_interval = (y_eval_true >= lower_eval) & (y_eval_true <= upper_eval)
        coverage = float(np.mean(in_interval))
        mean_interval_width = float(np.mean(upper_eval - lower_eval))

    history_count = int(min(history_windows, anchors.size))
    anchors_history = anchors[-history_count:]

    x_history = np.stack([x_all[i - artifact.lookback : i] for i in anchors_history], axis=0)
    y_history_true = np.stack([y_all[i : i + artifact.horizon] for i in anchors_history], axis=0)
    x_history_scaled = (x_history - artifact.x_mean) / artifact.x_scale
    y_history_pred_scaled = _predict_scaled_batch(
        model=artifact.model,
        x_batch=x_history_scaled,
        device=artifact.device,
        batch_size=batch_size,
    )
    y_history_pred = y_history_pred_scaled * artifact.y_scale + artifact.y_mean
    y_history_baseline = np.stack(
        [np.full(artifact.horizon, y_all[i - 1], dtype=np.float32) for i in anchors_history],
        axis=0,
    )

    history_records: list[dict[str, Any]] = []
    history_index: list[pd.Timestamp] = []
    for row_idx, anchor in enumerate(anchors_history):
        horizon_index = df_sorted.index[int(anchor) : int(anchor) + artifact.horizon]
        pred_row = y_history_pred[row_idx]
        true_row = y_history_true[row_idx]
        baseline_row = y_history_baseline[row_idx]

        for step_idx, ts in enumerate(horizon_index):
            history_records.append(
                {
                    "true_load_mw": float(true_row[step_idx]),
                    "forecast_load_mw": float(pred_row[step_idx]),
                    "naive_baseline_load_mw": float(baseline_row[step_idx]),
                    "lower_pi": float(pred_row[step_idx] - q_hat[step_idx]) if n_cal > 0 else np.nan,
                    "upper_pi": float(pred_row[step_idx] + q_hat[step_idx]) if n_cal > 0 else np.nan,
                    "anchor_timestamp": df_sorted.index[int(anchor)],
                }
            )
            history_index.append(ts)

    history_df = pd.DataFrame(history_records, index=pd.DatetimeIndex(history_index))
    history_df = history_df.sort_index()

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
        "coverage": coverage,
        "target_coverage": 1.0 - float(alpha),
        "coverage_gap": float(coverage - (1.0 - float(alpha))) if n_cal > 0 else float("nan"),
        "mean_interval_width": mean_interval_width,
        "n_eval_windows": eval_count,
        "n_calibration_windows": n_cal,
        "q_level": q_level,
        "latest_window_df": latest_df,
        "forecast_history_df": history_df,
    }
