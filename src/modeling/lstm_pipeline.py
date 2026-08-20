from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.forecast_config import DEFAULT_CONFIG, ForecastConfig
from src.modeling.evaluation import (
    prediction_metrics,
    rolling_aci_intervals,
    seasonal_baselines,
    tune_aci_on_validation,
)


@dataclass(slots=True)
class LSTMTrainConfig:
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.15
    patience: int = 5
    weight_decay: float = 1e-4
    model_out_path: Path = DEFAULT_CONFIG.model_path


class DailyLoadLSTM(nn.Module):
    """Encode load history with an LSTM, then combine it with known-at-origin context."""

    def __init__(
        self,
        context_size: int,
        horizon: int = 24,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + context_size, horizon)

    def forward(self, history: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(history)
        combined = torch.cat([encoded[:, -1, :], context], dim=1)
        return self.head(self.dropout(combined))


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_daily_samples(path: Path, config: ForecastConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Daily sample table not found: {path}")
    samples = pd.read_parquet(path)
    if not isinstance(samples.index, pd.DatetimeIndex):
        samples.index = pd.to_datetime(samples.index, utc=True)
    samples = samples.sort_index()
    required = {
        "split",
        *config.history_columns,
        *config.context_columns,
        *config.target_columns,
    }
    missing = sorted(required.difference(samples.columns))
    if missing:
        raise ValueError(f"Daily sample table is missing columns: {missing[:10]}")
    return samples


def _scale_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.where(scale == 0, 1.0, scale)


def _arrays(
    samples: pd.DataFrame,
    config: ForecastConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = samples[list(config.history_columns)].to_numpy(dtype=np.float32)
    context = samples[list(config.context_columns)].to_numpy(dtype=np.float32)
    target = samples[list(config.target_columns)].to_numpy(dtype=np.float32)
    return history, context, target


def _loader(
    history: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(history[:, :, None], dtype=torch.float32),
        torch.tensor(context, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    for history, context, target in loader:
        history = history.to(device)
        context = context.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(history, context)
        loss = loss_fn(prediction, target)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(history)
        total_rows += len(history)
    return total_loss / max(total_rows, 1)


def _predict(
    model: nn.Module,
    history: np.ndarray,
    context: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(history), batch_size):
            history_tensor = torch.tensor(
                history[start : start + batch_size, :, None], dtype=torch.float32, device=device
            )
            context_tensor = torch.tensor(
                context[start : start + batch_size], dtype=torch.float32, device=device
            )
            outputs.append(model(history_tensor, context_tensor).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _jsonable_forecast_config(config: ForecastConfig) -> dict[str, Any]:
    return {
        "timezone": config.timezone,
        "forecast_origin_hour_local": config.forecast_origin_hour_local,
        "horizon_hours": config.horizon_hours,
        "lookback_hours": config.lookback_hours,
        "random_seed": config.random_seed,
        "weather_model": config.weather_model,
        "weather_run_hour_utc": config.weather_run_hour_utc,
        "weather_availability_delay_hours": config.weather_availability_delay_hours,
        "weather_archive_hours": config.weather_archive_hours,
        "forecast_frequency": "hourly",
        "weather_variables": list(config.weather_variables),
        "train_end": config.train_end,
        "validation_end": config.validation_end,
        "final_test_end": config.final_test_end,
        "nominal_coverage": config.nominal_coverage,
    }


def run_training_pipeline(
    train_config: LSTMTrainConfig | None = None,
    forecast_config: ForecastConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    cfg = train_config or LSTMTrainConfig()
    set_deterministic_seed(forecast_config.random_seed)
    samples = load_daily_samples(forecast_config.features_path, forecast_config)
    train = samples[samples["split"] == "train"]
    validation = samples[samples["split"] == "validation"]
    final_test = samples[samples["split"] == "final_test"]
    if min(len(train), len(validation), len(final_test)) < 1:
        raise ValueError("Train, validation, and final-test splits must all contain hourly origins")

    h_train, c_train, y_train = _arrays(train, forecast_config)
    h_val, c_val, y_val = _arrays(validation, forecast_config)
    h_test, c_test, y_test = _arrays(final_test, forecast_config)

    history_mean, history_scale = _scale_fit(h_train.reshape(-1, 1))
    context_mean, context_scale = _scale_fit(c_train)
    target_mean, target_scale = _scale_fit(y_train.reshape(-1, 1))
    def scale_history(values: np.ndarray) -> np.ndarray:
        return (values - history_mean[0]) / history_scale[0]

    def scale_context(values: np.ndarray) -> np.ndarray:
        return (values - context_mean) / context_scale

    def scale_target(values: np.ndarray) -> np.ndarray:
        return (values - target_mean[0]) / target_scale[0]

    train_loader = _loader(
        scale_history(h_train),
        scale_context(c_train),
        scale_target(y_train),
        cfg.batch_size,
        True,
    )
    val_loader = _loader(
        scale_history(h_val),
        scale_context(c_val),
        scale_target(y_val),
        cfg.batch_size,
        False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DailyLoadLSTM(
        context_size=len(forecast_config.context_columns),
        horizon=forecast_config.horizon_hours,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.L1Loss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = _run_epoch(model, val_loader, loss_fn, device)
        history_rows.append({"epoch": epoch, "train_mae_scaled": train_loss, "val_mae_scaled": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)

    def predict_unscaled(history: np.ndarray, context: np.ndarray) -> np.ndarray:
        scaled = _predict(
            model,
            scale_history(history),
            scale_context(context),
            device,
            cfg.batch_size,
        )
        return scaled * target_scale[0] + target_mean[0]

    val_pred = predict_unscaled(h_val, c_val)
    test_pred = predict_unscaled(h_test, c_test)
    baselines = seasonal_baselines(final_test, forecast_config)
    chosen_aci, aci_tuning = tune_aci_on_validation(
        y_val,
        val_pred,
        validation.index,
        pd.DatetimeIndex(validation["target_end_utc"]),
        forecast_config,
    )
    aci_result = rolling_aci_intervals(
        y_val - val_pred,
        y_test,
        test_pred,
        1.0 - forecast_config.nominal_coverage,
        chosen_aci,
        calibration_target_ends=pd.DatetimeIndex(validation["target_end_utc"]),
        evaluation_origins=final_test.index,
        evaluation_target_ends=pd.DatetimeIndex(final_test["target_end_utc"]),
    )

    model_metrics = prediction_metrics(y_test, test_pred)
    baseline_metrics = {
        name: prediction_metrics(y_test, prediction) for name, prediction in baselines.items()
    }
    metrics: dict[str, Any] = {
        "forecast_contract": _jsonable_forecast_config(forecast_config),
        "data": {
            "train_origins": len(train),
            "validation_origins": len(validation),
            "final_test_origins": len(final_test),
            "train_start": str(train.index.min()),
            "train_end": str(train.index.max()),
            "validation_start": str(validation.index.min()),
            "validation_end": str(validation.index.max()),
            "final_test_start": str(final_test.index.min()),
            "final_test_end": str(final_test.index.max()),
        },
        "features": {
            "load_history_hours": forecast_config.lookback_hours,
            "context_feature_count": len(forecast_config.context_columns),
            "weather_feature_count": len(forecast_config.weather_columns),
            "context_columns": list(forecast_config.context_columns),
        },
        "lstm": model_metrics,
        "baselines": baseline_metrics,
        "aci": {
            "nominal_coverage": forecast_config.nominal_coverage,
            "selected_on": "validation",
            "config": asdict(chosen_aci),
            "overall_coverage": aci_result["overall_coverage"],
            "mean_interval_width": aci_result["mean_interval_width"],
            "coverage_by_lead": aci_result["coverage_by_lead"],
            "width_by_lead": aci_result["width_by_lead"],
            "validation_candidates": aci_tuning.to_dict(orient="records"),
        },
        "training": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
            "epochs_completed": len(history_rows),
            "best_validation_mae_scaled": best_val,
            "device": device,
        },
    }

    cfg.model_out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": 4,
            "model_state_dict": model.state_dict(),
            "history_columns": list(forecast_config.history_columns),
            "context_columns": list(forecast_config.context_columns),
            "target_columns": list(forecast_config.target_columns),
            "history_mean": history_mean,
            "history_scale": history_scale,
            "context_mean": context_mean,
            "context_scale": context_scale,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "forecast_config": _jsonable_forecast_config(forecast_config),
            "aci_config": asdict(chosen_aci),
        },
        cfg.model_out_path,
    )
    metrics_path = cfg.model_out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison = pd.DataFrame(
        [
            {"model": "LSTM", **{key: model_metrics[key] for key in ("overall_mae", "overall_rmse")}},
            *[
                {
                    "model": name.replace("_", " ").title(),
                    **{key: values[key] for key in ("overall_mae", "overall_rmse")},
                }
                for name, values in baseline_metrics.items()
            ],
        ]
    ).sort_values("overall_mae")
    return {
        "model_out_path": cfg.model_out_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "compare": comparison,
        "training_history": pd.DataFrame(history_rows),
    }
