from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TARGET_COL = "load_load_mw"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = DATA_DIR / "lstm_features.parquet"
MODEL_DIR = DATA_DIR / "models"


def _split_conformal_quantile(scores: list[float] | np.ndarray, alpha: float) -> tuple[float, float]:
    score_arr = np.asarray(scores, dtype=float)
    if score_arr.ndim != 1:
        raise ValueError("scores must be a 1D array-like.")
    if score_arr.size < 1:
        raise ValueError("Need at least one score.")

    n = score_arr.size
    q_level = np.ceil((n + 1) * (1.0 - float(alpha))) / n
    q_level = float(np.clip(q_level, 0.0, 1.0))

    q_hat = np.quantile(score_arr, q_level, method="higher")

    return float(q_hat), q_level


@dataclass(slots=True)
class LSTMTrainConfig:
    target_col: str = TARGET_COL
    features_path: Path = FEATURES_PATH
    lookback: int = 168 * 2
    horizon: int = 24
    batch_size: int = 256
    infer_batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 1e-3
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    top_k_features: int = 5
    corr_threshold: float = 0.95
    hidden_size: int = 128
    num_layers: int = 5
    dropout: float = 0.15
    patience: int = 5
    weight_decay: float = 1e-4
    model_out_path: Path = MODEL_DIR / "best_lstm_24h.pt"


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_horizon: int,
        hidden_size: int = 128,
        num_layers: int = 5,
        dropout: float = 0.15,
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


class ACI(BaseEstimator, RegressorMixin):
    def __init__(self, base_model: Any, alpha: float = 0.1, eta: float = 0.01, window_size: int | None = None):
        self.base_model = base_model
        self.alpha = alpha
        self.eta = eta
        self.window_size = window_size

    def _validate_params(self) -> None:
        if not (0 < float(self.alpha) < 1):
            raise ValueError("alpha must be in (0, 1)")
        if not (0 < float(self.eta) <= 1):
            raise ValueError("eta must be in (0, 1]")
        if self.window_size is not None:
            if self.window_size < 1:
                raise ValueError("window_size must be None or an integer >= 1")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ACI":
        self._validate_params()
        x_checked, y_checked = check_X_y(X, y)

        self.model_ = clone(self.base_model)
        self.model_.fit(x_checked, y_checked)

        scores = np.abs(y_checked - self.model_.predict(x_checked))
        scores = np.asarray(scores, dtype=float)

        self.window_size_ = len(scores) if self.window_size is None else min(self.window_size, len(scores))
        if self.window_size_ < 1:
            raise ValueError("Need at least one calibration score.")

        self.score_buffer_ = list(scores[-self.window_size_ :])
        self.alpha_t_ = float(self.alpha)
        self.q_hat_ = _split_conformal_quantile(self.score_buffer_, self.alpha_t_)[0]
        return self

    def calibrate(self, X: np.ndarray | None = None, y: np.ndarray | None = None, reset_alpha: bool = True) -> "ACI":
        check_is_fitted(self, ["model_"])
        if X is None and y is None:
            return self

        if X is None or y is None:
            raise ValueError("X and y must both be provided when calibrating.")

        x_checked, y_checked = check_X_y(X, y)
        scores = np.abs(y_checked - self.model_.predict(x_checked))
        scores = np.asarray(scores, dtype=float)

        self.window_size_ = len(scores) if self.window_size is None else min(self.window_size, len(scores))
        if self.window_size_ < 1:
            raise ValueError("Need at least one calibration score.")

        self.score_buffer_ = list(scores[-self.window_size_ :])

        if reset_alpha or not hasattr(self, "alpha_t_"):
            self.alpha_t_ = float(self.alpha)

        self.q_hat_ = _split_conformal_quantile(self.score_buffer_, self.alpha_t_)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        x_checked = check_array(X)
        return self.model_.predict(x_checked)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["model_", "q_hat_"])
        yhat = self.predict(X)
        return yhat - self.q_hat_, yhat + self.q_hat_

    def predict_interval_sequential(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["model_", "q_hat_", "score_buffer_", "alpha_t_"])
        x_checked = check_array(X)
        y_checked = np.asarray(y, dtype=float)

        if x_checked.shape[0] != y_checked.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        n = x_checked.shape[0]
        lower = np.zeros(n, dtype=float)
        upper = np.zeros(n, dtype=float)

        alpha_t = float(self.alpha_t_)
        score_buffer = list(self.score_buffer_)

        for t in range(n):
            q_hat_t = _split_conformal_quantile(score_buffer, alpha_t)[0]

            yhat_t = float(self.model_.predict(x_checked[t].reshape(1, -1))[0])
            lower[t] = yhat_t - q_hat_t
            upper[t] = yhat_t + q_hat_t

            score_t = abs(y_checked[t] - yhat_t)
            err_t = int(score_t > q_hat_t)

            alpha_t = float(np.clip(alpha_t + self.eta * (self.alpha - err_t), 1e-6, 1 - 1e-6))

            score_buffer.append(float(score_t))
            if len(score_buffer) > self.window_size_:
                score_buffer.pop(0)

        self.alpha_t_ = alpha_t
        self.score_buffer_ = score_buffer
        self.q_hat_ = _split_conformal_quantile(self.score_buffer_, self.alpha_t_)[0]
        return lower, upper


def load_lstm_dataset(input_path: Path, target_col: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    df = pd.read_parquet(input_path)
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    if df.index.has_duplicates:
        raise ValueError("Datetime index contains duplicates.")

    return df.sort_index()


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    return {"mae": mae, "rmse": rmse}


def naive_last_value_multi_horizon(
    target: pd.Series,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    idx: list[pd.Timestamp] = []
    arr = target.to_numpy()

    for i in range(lookback, len(arr) - horizon + 1):
        last_value = arr[i - 1]
        y_true.append(arr[i : i + horizon])
        y_pred.append(np.full(horizon, last_value, dtype=np.float32))
        idx.append(target.index[i])

    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32), pd.DatetimeIndex(idx)


def select_features_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    top_k: int = 5,
    corr_threshold: float = 0.95,
) -> tuple[list[str], pd.DataFrame, dict[str, float | int]]:
    base_cols = [c for c in train_df.columns if c != target_col]

    nunique = train_df[base_cols].nunique(dropna=False)
    cols_var = nunique[nunique > 1].index.tolist()

    corr = train_df[cols_var].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [col for col in upper.columns if (upper[col] > corr_threshold).any()]
    cols_filtered = [c for c in cols_var if c not in drop_corr]

    x_train = train_df[cols_filtered]
    y_train = train_df[target_col]
    x_val = val_df[cols_filtered]
    y_val = val_df[target_col]

    model_full = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=64,
        random_state=42,
    )
    model_full.fit(x_train, y_train)
    pred_full = np.asarray(model_full.predict(x_val), dtype=np.float32)
    mae_full = mean_absolute_error(y_val, pred_full)

    importance_df = pd.DataFrame(
        {
            "feature": x_train.columns,
            "gain": model_full.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)

    selected = importance_df[importance_df["gain"] > 0]["feature"].head(top_k).tolist()
    if len(selected) == 0:
        selected = importance_df["feature"].head(min(top_k, len(importance_df))).tolist()

    model_sel = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=64,
        random_state=42,
    )
    model_sel.fit(x_train[selected], y_train)
    pred_sel = np.asarray(model_sel.predict(x_val[selected]), dtype=np.float32)
    mae_sel = mean_absolute_error(y_val, pred_sel)

    summary: dict[str, float | int] = {
        "n_base": len(base_cols),
        "n_after_variance_corr": len(cols_filtered),
        "n_selected": len(selected),
        "val_mae_full_filtered": float(mae_full),
        "val_mae_selected": float(mae_sel),
    }
    return selected, importance_df, summary


def make_sequences(
    x: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_seq: list[np.ndarray] = []
    y_seq: list[np.ndarray] = []
    for i in range(lookback, len(x) - horizon + 1):
        x_seq.append(x[i - lookback : i])
        y_seq.append(y[i : i + horizon])
    return np.asarray(x_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def build_data_loaders(
    x_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    x_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train_seq), torch.tensor(y_train_seq)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val_seq), torch.tensor(y_val_seq)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses: list[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        with torch.set_grad_enabled(is_train):
            pred = model(xb)
            loss = criterion(pred, yb)

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses))


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    patience: int,
) -> pd.DataFrame:
    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion=criterion, optimizer=optimizer, device=device)
        val_loss = run_epoch(model, val_loader, criterion=criterion, optimizer=None, device=device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return pd.DataFrame(history)


def predict_batched(
    model: nn.Module,
    x_test_seq: np.ndarray,
    device: str,
    infer_batch_size: int,
) -> np.ndarray:
    model.eval()
    pred_batches: list[np.ndarray] = []
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test_seq, dtype=torch.float32)),
        batch_size=infer_batch_size,
        shuffle=False,
        drop_last=False,
    )

    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device, non_blocking=True)

            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(xb)
            else:
                pred = model(xb)

            pred_batches.append(pred.float().cpu().numpy())

    if device == "cuda":
        torch.cuda.empty_cache()

    return np.concatenate(pred_batches, axis=0)


def save_model_artifacts(
    model: nn.Module,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    feature_cols: list[str],
    config: LSTMTrainConfig,
) -> Path:
    config.model_out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "lookback": config.lookback,
            "horizon": config.horizon,
            "target_col": config.target_col,
            "x_scaler_mean": x_scaler.mean_,
            "x_scaler_scale": x_scaler.scale_,
            "y_scaler_mean": y_scaler.mean_,
            "y_scaler_scale": y_scaler.scale_,
        },
        config.model_out_path,
    )
    return config.model_out_path


def run_training_pipeline(config: LSTMTrainConfig | None = None) -> dict[str, Any]:
    cfg = config or LSTMTrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    df = load_lstm_dataset(cfg.features_path, target_col=cfg.target_col)
    train_df, val_df, test_df = split_dataset(df, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio)

    y_test_true_naive, y_test_pred_naive, _ = naive_last_value_multi_horizon(
        test_df[cfg.target_col],
        lookback=cfg.lookback,
        horizon=cfg.horizon,
    )
    naive_metrics = evaluate_metrics(y_test_true_naive, y_test_pred_naive)

    selected_feature_cols, importance_df, fs_summary = select_features_lgbm(
        train_df=train_df,
        val_df=val_df,
        target_col=cfg.target_col,
        top_k=cfg.top_k_features,
        corr_threshold=cfg.corr_threshold,
    )

    feature_cols = list(selected_feature_cols)
    if cfg.target_col not in feature_cols:
        feature_cols = [cfg.target_col] + feature_cols

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(train_df[feature_cols])
    x_val = x_scaler.transform(val_df[feature_cols])
    x_test = x_scaler.transform(test_df[feature_cols])

    y_train = y_scaler.fit_transform(train_df[[cfg.target_col]]).reshape(-1)
    y_val = y_scaler.transform(val_df[[cfg.target_col]]).reshape(-1)
    y_test = y_scaler.transform(test_df[[cfg.target_col]]).reshape(-1)

    x_train_seq, y_train_seq = make_sequences(x_train, y_train, cfg.lookback, cfg.horizon)
    x_val_seq, y_val_seq = make_sequences(x_val, y_val, cfg.lookback, cfg.horizon)
    x_test_seq, y_test_seq = make_sequences(x_test, y_test, cfg.lookback, cfg.horizon)

    train_loader, val_loader = build_data_loaders(
        x_train_seq=x_train_seq,
        y_train_seq=y_train_seq,
        x_val_seq=x_val_seq,
        y_val_seq=y_val_seq,
        batch_size=cfg.batch_size,
    )

    model = LSTMRegressor(
        input_size=len(feature_cols),
        output_horizon=cfg.horizon,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.L1Loss()

    history_df = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=cfg.epochs,
        patience=cfg.patience,
    )

    y_test_pred_scaled = predict_batched(
        model=model,
        x_test_seq=x_test_seq,
        device=device,
        infer_batch_size=cfg.infer_batch_size,
    )

    y_test_true = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(y_test_seq.shape)
    y_test_pred_lstm = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).reshape(y_test_pred_scaled.shape)

    lstm_metrics = evaluate_metrics(y_test_true, y_test_pred_lstm)

    compare_df = pd.DataFrame(
        [
            {"model": "naive_last_value_24h", **naive_metrics},
            {"model": "lstm_24h", **lstm_metrics},
        ]
    ).sort_values("mae")

    model_out_path = save_model_artifacts(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_cols=feature_cols,
        config=cfg,
    )

    metrics_path = model_out_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
                "feature_selection": fs_summary,
                "naive_metrics": naive_metrics,
                "lstm_metrics": lstm_metrics,
            },
            fp,
            indent=2,
        )

    return {
        "config": cfg,
        "feature_cols": feature_cols,
        "feature_importance": importance_df,
        "feature_selection_summary": fs_summary,
        "history": history_df,
        "compare": compare_df,
        "model_out_path": model_out_path,
        "metrics_path": metrics_path,
    }
