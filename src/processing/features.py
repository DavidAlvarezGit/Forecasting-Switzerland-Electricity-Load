from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if __package__ in (None, ""):
    from helpers import find_project_root
else:
    from .helpers import find_project_root


TARGET_COL = "load_load_mw"
AGGREGATED_DATASET_REL_PATH = Path("data") / "interim" / "aggregated.parquet"
FEATURE_DATASET_REL_PATH = Path("data") / "processed" / "features.parquet"


def _as_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    return df.index

def _load_aggregated(project_root: Path, input_path: str | Path | None = None) -> pd.DataFrame:
    resolved_input = Path(input_path) if input_path is not None else project_root / AGGREGATED_DATASET_REL_PATH
    if not resolved_input.exists():
        raise FileNotFoundError(f"Aggregated data not found: {resolved_input}")

    df = pd.read_parquet(resolved_input)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Aggregated data must have a DatetimeIndex")

    return df.sort_index()


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = _as_datetime_index(out)

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["dayofyear"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return out


def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    return out


def _add_load_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def _add_load_rolling_features(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    shifted = out[target_col].shift(1)

    for window in windows:
        roll = shifted.rolling(window=window, min_periods=window)
        out[f"{target_col}_roll_mean_{window}"] = roll.mean()
        out[f"{target_col}_roll_std_{window}"] = roll.std()
    return out


def build_feature_table(
    aggregated_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    if target_col not in aggregated_df.columns:
        raise ValueError(f"Target column not found in aggregated data: {target_col}")

    lag_values = lags or [1, 2, 3, 4, 5, 6, 7, 12, 24, 36, 48, 72, 100, 150, 168]
    window_values = rolling_windows or [24, 168]

    features = aggregated_df.copy()
    features = _add_calendar_features(features)
    features = _add_cyclical_features(features)
    features = _add_load_lag_features(features, target_col=target_col, lags=lag_values)
    features = _add_load_rolling_features(features, target_col=target_col, windows=window_values)

    features = features.dropna().sort_index()
    return features


def run_feature_pipeline(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    project_root = find_project_root(Path.cwd().resolve())
    aggregated_df = _load_aggregated(project_root, input_path)
    feature_df = build_feature_table(aggregated_df, target_col=target_col)

    resolved_output = Path(output_path) if output_path is not None else project_root / FEATURE_DATASET_REL_PATH
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(resolved_output)

    meta: dict[str, int | str] = {
        "rows": int(len(feature_df)),
        "columns": int(feature_df.shape[1]),
        "start": str(feature_df.index.min()),
        "end": str(feature_df.index.max()),
        "output_path": str(resolved_output),
    }
    return feature_df, meta


def main() -> None:
    feature_df, meta = run_feature_pipeline()
    print(
        f"Feature dataset: {meta['start']} -> {meta['end']} | "
        f"rows={meta['rows']} cols={meta['columns']}"
    )
    print(f"Feature output: {meta['output_path']}")
    print(feature_df.head(5))


if __name__ == "__main__":
    main()
