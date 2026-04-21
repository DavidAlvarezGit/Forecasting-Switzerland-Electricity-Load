from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import inference as inference_mod

forecast_next_horizon = inference_mod.forecast_next_horizon
load_lstm_artifact = inference_mod.load_lstm_artifact
forecast_next_horizon_with_intervals = getattr(
    inference_mod,
    "forecast_next_horizon_with_intervals",
    None,
)
evaluate_recent_backtest = getattr(
    inference_mod,
    "evaluate_recent_backtest",
    None,
)


@st.cache_resource(show_spinner=False)
def _load_artifact(model_path: str):
    return load_lstm_artifact(model_path)


@st.cache_data(show_spinner=False)
def _load_features(features_path: str) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def main() -> None:
    st.set_page_config(page_title="Load Forecast", layout="wide")
    st.title("Electricity Demand Forecast Model")
    st.caption("Most Recent Model Predictions")

    default_model_path = str(Path("data") / "processed" / "models" / "best_lstm_24h.pt")
    default_features_path = str(Path("data") / "processed" / "lstm_features.parquet")

    with st.sidebar:
        st.header("Inputs")
        model_path = st.text_input("Model data", value=default_model_path)
        features_path = st.text_input("Predictors", value=default_features_path)
        st.header("Intervals")
        show_intervals = st.checkbox("Show prediction intervals", value=True)
        confidence = st.slider("Confidence level", min_value=0.70, max_value=0.99, value=0.90, step=0.01)
        st.header("Performance")
        compute_backtest = st.checkbox("Compute backtest metrics", value=False)
        include_forecast_history = st.checkbox("Include historical forecast line", value=False)
        calibration_windows = st.number_input(
            "Calibration windows",
            min_value=24,
            max_value=1000,
            value=96,
            step=24,
        )
        eval_windows = st.number_input(
            "Backtest windows",
            min_value=24,
            max_value=1000,
            value=120,
            step=24,
        )
        past_points = st.number_input(
            "Past points in first plot",
            min_value=24,
            max_value=2000,
            value=336,
            step=24,
        )
        history_points = st.number_input(
            "Forecast history points",
            min_value=24,
            max_value=100000,
            value=336,
            step=24,
        )
        per_horizon = st.checkbox("Per-horizon interval width", value=True)

    if not Path(model_path).exists():
        st.warning("Model data not found. Train and save a model first.")
        st.stop()
    if not Path(features_path).exists():
        st.warning("Predictors not found. Build features first.")
        st.stop()

    try:
        artifact = _load_artifact(model_path)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    try:
        df = _load_features(features_path)
    except Exception as exc:
        st.error(f"Failed to load features: {exc}")
        st.stop()

    needed_cols = [c for c in artifact.feature_cols if c in df.columns]
    if artifact.target_col in df.columns:
        needed_cols.append(artifact.target_col)
    if needed_cols:
        # Trim the working dataframe to the columns required for inference/backtest.
        df = df.loc[:, pd.Index(needed_cols).unique()]

    st.write(f"Feature table rows: {len(df)} | columns: {df.shape[1]}")

    if len(df) < artifact.lookback:
        st.error(
            f"Not enough rows for forecasting. Need at least {artifact.lookback}, found {len(df)}."
        )
        st.stop()

    alpha = 1.0 - float(confidence)
    backtest: dict[str, Any] | None = None

    if compute_backtest and evaluate_recent_backtest is not None:
        try:
            backtest = evaluate_recent_backtest(
                df,
                artifact,
                alpha=alpha,
                calibration_windows=int(calibration_windows),
                eval_windows=int(eval_windows),
                per_horizon=per_horizon,
                include_history=include_forecast_history,
                history_windows=int(history_points),
            )
        except Exception as exc:
            st.warning(
                "Backtest metrics could not be computed. "
                f"Reason: {exc}"
            )

    st.subheader("Model Error vs Baseline")
    if backtest is not None:
        mae_model = float(backtest["mae_model"])
        mae_baseline = float(backtest["mae_baseline"])
        mae_improvement_abs = float(backtest["mae_improvement_abs"])
        mae_improvement_pct = float(backtest["mae_improvement_pct"])
        coverage = float(backtest["coverage"])
        target_coverage = float(backtest["target_coverage"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model MAE", f"{mae_model:.2f}")
        col2.metric("Naive MAE", f"{mae_baseline:.2f}")
        col3.metric(
            "MAE improvement",
            f"{mae_improvement_abs:.2f}",
            delta=f"{mae_improvement_pct:.1f}% vs baseline",
        )

        if pd.notna(coverage):
            col4.metric(
                "PI coverage",
                f"{coverage:.1%}",
                delta=f"target {target_coverage:.1%}",
            )
        else:
            col4.metric("PI coverage", "N/A")

        st.caption(
            f"mean PI width={float(backtest['mean_interval_width']):.2f}, "
            f"evaluation windows={int(backtest['n_eval_windows'])}, "
            f"calibration windows={int(backtest['n_calibration_windows'])}."
        )
    else:
        if compute_backtest:
            st.info("Backtest metrics unavailable in this runtime.")
        else:
            st.info("Backtest metrics are disabled to reduce CPU and memory usage.")

    if show_intervals and forecast_next_horizon_with_intervals is not None:
        try:
            forecast_df = forecast_next_horizon_with_intervals(
                df,
                artifact,
                alpha=alpha,
                calibration_windows=int(calibration_windows),
                per_horizon=per_horizon,
            )
            forecast = forecast_df["forecast_load_mw"]
        except Exception as exc:
            st.warning(
                "Prediction-interval computation failed; showing point forecast only. "
                f"Reason: {exc}"
            )
            try:
                forecast = forecast_next_horizon(df, artifact)
            except Exception as forecast_exc:
                st.error(f"Forecast failed: {forecast_exc}")
                st.stop()
            forecast_df = forecast.to_frame(name="forecast_load_mw")
    elif show_intervals and forecast_next_horizon_with_intervals is None:
        st.warning(
            "Interval function is not available in the currently loaded module. "
            "Restart Streamlit to pick up the latest code; showing point forecast for now."
        )
        try:
            forecast = forecast_next_horizon(df, artifact)
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
            st.stop()
        forecast_df = forecast.to_frame(name="forecast_load_mw")
    else:
        try:
            forecast = forecast_next_horizon(df, artifact)
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
            st.stop()
        forecast_df = forecast.to_frame(name="forecast_load_mw")

    st.subheader("Forecast")
    horizon_to_show = st.slider("Steps to display", min_value=1, max_value=artifact.horizon, value=min(24, artifact.horizon))
    forecast_show = forecast.iloc[:horizon_to_show]
    forecast_df_show = forecast_df.iloc[:horizon_to_show].copy()
    forecast_before = pd.Series(dtype=float)
    if backtest is not None:
        forecast_history_df = backtest.get("forecast_history_df")
        if isinstance(forecast_history_df, pd.DataFrame) and "forecast_load_mw" in forecast_history_df.columns:
            forecast_history_df = forecast_history_df.sort_index()
            forecast_before = forecast_history_df["forecast_load_mw"].groupby(level=0).last().sort_index()
            if len(forecast_before) > int(history_points):
                forecast_before = forecast_before.tail(int(history_points))
        elif include_forecast_history:
            latest_window_df = backtest.get("latest_window_df")
            if isinstance(latest_window_df, pd.DataFrame) and "forecast_load_mw" in latest_window_df.columns:
                forecast_before = latest_window_df["forecast_load_mw"].copy()

    context = pd.Series(dtype=float)
    real_load_on_horizon = pd.Series(dtype=float)
    if artifact.target_col in df.columns:
        context = df[artifact.target_col].dropna().tail(int(past_points))
        real_load_on_horizon = df[artifact.target_col].reindex(forecast_df_show.index)

    if real_load_on_horizon.notna().any():
        st.caption("Real load is overlaid on forecast timestamps where observed values are available.")

    has_intervals = {"lower_pi", "upper_pi"}.issubset(forecast_df_show.columns)
    chart_index = forecast_df_show.index
    if len(context) > 0:
        chart_index = context.index.union(chart_index)

    chart_df = pd.DataFrame(index=chart_index)
    chart_df["actual_recent"] = context.reindex(chart_index)
    chart_df["forecast_load_mw"] = forecast_show.reindex(chart_index)
    chart_df["forecast_before_mw"] = forecast_before.reindex(chart_index)
    chart_df["real_load_mw"] = real_load_on_horizon.reindex(chart_index)

    if has_intervals:
        st.caption(
            "Prediction interval shown as lower/upper lines. "
            f"Confidence={confidence:.0%}, calibration_windows={int(forecast_df_show['calibration_windows_used'].iloc[0])}."
        )
        chart_df["lower_pi"] = forecast_df_show["lower_pi"].reindex(chart_index)
        chart_df["upper_pi"] = forecast_df_show["upper_pi"].reindex(chart_index)

    st.line_chart(chart_df)


if __name__ == "__main__":
    main()
