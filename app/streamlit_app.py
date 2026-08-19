from __future__ import annotations

import math
import sys
from html import escape
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_core import DashboardResults, build_dashboard_results
from src.modeling import inference as inference_mod


DEFAULT_MODEL_PATH = "data/processed/models/best_lstm_24h.pt"
DEFAULT_FEATURES_PATH = "data/processed/lstm_features.parquet"
CALIBRATION_WINDOWS = 120
EVALUATION_WINDOWS = 120
HISTORY_WINDOWS = 336

SWISS_RED = "#D52B1E"
INK = "#18212B"
BLUE = "#1769AA"
SLATE = "#687684"


@st.cache_resource(show_spinner=False)
def _load_artifact(model_path: str, modified_ns: int):
    del modified_ns
    return inference_mod.load_lstm_artifact(model_path)


@st.cache_data(show_spinner=False)
def _load_features(features_path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    df = pd.read_parquet(features_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


@st.cache_data(show_spinner=False, ttl=3600)
def _compute_results(
    model_path: str,
    model_modified_ns: int,
    features_path: str,
    features_modified_ns: int,
    confidence: float,
    per_horizon: bool,
    include_intervals: bool,
) -> DashboardResults:
    artifact = _load_artifact(model_path, model_modified_ns)
    df = _load_features(features_path, features_modified_ns)
    return build_dashboard_results(
        df,
        artifact,
        forecast_next_horizon=inference_mod.forecast_next_horizon,
        forecast_next_horizon_with_intervals=getattr(
            inference_mod, "forecast_next_horizon_with_intervals", None
        ),
        evaluate_recent_backtest=getattr(inference_mod, "evaluate_recent_backtest", None),
        confidence=confidence,
        calibration_windows=CALIBRATION_WINDOWS,
        eval_windows=EVALUATION_WINDOWS,
        history_windows=HISTORY_WINDOWS,
        per_horizon=per_horizon,
        include_intervals=include_intervals,
    )


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _format_mw(value: float | int | None, decimals: int = 0) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):,.{decimals}f} MW"


def _format_percent(value: float | int | None, decimals: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{decimals}%}"


def _format_timestamp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return pd.Timestamp(value).strftime("%d %b %Y · %H:%M")


def _baseline_label(name: Any) -> str:
    labels = {
        "persistence": "Persistence",
        "seasonal_24h": "Previous day",
        "seasonal_168h": "Previous week",
        "seasonal_24h_168h_blend": "Day/week seasonal blend",
    }
    return labels.get(str(name), str(name).replace("_", " ").title())


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            .block-container {{
                max-width: 1480px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }}
            .forecast-hero {{
                border: 1px solid rgba(24, 33, 43, 0.10);
                border-radius: 18px;
                padding: 1.5rem 1.65rem;
                margin-bottom: 1rem;
                background:
                    radial-gradient(circle at 94% 14%, rgba(213,43,30,.14), transparent 24%),
                    linear-gradient(135deg, #ffffff 0%, #fbfcfd 100%);
                box-shadow: 0 10px 32px rgba(24, 33, 43, 0.06);
            }}
            .forecast-eyebrow {{
                color: {SWISS_RED};
                font-size: .75rem;
                font-weight: 750;
                letter-spacing: .11em;
                text-transform: uppercase;
                margin-bottom: .35rem;
            }}
            .forecast-title {{
                color: {INK};
                font-size: clamp(2rem, 4vw, 3.2rem);
                font-weight: 760;
                letter-spacing: -.04em;
                line-height: 1.02;
                margin: 0;
            }}
            .forecast-subtitle {{
                color: {SLATE};
                font-size: 1rem;
                max-width: 780px;
                margin: .7rem 0 1.05rem 0;
            }}
            .forecast-chip {{
                display: inline-flex;
                align-items: center;
                gap: .35rem;
                border: 1px solid rgba(24, 33, 43, .10);
                border-radius: 999px;
                color: #44515E;
                background: rgba(255,255,255,.82);
                font-size: .78rem;
                font-weight: 620;
                margin-right: .45rem;
                margin-top: .25rem;
                padding: .35rem .65rem;
            }}
            div[data-testid="stMetric"] {{
                background: #ffffff;
                border: 1px solid rgba(24, 33, 43, 0.09);
                border-radius: 14px;
                padding: .9rem 1rem;
                box-shadow: 0 5px 18px rgba(24, 33, 43, 0.04);
            }}
            div[data-testid="stMetricLabel"] {{ color: {SLATE}; }}
            div[data-testid="stMetricValue"] {{ letter-spacing: -.03em; }}
            .section-kicker {{
                color: {SWISS_RED};
                font-weight: 700;
                font-size: .76rem;
                letter-spacing: .09em;
                text-transform: uppercase;
                margin-bottom: -.55rem;
            }}
            [data-testid="stSidebar"] {{ border-right: 1px solid rgba(24, 33, 43, .08); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(results: DashboardResults) -> None:
    metadata = results.metadata
    updated = escape(_format_timestamp(metadata["dataset_end"]))
    horizon = int(metadata["horizon"])
    confidence = float(metadata["confidence"])
    st.markdown(
        f"""
        <div class="forecast-hero">
            <div class="forecast-eyebrow">Swiss grid outlook</div>
            <h1 class="forecast-title">Electricity load forecast</h1>
            <p class="forecast-subtitle">
                See expected national demand, the likely forecast range, and how the
                model recently compared with simple forecasts based on past demand.
            </p>
            <span class="forecast-chip">● Data through {updated}</span>
            <span class="forecast-chip">Forecasts {horizon} hours ahead</span>
            <span class="forecast-chip">{confidence:.0%} confidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_metrics(results: DashboardResults) -> None:
    forecast = results.forecast["forecast_load_mw"].dropna()
    actual = results.recent_actual.dropna()
    latest_actual = float(actual.iloc[-1]) if not actual.empty else None
    first_forecast = float(forecast.iloc[0]) if not forecast.empty else None
    peak = float(forecast.max()) if not forecast.empty else None
    peak_time = forecast.idxmax() if not forecast.empty else None
    mean_forecast = float(forecast.mean()) if not forecast.empty else None

    delta = None
    if latest_actual is not None and first_forecast is not None and latest_actual != 0:
        delta = f"{(first_forecast / latest_actual - 1):+.1%} vs latest actual"

    columns = st.columns(4)
    columns[0].metric("Latest observed load", _format_mw(latest_actual))
    columns[1].metric("Next-hour forecast", _format_mw(first_forecast), delta=delta)
    columns[2].metric(
        "Forecast peak",
        _format_mw(peak),
        delta=_format_timestamp(peak_time),
        delta_color="off",
    )
    columns[3].metric("Mean forecast load", _format_mw(mean_forecast))


def _timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    output = df.reset_index()
    return output.rename(columns={output.columns[0]: "timestamp"})


def _forecast_chart(
    results: DashboardResults,
    *,
    horizon: int,
    context_points: int,
) -> alt.LayerChart:
    forecast = results.forecast.iloc[:horizon].copy()
    forecast_plot = _timestamp_column(forecast)
    actual = results.recent_actual.tail(context_points).rename("load_mw").to_frame()
    actual_plot = _timestamp_column(actual)

    history = results.backtest_history
    history_plot = pd.DataFrame()
    if not history.empty and "forecast_load_mw" in history.columns:
        backcast = history["forecast_load_mw"].tail(context_points).rename("load_mw").to_frame()
        history_plot = _timestamp_column(backcast)

    x_axis = alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d %b · %H:%M", labelOverlap=True))
    y_axis = alt.Y("load_mw:Q", title="Load (MW)", scale=alt.Scale(zero=False))

    layers: list[alt.Chart] = []
    if {"lower_pi", "upper_pi"}.issubset(forecast_plot.columns):
        interval = (
            alt.Chart(forecast_plot)
            .mark_area(color=SWISS_RED, opacity=0.12)
            .encode(
                x=x_axis,
                y=alt.Y("lower_pi:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
                y2="upper_pi:Q",
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time", format="%d %b %Y, %H:%M"),
                    alt.Tooltip("lower_pi:Q", title="Lower bound", format=",.0f"),
                    alt.Tooltip("upper_pi:Q", title="Upper bound", format=",.0f"),
                ],
            )
        )
        layers.append(interval)

    if not history_plot.empty:
        layers.append(
            alt.Chart(history_plot)
            .mark_line(color=SLATE, opacity=0.55, strokeDash=[5, 4], strokeWidth=1.5)
            .encode(x=x_axis, y=y_axis)
        )

    if not actual_plot.empty:
        layers.append(
            alt.Chart(actual_plot)
            .mark_line(color=BLUE, strokeWidth=2.3)
            .encode(
                x=x_axis,
                y=y_axis,
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time", format="%d %b %Y, %H:%M"),
                    alt.Tooltip("load_mw:Q", title="Observed", format=",.0f"),
                ],
            )
        )

    forecast_line = (
        alt.Chart(forecast_plot)
        .mark_line(color=SWISS_RED, strokeWidth=3)
        .encode(
            x=x_axis,
            y=alt.Y("forecast_load_mw:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%d %b %Y, %H:%M"),
                alt.Tooltip("forecast_load_mw:Q", title="Forecast", format=",.0f"),
            ],
        )
    )
    layers.append(forecast_line)

    if not forecast_plot.empty:
        boundary = pd.DataFrame({"timestamp": [forecast_plot["timestamp"].iloc[0]]})
        layers.append(
            alt.Chart(boundary)
            .mark_rule(color=SWISS_RED, opacity=0.35, strokeDash=[3, 3])
            .encode(x="timestamp:T")
        )

    return (
        alt.layer(*layers)
        .properties(height=440)
        .resolve_scale(y="shared")
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#E8EBEF", domainColor="#D6DADE", labelColor=SLATE, titleColor=SLATE)
    )


def _performance_chart(results: DashboardResults) -> alt.Chart | None:
    history = results.backtest_history
    baseline_column = (
        "baseline_load_mw"
        if "baseline_load_mw" in history.columns
        else "naive_baseline_load_mw"
    )
    required = ["true_load_mw", "forecast_load_mw", baseline_column]
    if history.empty or not set(required).issubset(history.columns):
        return None

    baseline_label = _baseline_label(results.metadata.get("baseline_name"))

    chart_df = history[required].tail(240).rename(
        columns={
            "true_load_mw": "Observed",
            "forecast_load_mw": "LSTM forecast",
            baseline_column: baseline_label,
        }
    )
    chart_df = _timestamp_column(chart_df).melt(
        id_vars="timestamp", var_name="series", value_name="load_mw"
    )
    return (
        alt.Chart(chart_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d %b · %H:%M")),
            y=alt.Y("load_mw:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Observed", "LSTM forecast", baseline_label],
                    range=[BLUE, SWISS_RED, SLATE],
                ),
                legend=alt.Legend(orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Observed", "LSTM forecast", baseline_label],
                    range=[[1, 0], [1, 0], [6, 4]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%d %b %Y, %H:%M"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("load_mw:Q", title="Load", format=",.0f"),
            ],
        )
        .properties(height=390)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#E8EBEF", domainColor="#D6DADE", labelColor=SLATE, titleColor=SLATE)
    )


def _render_forecast_tab(
    results: DashboardResults,
    *,
    horizon: int,
    context_points: int,
) -> None:
    st.markdown('<p class="section-kicker">Forward outlook</p>', unsafe_allow_html=True)
    st.subheader(f"Next {horizon} hours")
    st.caption(
        "Observed load is blue, the current LSTM forecast is red, and recent historical "
        "forecasts are dashed. The shaded band is the calibrated prediction interval."
    )
    st.altair_chart(
        _forecast_chart(results, horizon=horizon, context_points=context_points),
        use_container_width=True,
    )

    forecast_table = results.forecast.iloc[:horizon].copy()
    table_names = {
        "forecast_load_mw": "Forecast (MW)",
        "lower_pi": "Lower bound (MW)",
        "upper_pi": "Upper bound (MW)",
        "interval_width": "Interval width (MW)",
    }
    visible_columns = [column for column in table_names if column in forecast_table.columns]
    display_table = forecast_table[visible_columns].rename(columns=table_names)
    display_table.index.name = "Timestamp"

    with st.expander("Forecast values and export"):
        st.dataframe(display_table.round(1), use_container_width=True)
        st.download_button(
            "Download forecast CSV",
            data=display_table.to_csv().encode("utf-8"),
            file_name="switzerland_load_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_performance_tab(results: DashboardResults) -> None:
    metrics = results.metrics
    if not metrics:
        st.info("Recent accuracy results are not available for this model and dataset.")
        return

    st.markdown('<p class="section-kicker">Recent accuracy check</p>', unsafe_allow_html=True)
    st.subheader("Model performance")
    st.caption(
        "We compare the model with four simple forecasts: repeat the latest value, use the "
        "same hour yesterday, use the same hour last week, or average yesterday and last week. "
        "The dashboard shows the simple forecast with the smallest average error. MAE is the "
        "average difference from the real demand, so smaller is better."
    )

    columns = st.columns(4)
    baseline_label = _baseline_label(results.metadata.get("baseline_name"))
    columns[0].metric("Model average error", _format_mw(metrics.get("mae_model"), 1))
    columns[1].metric(f"{baseline_label} average error", _format_mw(metrics.get("mae_baseline"), 1))
    improvement = metrics.get("mae_improvement_pct")
    columns[2].metric(
        "Error reduction",
        f"{float(improvement):.1f}%" if improvement is not None else "—",
        delta=f"vs {baseline_label.lower()}",
        delta_color="off",
    )
    coverage = metrics.get("coverage")
    target = metrics.get("target_coverage")
    coverage_delta = None
    if coverage is not None and target is not None and math.isfinite(float(coverage)):
        coverage_delta = f"{float(coverage) - float(target):+.1%} vs target"
    columns[3].metric("Actual values inside forecast range", _format_percent(coverage), delta=coverage_delta)

    chart = _performance_chart(results)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

    baseline_metrics = results.metadata.get("baseline_eval_metrics", {})
    if isinstance(baseline_metrics, dict) and baseline_metrics:
        baseline_rows = [
            {
                "Baseline": _baseline_label(name),
                "MAE (MW)": values.get("mae"),
                "RMSE (MW)": values.get("rmse"),
            }
            for name, values in baseline_metrics.items()
            if isinstance(values, dict)
        ]
        with st.expander("Compare all simple forecasts"):
            st.dataframe(
                pd.DataFrame(baseline_rows).sort_values("MAE (MW)").round(1),
                hide_index=True,
                use_container_width=True,
            )
            calibration_choice = _baseline_label(
                results.metadata.get("baseline_selected_on_calibration")
            )
            st.caption(
                f"During the earlier comparison period, {calibration_choice} had the smallest error. "
                "All four methods are shown so you can judge the model against each one."
            )

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Error details")
        details = pd.DataFrame(
            {
                "Metric": ["Average error (MAE)", "Large-error score (RMSE)"],
                "LSTM": [metrics.get("mae_model"), metrics.get("rmse_model")],
                baseline_label: [
                    metrics.get("mae_baseline"),
                    metrics.get("rmse_baseline"),
                ],
            }
        )
        st.dataframe(details.round(1), hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### How this was checked")
        st.write(f"**Recent forecasts checked:** {int(metrics.get('n_eval_windows', 0)):,}")
        st.write(f"**Earlier forecasts used to set the range:** {int(metrics.get('n_calibration_windows', 0)):,}")
        st.write(f"**Average forecast-range width:** {_format_mw(metrics.get('mean_interval_width'), 1)}")
        st.write(f"**Range adjustment speed:** {float(metrics.get('aci_eta', 0)):.3f}")


def _render_data_tab(
    results: DashboardResults,
    df: pd.DataFrame,
    *,
    model_path: Path,
    features_path: Path,
) -> None:
    metadata = results.metadata
    st.markdown('<p class="section-kicker">Provenance & readiness</p>', unsafe_allow_html=True)
    st.subheader("Data and model health")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Feature dataset")
        st.metric("Rows", f"{int(metadata['dataset_rows']):,}")
        st.write(f"**Coverage:** {_format_timestamp(metadata['dataset_start'])} → {_format_timestamp(metadata['dataset_end'])}")
        st.write(f"**Columns:** {int(metadata['dataset_columns']):,}")
        st.write(f"**Missing model inputs:** {_format_percent(metadata['missing_share'], 2)}")
        st.write(f"**Target:** `{metadata['target_column']}`")
    with right:
        st.markdown("#### Model details")
        st.metric("Forecast horizon", f"{int(metadata['horizon'])} hours")
        st.write(f"**Lookback window:** {int(metadata['lookback'])} hours")
        st.write(f"**Input features:** {int(metadata['feature_count'])}")
        st.write(f"**Forecast-weather inputs:** {int(metadata['forecast_weather_feature_count'])}")
        st.write(f"**Model file version:** {int(metadata['model_version'])}")
        st.write(f"**Calculation device:** `{metadata['device']}`")
        st.write(f"**Forecast range:** {metadata['interval_method']}")

    with st.expander("Files used"):
        st.code(f"Model:    {model_path}\nFeatures: {features_path}", language="text")

    with st.expander("Latest feature rows"):
        preview_columns = [metadata["target_column"]]
        preview_columns.extend(column for column in df.columns if column not in preview_columns)
        st.dataframe(df[preview_columns[:8]].tail(24), use_container_width=True)


def _render_missing_file(path: Path, label: str) -> None:
    st.error(f"{label} not found")
    st.write(f"Expected path: `{path}`")
    if label == "Feature table":
        st.code("poetry run python src/processing/processing.py", language="bash")
    else:
        st.code("poetry run python -m src.modeling.train_lstm", language="bash")


def main() -> None:
    st.set_page_config(
        page_title="Swiss Load Outlook",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    with st.sidebar:
        st.markdown("## ⚡ Swiss Load Outlook")
        st.caption("Operational forecast controls")
        with st.expander("Data sources"):
            model_value = st.text_input("Model file", value=DEFAULT_MODEL_PATH)
            features_value = st.text_input("Feature table", value=DEFAULT_FEATURES_PATH)

    model_path = _resolve_path(model_value)
    features_path = _resolve_path(features_value)
    if not model_path.is_file():
        _render_missing_file(model_path, "Model file")
        st.stop()
    if not features_path.is_file():
        _render_missing_file(features_path, "Feature table")
        st.stop()

    model_modified_ns = model_path.stat().st_mtime_ns
    features_modified_ns = features_path.stat().st_mtime_ns
    try:
        artifact = _load_artifact(str(model_path), model_modified_ns)
        df = _load_features(str(features_path), features_modified_ns)
    except Exception as exc:
        st.error("The dashboard could not load its model or feature data.")
        st.exception(exc)
        st.stop()

    max_context = max(24, min(1_000, len(df)))
    default_context = min(336, max_context)
    with st.sidebar:
        st.markdown("### Forecast view")
        horizon = st.slider(
            "Hours to display",
            min_value=1,
            max_value=int(artifact.horizon),
            value=min(24, int(artifact.horizon)),
        )
        context_points = st.slider(
            "History to display",
            min_value=24,
            max_value=max_context,
            value=default_context,
            step=24,
            help="Number of observed hourly points shown before the forecast.",
        )
        st.markdown("### Uncertainty")
        include_intervals = st.checkbox("Show prediction interval", value=True)
        confidence = st.slider(
            "Confidence level",
            min_value=0.70,
            max_value=0.99,
            value=0.90,
            step=0.01,
            disabled=not include_intervals,
        )
        per_horizon = st.checkbox(
            "Adjust each forecast hour separately",
            value=True,
            disabled=not include_intervals,
            help="Lets each hour use a forecast range based on its own recent errors.",
        )
        st.caption(
            f"The accuracy check uses {CALIBRATION_WINDOWS} earlier forecasts to set the ranges "
            f"and {EVALUATION_WINDOWS} recent forecasts to measure results."
        )
        if st.button("Clear dashboard cache", use_container_width=True):
            _load_artifact.clear()
            _load_features.clear()
            _compute_results.clear()
            st.rerun()

    with st.spinner("Running forecast and recent backtest…"):
        try:
            results = _compute_results(
                str(model_path),
                model_modified_ns,
                str(features_path),
                features_modified_ns,
                confidence,
                per_horizon,
                include_intervals,
            )
        except Exception as exc:
            st.error("Forecast generation failed.")
            st.exception(exc)
            st.stop()

    _render_hero(results)
    for notice in results.notices:
        st.warning(notice)
    _render_summary_metrics(results)

    forecast_tab, performance_tab, data_tab = st.tabs(
        ["Forecast", "Performance", "Data & model"]
    )
    with forecast_tab:
        _render_forecast_tab(results, horizon=horizon, context_points=context_points)
    with performance_tab:
        _render_performance_tab(results)
    with data_tab:
        _render_data_tab(
            results,
            df,
            model_path=model_path,
            features_path=features_path,
        )

    st.caption(
        "Forecasts are model estimates and should be interpreted alongside operational context. "
        "Times are displayed in the timezone stored in the feature dataset."
    )


if __name__ == "__main__":
    main()
