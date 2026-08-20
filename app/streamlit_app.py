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

from app.dashboard_core import DashboardResults, build_dashboard_results  # noqa: E402
from src.modeling import inference as inference_mod  # noqa: E402

DEFAULT_MODEL_PATH = "data/processed/models/lstm_24h.pt"
DEFAULT_FEATURES_PATH = "data/processed/forecast_samples.parquet"
EVALUATION_ORIGINS = 120 * 24
HISTORY_ORIGINS = 30 * 24
RESULTS_CACHE_VERSION = 4

RED = "#D52B1E"
BLUE = "#1769AA"
SLATE = "#687684"
GOLD = "#9A7B4F"


@st.cache_resource(show_spinner=False)
def _load_artifact(path: str, modified_ns: int):
    del modified_ns
    return inference_mod.load_lstm_artifact(path)


@st.cache_data(show_spinner=False)
def _load_samples(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    samples = pd.read_parquet(path)
    if not isinstance(samples.index, pd.DatetimeIndex):
        samples.index = pd.to_datetime(samples.index, utc=True)
    return samples.sort_index()


@st.cache_data(show_spinner=False, ttl=3600)
def _compute_results(
    model_path: str,
    model_modified_ns: int,
    samples_path: str,
    samples_modified_ns: int,
    include_intervals: bool,
    cache_version: int,
) -> DashboardResults:
    del cache_version
    artifact = _load_artifact(model_path, model_modified_ns)
    samples = _load_samples(samples_path, samples_modified_ns)
    return build_dashboard_results(
        samples,
        artifact,
        forecast_next_horizon=inference_mod.forecast_next_horizon,
        forecast_next_horizon_with_intervals=inference_mod.forecast_next_horizon_with_intervals,
        evaluate_recent_backtest=inference_mod.evaluate_recent_backtest,
        eval_origins=EVALUATION_ORIGINS,
        history_origins=HISTORY_ORIGINS,
        include_intervals=include_intervals,
    )


def _path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _mw(value: float | int | None, decimals: int = 0) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):,.{decimals}f} MW"


def _percent(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.1%}"


def _time(value: Any, timezone: str = "Europe/Zurich") -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(timezone)
    return timestamp.strftime("%d %b %Y · %H:%M")


def _styles() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{ max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }}
        .hero {{ border: 1px solid rgba(24,33,43,.10); border-radius: 18px; padding: 1.5rem 1.7rem;
          background: radial-gradient(circle at 94% 14%, rgba(213,43,30,.14), transparent 24%), #fff;
          box-shadow: 0 10px 32px rgba(24,33,43,.06); margin-bottom: 1rem; }}
        .eyebrow {{ color:{RED}; font-size:.75rem; font-weight:750; letter-spacing:.11em; text-transform:uppercase; }}
        .hero h1 {{ margin:.25rem 0 .65rem; font-size:clamp(2rem,4vw,3.15rem); letter-spacing:-.04em; }}
        .hero p {{ color:{SLATE}; max-width:850px; margin:0 0 .8rem; }}
        .chip {{ display:inline-block; border:1px solid rgba(24,33,43,.10); border-radius:999px;
          padding:.35rem .65rem; margin:.2rem .35rem 0 0; font-size:.78rem; color:#44515E; }}
        div[data-testid="stMetric"] {{ background:white; border:1px solid rgba(24,33,43,.09);
          border-radius:14px; padding:.9rem 1rem; box-shadow:0 5px 18px rgba(24,33,43,.04); }}
        .kicker {{ color:{RED}; font-weight:700; font-size:.76rem; letter-spacing:.09em;
          text-transform:uppercase; margin-bottom:-.55rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(results: DashboardResults) -> None:
    metadata = results.metadata
    origin = escape(_time(metadata["latest_origin"], metadata["timezone"]))
    frequency = metadata["forecast_frequency"]
    title = "Hourly" if frequency == "hourly" else "Daily"
    schedule = "latest complete hourly origin" if frequency == "hourly" else "latest noon origin"
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">Swiss grid outlook</div>
          <h1>{title} electricity load forecast</h1>
          <p>This forecast uses the {schedule}. The model predicts the following
          24 hourly demand values using load known at issuance and one auditable weather-model run.</p>
          <span class="chip">Forecast origin: {origin}</span>
          <span class="chip">24 hourly leads</span>
          <span class="chip">{metadata['nominal_coverage']:.0%} adaptive forecast range</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _summary(results: DashboardResults) -> None:
    forecast = results.forecast["forecast_load_mw"]
    metrics = results.metrics
    columns = st.columns(4)
    columns[0].metric("First forecast hour", _mw(forecast.iloc[0]))
    columns[1].metric("24-hour average", _mw(forecast.mean()))
    columns[2].metric("24-hour peak", _mw(forecast.max()), delta=_time(forecast.idxmax()), delta_color="off")
    columns[3].metric("Final-test MAE", _mw(metrics.get("mae_model"), 1))


def _forecast_chart(results: DashboardResults, context_hours: int) -> alt.LayerChart:
    forecast = results.forecast.reset_index(names="timestamp")
    actual = (
        results.recent_actual.tail(context_hours)
        .rename("load_mw")
        .rename_axis("timestamp")
        .reset_index()
    )
    layers: list[alt.Chart] = []
    if {"lower_pi", "upper_pi"}.issubset(forecast.columns):
        layers.append(
            alt.Chart(forecast).mark_area(color=RED, opacity=.13).encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("lower_pi:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
                y2="upper_pi:Q",
            )
        )
    if not actual.empty:
        layers.append(
            alt.Chart(actual).mark_line(color=BLUE, strokeWidth=2.3).encode(
                x="timestamp:T", y=alt.Y("load_mw:Q", scale=alt.Scale(zero=False))
            )
        )
    layers.append(
        alt.Chart(forecast).mark_line(color=RED, strokeWidth=3).encode(
            x="timestamp:T",
            y=alt.Y("forecast_load_mw:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
            tooltip=["timestamp:T", alt.Tooltip("forecast_load_mw:Q", format=",.0f")],
        )
    )
    return alt.layer(*layers).properties(height=430).resolve_scale(y="shared")


def _performance_chart(results: DashboardResults) -> alt.Chart | None:
    history = results.backtest_history
    required = {
        "forecast_origin_utc",
        "true_load_mw",
        "forecast_load_mw",
        "previous_day_load_mw",
        "previous_week_load_mw",
    }
    if history.empty or not required.issubset(history.columns):
        return None
    # Hourly forecast windows overlap by design. Keep the latest forecast for each
    # target timestamp in the chart; metrics retain every origin/lead pair.
    plot = (
        history.reset_index(names="timestamp")
        .sort_values(["timestamp", "forecast_origin_utc"])
        .drop_duplicates("timestamp", keep="last")
        .tail(24 * 7)
        .rename(
            columns={
                "true_load_mw": "Observed",
                "forecast_load_mw": "LSTM",
                "previous_day_load_mw": "Same hour yesterday",
                "previous_week_load_mw": "Same hour last week",
            }
        )
    )
    long = plot.melt(
        id_vars="timestamp",
        value_vars=["Observed", "LSTM", "Same hour yesterday", "Same hour last week"],
        var_name="series",
        value_name="load_mw",
    )
    return alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("load_mw:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "series:N",
            title=None,
            scale=alt.Scale(
                domain=["Observed", "LSTM", "Same hour yesterday", "Same hour last week"],
                range=[BLUE, RED, SLATE, GOLD],
            ),
        ),
        tooltip=["timestamp:T", "series:N", alt.Tooltip("load_mw:Q", format=",.0f")],
    ).properties(height=390)


def _forecast_tab(results: DashboardResults, context_hours: int) -> None:
    st.markdown('<p class="kicker">24-hour outlook</p>', unsafe_allow_html=True)
    st.subheader("24-hour forecast")
    st.caption(
        "Blue shows demand known when the forecast was issued. Red starts one hour later, "
        "and shading shows the uncertainty range."
    )
    st.altair_chart(_forecast_chart(results, context_hours), use_container_width=True)
    table = results.forecast.rename(
        columns={
            "forecast_load_mw": "Forecast (MW)",
            "lower_pi": "Lower bound (MW)",
            "upper_pi": "Upper bound (MW)",
            "interval_width": "Range width (MW)",
        }
    )
    with st.expander("Forecast values and CSV export"):
        st.dataframe(table.round(1), use_container_width=True)
        st.download_button(
            "Download forecast CSV",
            table.to_csv().encode("utf-8"),
            "switzerland_hourly_load_forecast.csv",
            "text/csv",
            use_container_width=True,
        )


def _performance_tab(results: DashboardResults) -> None:
    metrics = results.metrics
    if not metrics:
        st.info("Final-test results are unavailable.")
        return
    st.markdown('<p class="kicker">Untouched final test</p>', unsafe_allow_html=True)
    st.subheader("Rolling-origin performance")
    frequency = results.metadata["forecast_frequency"]
    cadence = "every hour" if frequency == "hourly" else "once per day"
    st.caption(
        f"One forecast is evaluated {cadence}. The LSTM is compared with the same hours "
        "yesterday and the same hours last week. Smaller errors are better."
    )
    baselines = metrics["baseline_metrics"]
    columns = st.columns(4)
    columns[0].metric("LSTM average error", _mw(metrics["mae_model"], 1))
    columns[1].metric("Yesterday average error", _mw(baselines["previous_day"]["overall_mae"], 1))
    columns[2].metric("Last week average error", _mw(baselines["previous_week"]["overall_mae"], 1))
    columns[3].metric("Values inside forecast range", _percent(metrics["coverage"]))
    chart = _performance_chart(results)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### Results by forecast hour")
        rows = []
        for lead in (1, 6, 12, 24):
            key = str(lead)
            rows.append(
                {
                    "Hour ahead": lead,
                    "LSTM MAE": metrics["model_metrics"]["headline_leads"][key]["mae"],
                    "Yesterday MAE": baselines["previous_day"]["headline_leads"][key]["mae"],
                    "Last week MAE": baselines["previous_week"]["headline_leads"][key]["mae"],
                    "Inside forecast range": f"{metrics['coverage_by_lead'][key]:.1%}",
                }
            )
        st.dataframe(pd.DataFrame(rows).round(1), hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### Evaluation details")
        aci = metrics["aci_config"]
        st.write(f"**Test forecasts:** {metrics['n_eval_windows']:,}")
        st.write(f"**Validation forecasts for ranges:** {metrics['n_calibration_windows']:,}")
        st.write(f"**Target coverage:** {metrics['target_coverage']:.0%}")
        st.write(f"**Average range width:** {_mw(metrics['mean_interval_width'], 1)}")
        st.write(f"**Adjustment speed / memory:** {aci['eta']:.3f} / {aci['window_size']} forecasts")
        st.caption(
            "The adaptive conformal inference (ACI) settings were chosen on validation data only."
        )


def _data_tab(
    results: DashboardResults,
    samples: pd.DataFrame,
    model_path: Path,
    samples_path: Path,
) -> None:
    metadata = results.metadata
    st.markdown('<p class="kicker">Point-in-time audit</p>', unsafe_allow_html=True)
    st.subheader("Data and model")
    left, right = st.columns(2)
    with left:
        label = "Hourly forecast origins" if metadata["forecast_frequency"] == "hourly" else "Daily forecast origins"
        st.metric(label, f"{metadata['dataset_rows']:,}")
        st.write(f"**Origin range:** {_time(metadata['dataset_start'])} → {_time(metadata['dataset_end'])}")
        st.write(f"**Split counts:** {metadata['split_counts']}")
        st.write(f"**Latest weather run:** {_time(metadata['latest_weather_run'], 'UTC')} UTC")
        st.write(f"**Run available by:** {_time(metadata['latest_weather_available'], 'UTC')} UTC")
        st.write(f"**Weather model:** {metadata['latest_weather_model']}")
        st.write(f"**Archived on:** {_time(metadata['latest_weather_retrieved'], 'UTC')} UTC")
    with right:
        st.metric("Forecast horizon", f"{metadata['horizon']} hours")
        schedule = "Every hour" if metadata["forecast_frequency"] == "hourly" else "Daily at noon"
        st.write(f"**Forecast schedule:** {schedule} ({metadata['timezone']})")
        st.write(f"**Load history:** {metadata['lookback']} hours")
        st.write(f"**Context inputs:** {metadata['feature_count']}")
        st.write(f"**Weather inputs:** {metadata['forecast_weather_feature_count']}")
        st.write(f"**Forecast ranges:** {metadata['interval_method']}")
    with st.expander("Files and latest audit rows"):
        st.code(f"Model:   {model_path}\nSamples: {samples_path}", language="text")
        audit = [
            "forecast_origin_local",
            "weather_run_utc",
            "weather_available_utc",
            "weather_model",
            "weather_run_is_fallback",
            "weather_source",
            "weather_retrieved_at_utc",
            "target_start_utc",
            "target_end_utc",
            "split",
        ]
        st.dataframe(samples[[column for column in audit if column in samples]].tail(10))


def main() -> None:
    st.set_page_config(page_title="Swiss Load Outlook", page_icon="⚡", layout="wide")
    _styles()
    with st.sidebar:
        st.markdown("## ⚡ Swiss Load Outlook")
        st.caption("Rolling 24-hour electricity forecasts")
        with st.expander("Files"):
            model_value = st.text_input("Model", DEFAULT_MODEL_PATH)
            samples_value = st.text_input("Hourly samples", DEFAULT_FEATURES_PATH)
        context_hours = st.slider("Observed history to show", 24, 24 * 14, 24 * 7, step=24)
        include_intervals = st.checkbox("Show forecast range", value=True)
        st.caption("Range settings are fixed from validation and cannot be tuned on this dashboard.")
        if st.button("Clear dashboard cache", use_container_width=True):
            _load_artifact.clear()
            _load_samples.clear()
            _compute_results.clear()
            st.rerun()

    model_path = _path(model_value)
    samples_path = _path(samples_value)
    if not model_path.is_file() or not samples_path.is_file():
        st.error("The model or hourly sample file is missing.")
        st.code(
            "poetry run python -m src.processing.processing\n"
            "poetry run python -m src.modeling.train_lstm",
            language="bash",
        )
        st.stop()
    model_mtime = model_path.stat().st_mtime_ns
    samples_mtime = samples_path.stat().st_mtime_ns
    try:
        samples = _load_samples(str(samples_path), samples_mtime)
        results = _compute_results(
            str(model_path),
            model_mtime,
            str(samples_path),
            samples_mtime,
            include_intervals,
            RESULTS_CACHE_VERSION,
        )
    except Exception as exc:
        st.error("Forecast generation failed.")
        st.exception(exc)
        st.stop()

    _hero(results)
    for notice in results.notices:
        st.warning(notice)
    _summary(results)
    forecast_tab, performance_tab, data_tab = st.tabs(["Forecast", "Performance", "Data & model"])
    with forecast_tab:
        _forecast_tab(results, context_hours)
    with performance_tab:
        _performance_tab(results)
    with data_tab:
        _data_tab(results, samples, model_path, samples_path)


if __name__ == "__main__":
    main()
