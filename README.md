# Switzerland Electricity Load Forecasting

Forecast aggregate Swiss electricity demand for the next 24 hours from historical load, weather, and calendar signals. The project covers incremental data collection, hourly feature engineering, LSTM training, uncertainty calibration, recent backtesting, and an interactive Streamlit dashboard.

## Why this project exists

Electricity demand changes with the hour, weekday, season, weather, and recent consumption pattern. A useful forecast must therefore do more than produce one number: it should preserve time order during training, compare itself with a credible baseline, communicate uncertainty, and make the freshness of its inputs visible.

This project:

- collects Swiss load observations from ENTSO-E;
- collects historical and forecast weather data from Open-Meteo;
- aligns both sources into an hourly UTC dataset;
- engineers lagged, rolling, weather, and cyclical features;
- predicts the complete 24-hour horizon with one LSTM;
- calibrates prediction intervals from recent forecast residuals;
- compares the model with a last-value persistence baseline;
- presents forecasts, performance, and artifact health in Streamlit.

It is a forecasting prototype, not a production dispatch or trading system.

## How it works

1. ENTSO-E load data and Open-Meteo weather data are fetched incrementally and stored as Parquet partitions.
2. The processing pipeline aligns the sources by hourly UTC timestamp and creates the model feature table.
3. Training uses chronological train, validation, and test splits so future observations never leak into earlier samples.
4. Feature selection removes redundant inputs before the LSTM is trained with early stopping.
5. The model predicts all 24 lead times from the preceding 336 hours.
6. Recent absolute residuals provide split-conformal-style prediction widths, either separately by lead time or pooled across the horizon.
7. A rolling backtest compares the model with a persistence forecast that repeats the last observed load.
8. Streamlit serves the forward forecast, uncertainty band, recent backcasts, evaluation metrics, and data/model metadata.

If interval calibration or backtesting cannot run, the dashboard keeps the point forecast available and explains which diagnostics are missing.

## Results

The current checked-in model and feature snapshot were evaluated over the 120 most recent rolling forecast windows. Prediction intervals used the preceding 120 calibration windows at a 90% target coverage level.

| Measure | LSTM | Persistence baseline |
|---|---:|---:|
| Mean absolute error | 411.24 MW | 856.50 MW |
| MAE reduction | 52.0% | — |

| Interval measure | Result |
|---|---:|
| Target coverage | 90.0% |
| Empirical coverage | 95.4% |

These figures describe the recent rolling dashboard backtest for the included artifacts, not an independent production benchmark. The evaluation windows overlap, and performance may change after the data or model is rebuilt.

## Dashboard

The application is organized around three views:

- **Forecast** shows observed demand, recent model backcasts, the current 24-hour forecast, and its uncertainty band. The visible horizon and historical context are adjustable, and forecast values can be downloaded as CSV.
- **Performance** compares LSTM and persistence errors, plots both against observed load, and reports interval coverage and evaluation scope.
- **Data & model** reports dataset coverage, missing model inputs, lookback, horizon, feature count, inference device, artifact paths, and the latest feature rows.

Model and feature files can be changed from the sidebar. Cached results are keyed by each artifact's modification time, so replacing a file invalidates stale dashboard output.

## Run locally

Requirements:

- Python 3.12
- Poetry

```powershell
poetry install
poetry run streamlit run app/streamlit_app.py
```

The repository includes the default runtime artifacts:

```text
data/processed/lstm_features.parquet
data/processed/models/best_lstm_24h.pt
```

Use the Poetry environment rather than a global Python installation. The included Parquet file requires the PyArrow version pinned by the project.

## Rebuild the data and model

Fetching new ENTSO-E data requires an API key in a root-level `.env` file:

```env
ENTSOE_API_KEY=your_api_key_here
```

Run the pipeline from the repository root:

```powershell
poetry run python src/ingestion/entsoe.py
poetry run python src/ingestion/openmeteo.py
poetry run python src/processing/processing.py
poetry run python -m src.modeling.train_lstm
```

The ingestion entry points currently define their date ranges in code. State files under `data/state/` record the latest completed timestamps so later runs can append rather than refetch the full history.

Training writes the selected checkpoint to `data/processed/models/best_lstm_24h.pt` and its evaluation summary to the matching `.metrics.json` file. The checkpoint contains the weights, chosen feature names, normalization statistics, target column, lookback, and forecast horizon needed for inference.

## Configuration

The main training settings are defined by `LSTMTrainConfig` in [`src/modeling/lstm_pipeline.py`](src/modeling/lstm_pipeline.py).

| Setting | Default |
|---|---:|
| Lookback | 336 hours |
| Forecast horizon | 24 hours |
| Train / validation / test split | 70% / 15% / 15% |
| Hidden size | 128 |
| LSTM layers | 5 |
| Maximum epochs | 15 |
| Early-stopping patience | 5 epochs |

Dashboard calibration and evaluation window counts are defined near the top of [`app/streamlit_app.py`](app/streamlit_app.py).

## Project structure

```text
.streamlit/             dashboard theme
app/                    Streamlit UI and framework-independent dashboard analysis
data/raw/               partitioned ENTSO-E and Open-Meteo data
data/interim/           aligned hourly dataset
data/processed/         model features and trained artifacts
data/state/             incremental-ingestion checkpoints
notebook/               exploratory analysis and training experiments
src/ingestion/          source clients and ingestion state
src/processing/         alignment and feature engineering
src/modeling/           LSTM training, inference, intervals, and backtesting
src/validation/         reusable data checks
```

## Validation

The current dashboard refactor was checked by:

- compiling the application modules;
- running the checked-in model and 19,990-row feature table through point forecasting, interval calibration, and recent backtesting;
- validating both Altair chart specifications;
- rendering all three dashboard views in headless Chromium without Streamlit exceptions.

There is not yet a committed automated test suite. Adding unit tests for feature construction and forecast-window boundaries, plus a repeatable benchmark report, is a priority before production use.

## Current limits

Before production use, the project would need:

- scheduled ingestion, retraining, and model promotion;
- automated tests and continuous integration;
- independent evaluation over additional seasons and demand regimes;
- monitoring for data freshness, feature drift, error, interval coverage, and latency;
- configurable ingestion ranges instead of module-level dates;
- authentication, access control, and a multi-user deployment design.

## License

Licensed under the [Apache License 2.0](LICENSE).
