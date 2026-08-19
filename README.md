# Switzerland Electricity Load Forecasting

Forecast aggregate Swiss electricity demand for the next 24 hours from load history, weather forecasts, observed weather, and calendar signals. The project covers incremental data collection, causal feature engineering, LSTM training, adaptive uncertainty calibration, recent backtesting, and an interactive Streamlit dashboard.

## Why this project exists

Electricity demand changes with the hour, weekday, season, weather, and recent consumption pattern. A useful forecast must therefore do more than produce one number: it should preserve information timing, compare itself with a clear reference forecast, communicate uncertainty, and expose the freshness of its inputs.

This project:

- collects Swiss load observations from ENTSO-E;
- collects observed, historical-forecast, fixed-vintage, and live weather data from Open-Meteo;
- aligns the sources into an hourly UTC dataset;
- engineers lagged, rolling, forecast-weather, and cyclical features without backward filling;
- ranks predictors against several points in the complete 24-hour forecast horizon;
- predicts all 24 lead times with one LSTM;
- calibrates intervals with Adaptive Conformal Inference (ACI);
- compares the model with one simple reference: demand from the same hour yesterday;
- presents forecasts, performance, and artifact health in Streamlit.

It is a forecasting prototype, not a production dispatch or trading system.

## How it works

1. ENTSO-E load and Open-Meteo weather products are fetched incrementally and stored as Parquet partitions.
2. Missing weather values are filled only from earlier observations. The two missing load hours in the current snapshot use the same hour one week earlier and remain marked by an audit feature.
3. Stitched historical forecasts are used only as past encoder inputs. Future-weather training features come from `previous_day1` values predicted 24 hours before their valid time; overlapping live forecasts replace them at inference time.
4. Training uses chronological train, validation, and test splits so future targets never leak into earlier samples.
5. LightGBM importance is aggregated across 1-, 6-, 12-, and 24-hour targets. The model retains 32 selected signals plus load history, including all seven calendar features and at least eight forecast-weather leads.
6. A two-layer LSTM reads the preceding 336 hours and predicts all 24 lead times in one pass.
7. ACI maintains rolling residual buffers and adapts its miscoverage level after observed misses. Backtests delay each update until that forecast's complete horizon is observable.
8. The only naive baseline repeats the observed demand from the same hour 24 hours earlier. It is fixed in advance and never selected from several alternatives.
9. Streamlit serves the forward forecast, uncertainty band, recent backcasts, evaluation metrics, and data/model metadata.

If interval calibration or backtesting cannot run, the dashboard keeps the point forecast available and explains which diagnostics are missing.

## Results

### Chronological test split

The selected model is compared with demand from the same hour yesterday.

| Measure | LSTM | Same hour yesterday |
|---|---:|---:|
| Mean absolute error | 420.21 MW | 513.64 MW |
| Root mean squared error | 533.74 MW | 674.60 MW |
| MAE reduction | 18.2% | - |

### Recent rolling backtest

The dashboard evaluates the 120 most recent hourly forecast windows against the fixed previous-day reference.

| Model | MAE | RMSE |
|---|---:|---:|
| LSTM | 464.64 MW | 599.10 MW |
| Previous day | 479.47 MW | 656.97 MW |

| Interval measure | Result |
|---|---:|
| Target coverage | 90.0% |
| Empirical ACI coverage | 95.5% |
| Mean interval width | 2,453.83 MW |

Recent MAE is 3.1% lower than the previous-day reference, while RMSE is 8.8% lower. These figures describe overlapping rolling windows for the included artifacts, not an independent production benchmark.

## Dashboard

The application is organized around three views:

- **Forecast** shows observed demand, recent model backcasts, the current 24-hour forecast, and its ACI uncertainty band. The visible horizon and historical context are adjustable, and forecast values can be downloaded as CSV.
- **Performance** compares the LSTM with demand from the same hour yesterday and reports ACI coverage and evaluation scope.
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

The repository includes the default feature table and model checkpoint under `data/processed/`. Use the Poetry environment rather than a global Python installation because the Parquet file requires the pinned PyArrow version.

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

Open-Meteo ingestion stores four distinct products:

- `weather_historical` for realised historical conditions;
- `weather_historical_forecast_hourly` for past encoder values in the same model-data domain as operational forecasts;
- `weather_previous_runs` for fixed 24-hour-vintage future covariates used during training;
- `weather_live_forecast` for the latest future covariates used operationally.

State files under `data/state/` record completed timestamps so later runs append instead of refetching the full history. Training writes the checkpoint and its JSON evaluation summary under `data/processed/models/`.

## Configuration

The main settings are defined by `LSTMTrainConfig` in [`src/modeling/lstm_pipeline.py`](src/modeling/lstm_pipeline.py).

| Setting | Default |
|---|---:|
| Lookback | 336 hours |
| Forecast horizon | 24 hours |
| Train / validation / test split | 70% / 15% / 15% |
| Selected predictors | 32 plus load history |
| Feature-ranking horizons | 1, 6, 12, and 24 hours |
| Hidden size | 64 |
| LSTM layers | 2 |
| Maximum epochs | 10 |
| Early-stopping patience | 3 epochs |

Dashboard calibration and evaluation window counts are defined near the top of [`app/streamlit_app.py`](app/streamlit_app.py).

## Project structure

```text
.streamlit/             dashboard theme
app/                    Streamlit UI and framework-independent dashboard analysis
data/raw/               partitioned ENTSO-E and Open-Meteo products
data/interim/           aligned hourly dataset
data/processed/         model features and trained artifacts
data/state/             incremental-ingestion checkpoints
notebook/               exploratory analysis and training experiments
src/ingestion/          source clients and ingestion state
src/processing/         causal alignment and feature engineering
src/modeling/           LSTM training, inference, ACI, and backtesting
src/validation/         reusable data checks
tests/                  causal preprocessing, baseline, weather-lead, and ACI tests
```

## Validation

The current pipeline was checked by:

- running six regression tests for causal imputation, weather-lead alignment, the previous-day baseline, and ACI adaptation;
- compiling the application and pipeline modules;
- rebuilding 19,691 causal feature rows with 168 fixed-vintage/live forecast-weather columns and no missing values;
- verifying horizon-aware selection retains 32 inputs, eight future-weather signals, and seven calendar signals;
- training and reloading the checked-in checkpoint;
- running point forecasting, ACI, the previous-day baseline, and recent backtesting end to end;
- validating both Altair chart specifications;
- rendering all three dashboard views in headless Chromium without Streamlit exceptions.

Run the regression suite with:

```powershell
python -m unittest discover -s tests -v
```

## Current limits

Before production use, the project would need:

- scheduled ingestion, retraining, and model promotion;
- independent evaluation over additional seasons and demand regimes;
- additional weather vintages and run metadata beyond the conservative 24-hour previous-run product;
- monitoring for data freshness, feature drift, error, interval coverage, and latency;
- configurable ingestion ranges instead of module-level dates;
- authentication, access control, and a multi-user deployment design.

## License

Licensed under the [Apache License 2.0](LICENSE).
