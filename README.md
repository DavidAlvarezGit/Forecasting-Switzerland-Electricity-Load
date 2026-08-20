# Switzerland Electricity Load Forecasting

Forecast Switzerland's aggregate electricity demand once per day for the following 24 hours. The project combines ENTSO-E load history with point-in-time ECMWF weather forecasts, evaluates two transparent seasonal baselines and an LSTM on daily rolling origins, calibrates adaptive uncertainty ranges, and presents the results in Streamlit.

## Why this project exists

An electricity forecast can look excellent while using information that would not have existed when the forecast was issued. Common failure modes include selecting weather values by target time instead of forecast run, filling missing history from future observations, and tuning uncertainty ranges on the final test period.

This project makes the forecast contract explicit:

- one forecast is issued at 12:00 `Europe/Zurich` each day;
- every issue predicts exactly `t+1` through `t+24`;
- daylight-saving changes preserve local noon and are handled in UTC internally;
- each origin uses one complete weather-model run available before issuance;
- missing load is filled only from older observations;
- all scalers and model choices are learned before the final test period;
- uncertainty settings are selected on validation data and updated only after outcomes are observable.

## How it works

```text
ENTSO-E load + archived ECMWF forecast runs
                    |
                    v
       point-in-time and DST validation
                    |
                    v
       one 24-hour sample per local day
                    |
                    v
   previous-day | previous-week | LSTM
                    |
                    v
        daily rolling-origin evaluation
                    |
                    v
 adaptive conformal forecast ranges (ACI)
                    |
                    v
             Streamlit dashboard
```

The primary weather input is the ECMWF 00:00 UTC run exposed by Open-Meteo's [Single Runs API](https://open-meteo.com/en/docs/single-runs-api). A conservative six-hour publication delay is recorded, which keeps that run before local noon in both winter and summer. If the archive reports a missing or incomplete run, ingestion searches backward only, accepts the most recent complete pre-origin run, and records the exact run, assumed availability time, retrieval time, source, model, target timestamps, and lead hours. It never replaces a missing vintage with a later forecast.

## Model inputs

There is no LightGBM ranking step and no top-five cutoff. The LSTM uses every declared production input:

| Input group | Inputs |
|---|---:|
| Load sequence | 336 hourly values ending at the forecast origin |
| Load context | 24-hour and 168-hour lags, means, and standard deviations |
| Forecast weather | 5 variables x 24 future leads = 120 values |
| Calendar | weekday and annual cycles plus a weekend flag |
| LSTM output | 24 hourly load values |

The weather variables are temperature, relative humidity, precipitation, cloud cover, and wind speed, averaged across ten Swiss locations. This retains all relevant forecast-weather and seasonality inputs while keeping the feature contract explicit and reproducible.

## Results

### Chronological evaluation

| Split | Daily origins | Period |
|---|---:|---|
| Train | 473 | 15 Mar 2024 - 30 Jun 2025 |
| Validation | 184 | 1 Jul 2025 - 31 Dec 2025 |
| Final test | 108 | 1 Jan 2026 - 18 Apr 2026 |

The model is trained on the train split. Early stopping and adaptive conformal settings use validation only. The final-test outcomes are untouched until the figures below are calculated.

| Forecast | MAE | RMSE |
|---|---:|---:|
| LSTM | **451.76 MW** | **569.86 MW** |
| Same hour yesterday | 515.79 MW | 676.11 MW |
| Same hour last week | 544.07 MW | 717.60 MW |

Against the previous-day baseline, the LSTM reduces MAE by 12.4% and RMSE by 15.7% on the final test.

### Error by forecast lead

| Hours ahead | LSTM MAE | Yesterday MAE | Last week MAE |
|---:|---:|---:|---:|
| 1 | **387.29 MW** | 571.93 MW | 553.50 MW |
| 6 | **380.69 MW** | 477.68 MW | 502.99 MW |
| 12 | 407.36 MW | **340.15 MW** | 406.46 MW |
| 24 | **451.24 MW** | 541.51 MW | 571.75 MW |

The LSTM does not win every lead: yesterday's value is stronger at lead 12. Reporting the horizon separately makes that weakness visible instead of hiding it inside one average.

### Adaptive forecast range

Adaptive Conformal Inference (ACI) targets 90% coverage. Its update rate, memory window, and per-lead configuration are selected on validation only; final-test errors update the calibrator only for subsequent daily origins.

| Measure | Final-test result |
|---|---:|
| Target coverage | 90.0% |
| Observed coverage | 90.7% |
| Mean full range width | 1,933.24 MW |
| Selected memory | 120 days |
| Per-lead calibration | Yes |

All 24 lead-specific errors, coverage values, interval widths, selected settings, split dates, and training metadata are published in [`daily_lstm_24h.metrics.json`](data/processed/models/daily_lstm_24h.metrics.json).

## Dashboard

The Streamlit application has three views:

- **Forecast** shows the latest archived 24-hour issue, observed context, point forecast, uncertainty range, and CSV export.
- **Performance** compares the LSTM with yesterday and last week, including results at leads 1, 6, 12, and 24.
- **Data & model** exposes forecast origins, split counts, weather run and availability metadata, feature counts, model version, and artifact paths.

Dashboard wording uses plain language; ACI remains named in the technical audit. Model and sample caches are keyed by file modification time, so replacing an artifact invalidates stale results.

## Run locally

Requirements:

- Python 3.12
- Poetry

```powershell
poetry install --with dev
poetry run python -m streamlit run app/streamlit_app.py
```

The compact daily sample table and trained model under `data/processed/` allow the dashboard to run without downloading raw data or retraining.

## Rebuild data and model

Create a root `.env` file with an [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) API key:

```env
ENTSOE_API_KEY=your_api_key_here
```

Then run:

```powershell
poetry run python -m src.ingestion.entsoe
poetry run python -m src.ingestion.openmeteo
poetry run python -m src.processing.processing
poetry run python -m src.modeling.train_lstm
```

Raw ENTSO-E partitions, weather runs, and ingestion state are incremental and rebuildable. The weather job checkpoints every 25 origins and retries individual failed dates, so one upstream error does not discard a long archive run.

## Tests and continuous integration

```powershell
poetry run ruff check app src tests
poetry run python -m unittest discover -s tests -v
```

The deterministic tests cover:

- local-noon origins across daylight-saving transitions;
- weather-run availability and rejection of late vintages;
- exact `t+1` to `t+24` target alignment;
- causal load imputation;
- explicit previous-day and previous-week baseline alignment;
- per-lead metrics and 24-value model output;
- adaptive conformal updates that affect only later origins.

GitHub Actions installs the Poetry lock, checks project metadata, runs Ruff, and executes the synthetic unit suite. CI does not download the full datasets or train the model.

## Project structure

```text
app/                    Streamlit presentation and dashboard orchestration
src/forecast_config.py  forecast origin, horizon, splits, weather, ACI, and paths
src/ingestion/          ENTSO-E and point-in-time weather-run ingestion
src/processing/         causal daily sample construction
src/modeling/           LSTM, baselines, rolling metrics, ACI, and inference
data/processed/         compact dashboard samples, model, and measured metrics
tests/                  deterministic forecasting-contract tests
.github/workflows/      dependency, lint, and unit-test CI
```

## Current limits

Before production use, the project would need:

- a scheduler that collects each weather run and publishes each daily issue in real time;
- monitoring for missing feeds, delayed ENTSO-E observations, revisions, and forecast drift;
- an operationally verified weather-publication service-level agreement instead of the conservative six-hour assumption;
- a longer evaluation history covering more weather and demand regimes;
- regional load or richer holiday/event inputs if they become reliably available at issuance;
- scheduled retraining, artifact versioning, rollback, and alerting.

The bundled dashboard is therefore an auditable archived evaluation, not a claim of a live production service.
