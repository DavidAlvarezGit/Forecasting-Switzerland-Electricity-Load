# Switzerland Electricity Load Forecasting

End-to-end **24-hour electricity demand forecasting for Switzerland** using ENTSO-E load data, archived weather forecasts, a multi-horizon LSTM, and adaptive conformal prediction.

The project is designed to reproduce a realistic forecasting setting: every historical prediction uses only information that would have been available at the forecast origin.

On the final held-out test period, the LSTM achieves **436.8 MW MAE**, improving on the previous-day baseline by **15.4%**. Adaptive conformal intervals achieve **89.5% empirical coverage for a 90% target**.

## Architecture

```text
ENTSO-E load + archived weather forecasts
                    |
                    v
          incremental ingestion
                    |
                    v
       point-in-time validation
                    |
                    v
        causal feature pipeline
                    |
                    v
     seasonal baselines + LSTM
                    |
                    v
      rolling-origin backtest
                    |
                    v
 adaptive conformal prediction
                    |
                    v
          Streamlit dashboard
```

## Forecasting setup

A new forecast is produced at every UTC hour.

Each forecast origin `t` predicts:

```text
t+1, t+2, ..., t+24
```

The pipeline uses chronological train, validation, and test periods. Future observations are excluded from preprocessing, model selection, calibration, and evaluation.

Weather features are also point-in-time: each historical forecast uses an archived weather-model run available before the corresponding forecast origin.

## Data

### Electricity load

Swiss aggregate electricity demand is collected from the **ENTSO-E Transparency Platform**.

### Weather

Historical forecast vintages are obtained through the **Open-Meteo Historical Forecast API**.

Five weather variables are aggregated across ten Swiss locations:

* temperature;
* relative humidity;
* precipitation;
* cloud cover;
* wind speed.

For each forecast origin, the model receives the next 24 forecast values for each weather variable.

## Model

The forecasting model is a **multi-horizon PyTorch LSTM** predicting all 24 future hours in one pass.

Inputs include:

| Feature group     |                                     Inputs |
| ----------------- | -----------------------------------------: |
| Load history      |                         Previous 336 hours |
| Load context      |   24h and 168h lags and rolling statistics |
| Weather forecasts |                     5 variables × 24 leads |
| Calendar          | Hour, weekday, annual cycles, weekend flag |
| Output            |                 24 hourly demand forecasts |

Two transparent seasonal baselines are evaluated alongside the LSTM:

* same hour one day earlier;
* same hour one week earlier.

## Evaluation

The project uses an **hourly rolling-origin backtest** rather than a random train/test split.

At each historical forecast origin, the system:

1. constructs features using only currently available information;
2. generates a 24-hour forecast;
3. compares predictions with future observations;
4. updates adaptive uncertainty calibration only after the relevant targets become observable.

### Data splits

| Split      | Forecast origins | Period                    |
| ---------- | ---------------: | ------------------------- |
| Train      |           12,793 | 14 Jan 2024 – 30 Jun 2025 |
| Validation |            4,416 | 1 Jul 2025 – 31 Dec 2025  |
| Final test |            2,590 | 1 Jan 2026 – 18 Apr 2026  |

## Results

### Final-test performance

| Model         |           MAE |          RMSE |
| ------------- | ------------: | ------------: |
| **LSTM**      | **436.77 MW** | **553.80 MW** |
| Previous day  |     516.29 MW |     676.73 MW |
| Previous week |     543.98 MW |     717.57 MW |

Compared with the strongest seasonal baseline, the LSTM reduces:

* **MAE by 15.4%**
* **RMSE by 18.2%**

### Performance by forecast horizon

| Hours ahead |      LSTM MAE | Previous day | Previous week |
| ----------: | ------------: | -----------: | ------------: |
|           1 | **382.54 MW** |    515.92 MW |     544.47 MW |
|           6 | **425.25 MW** |    516.63 MW |     544.97 MW |
|          12 | **434.94 MW** |    515.95 MW |     544.25 MW |
|          24 | **449.35 MW** |    516.15 MW |     542.59 MW |

Reporting errors by lead makes performance degradation across the 24-hour horizon visible rather than hiding it inside one aggregate metric.

## Uncertainty quantification

Forecasts are complemented with **Adaptive Conformal Inference (ACI)** prediction intervals targeting 90% coverage.

ACI hyperparameters are selected on the validation period only.

Because hourly 24-step forecasts overlap, calibration residuals are added only when their targets have actually become observable.

| Metric              |    Final test |
| ------------------- | ------------: |
| Target coverage     |         90.0% |
| Observed coverage   |         89.5% |
| Mean interval width |   1,817.39 MW |
| Calibration memory  | 168 residuals |

## Dashboard

A Streamlit application provides three views:

**Forecast** — latest 24-hour prediction, observed history, and uncertainty intervals.

**Performance** — comparison between the LSTM and seasonal baselines across the forecast horizon.

**Data & model** — model configuration, split information, feature metadata, and forecasting artifacts.

## Engineering

The project includes:

* incremental ENTSO-E and weather ingestion;
* Parquet-based data storage;
* causal preprocessing and feature generation;
* DST-safe UTC forecasting;
* model artifact persistence;
* deterministic unit tests;
* Ruff linting;
* GitHub Actions continuous integration;
* Streamlit deployment interface.

The test suite checks the forecasting contract, including target alignment, weather-vintage availability, causal load imputation, baseline alignment, 24-step model output, and delayed conformal updates.

## Run locally

Requirements:

```text
Python 3.12
Poetry
```

Install dependencies and launch the dashboard:

```powershell
poetry install --with dev
poetry run python -m streamlit run app/streamlit_app.py
```

To rebuild the data and model, add an ENTSO-E API key to `.env`:

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

## Project structure

```text
app/                    Streamlit dashboard
src/ingestion/          ENTSO-E and weather ingestion
src/processing/         point-in-time feature pipeline
src/modeling/           LSTM, baselines, evaluation and ACI
data/processed/         model artifacts and dashboard data
tests/                  deterministic unit tests
.github/workflows/      continuous integration
```

