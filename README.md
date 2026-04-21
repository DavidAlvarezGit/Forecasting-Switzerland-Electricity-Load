# Switzerland Electricity Load Forecasting

This repository builds and serves a forecasting workflow for Swiss electricity load using ENTSO-E load data and Open-Meteo weather features.

## 1. Project Overview

The project predicts short-term electricity demand with an LSTM-based model and serves forecasts in a Streamlit dashboard.

Why this project matters:

- electricity demand forecasting is a difficult task that matters for many companies
- weather and seasonality strongly affect load, so feature engineering and robust ingestion are essential
- a usable app should show both forecasts and uncertainty, not only point estimates

Current capabilities include:

- incremental ingestion from ENTSO-E and Open-Meteo
- preprocessing and feature generation for sequence modeling
- LSTM training
- inference with optional prediction intervals and backtest diagnostics
- Streamlit app for operational visualization

## 2. Problem Definition

Load forecasting combines non-stationary behavior, weather sensitivity, and strong intraday/weekly seasonality. A simple baseline may be stable but often underperforms during regime changes.

This repository addresses that by:

- consolidating multi-source hourly data into aligned datasets
- engineering lag, rolling, and cyclical features
- training a sequence model for multi-step forecasting (LSTM)
- evaluating against a naive baseline and interval coverage

## 3. Technical Approach

The workflow is organized into four stages:

1. Ingestion
- ENTSO-E load ingestion in [src/ingestion/entsoe.py](src/ingestion/entsoe.py)
- Open-Meteo historical, historical-forecast, and live forecast ingestion in [src/ingestion/openmeteo.py](src/ingestion/openmeteo.py)
- incremental updates via JSON state files in [data/state](data/state)

2. Processing and Feature Engineering
- raw-to-interim aggregation in [src/processing/helpers.py](src/processing/helpers.py)
- full pipeline entry in [src/processing/processing.py](src/processing/processing.py)
- LSTM feature table creation in [src/processing/features.py](src/processing/features.py)

3. Modeling and Inference
- training pipeline in [src/modeling/lstm_pipeline.py](src/modeling/lstm_pipeline.py)
- training entrypoint in [src/modeling/train_lstm.py](src/modeling/train_lstm.py)
- inference and backtest logic in [src/modeling/inference.py](src/modeling/inference.py)

4. App
- Streamlit dashboard in [app/streamlit_app.py](app/streamlit_app.py)
- default model path: [data/processed/models/best_lstm_24h.pt](data/processed/models)
- default feature table path: [data/processed/lstm_features.parquet](data/processed/lstm_features.parquet)



## 4. Current Outputs

The repository currently provides:

- processed feature datasets under [data/processed](data/processed)
- model checkpoints and selected model information under [data/processed/models](data/processed/models)
- an interactive dashboard with:
	- point forecasts
	- prediction intervals that are created using Adaptive Conformal Inference (ACI)
	- recent historical forecasts
	- model vs naive baseline metrics


## 5. How to Run

### Prerequisites

- Python 3.12
- Poetry
- ENTSO-E API key in `.env` as `ENTSOE_API_KEY`

Example `.env`:

```env
ENTSOE_API_KEY=your_key_here
```

### Install Dependencies

```bash
poetry install
```

### Run Data Ingestion

```bash
poetry run python src/ingestion/entsoe.py
poetry run python src/ingestion/openmeteo.py
```

### Build Aggregated and Feature Datasets

```bash
poetry run python src/processing/processing.py
```

### Train the LSTM Model

```bash
poetry run python -m src.modeling.train_lstm
```

### Launch the Streamlit App

```bash
poetry run streamlit run app/streamlit_app.py
```

## 6. Project Structure

```text
load_forecasting/
|- app/
|  |- streamlit_app.py
|- data/
|  |- raw/
|  |  |- entsoe/
|  |  |- weather_historical/
|  |  |- weather_historical_forecast_hourly/
|  |  |- weather_live_forecast/
|  |- interim/
|  |  |- aggregated.parquet
|  |- processed/
|  |  |- lstm_features.parquet
|  |  |- models/
|  |- state/
|     |- entsoe_state.json
|     |- openmeteo_state.json
|     |- openmeteo_historical_forecast_state.json
|- notebook/
|- src/
|  |- ingestion/
|  |  |- entsoe.py
|  |  |- openmeteo.py
|  |  |- settings.py
|  |  |- state_utils.py
|  |- processing/
|  |  |- helpers.py
|  |  |- processing.py
|  |  |- features.py
|  |- modeling/
|  |  |- lstm_pipeline.py
|  |  |- inference.py
|  |  |- train_lstm.py
|  |- validation/
|     |- core.py
|     |- data_checks.py
|- pyproject.toml
|- requirements.txt
|- README.md
```

## 7. Future Improvements

- add automated retraining and model selection schedules
- add monitoring of the model over time
- add automated fetching of API data