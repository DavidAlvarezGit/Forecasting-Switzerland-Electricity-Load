from __future__ import annotations

from pathlib import Path

from .lstm_pipeline import LSTMTrainConfig, run_training_pipeline


def main() -> None:
    results = run_training_pipeline(LSTMTrainConfig())

    compare_df = results["compare"]
    model_out_path: Path = results["model_out_path"]
    metrics_path: Path = results["metrics_path"]

    print("Model comparison:")
    print(compare_df)
    print(f"Saved model to: {model_out_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
