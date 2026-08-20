from __future__ import annotations

from .features import run_feature_pipeline


def main() -> None:
    feature_df, feature_meta = run_feature_pipeline()
    print(
        f"Daily forecast samples: {feature_meta['start']} -> {feature_meta['end']} | "
        f"rows={feature_meta['rows']} cols={feature_meta['columns']}"
    )
    print(f"Feature output: {feature_meta['output_path']}")
    print(feature_df.tail(5))


if __name__ == "__main__":
    main()
