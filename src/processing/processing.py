from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if __package__ in (None, ""):
    from helpers import run_processing_pipeline
else:
    from .helpers import run_processing_pipeline


def main() -> None:
    df, meta = run_processing_pipeline()

    print(f"Using {meta['weather_features']} weather features across {meta['city_count']} cities")
    print(f"Weather columns after pivot: {meta['weather_columns']}")
    print(f"Date range: {meta['start']} -> {meta['end']} | rows={meta['rows']}")
    print(f"Output: {meta['output_path']}")
    print(df.head(5))


if __name__ == "__main__":
    main()
