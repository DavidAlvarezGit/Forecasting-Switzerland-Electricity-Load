import pandas as pd


def _as_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
    return df.index


def check_required_fields(df: pd.DataFrame, required_fields: list[str]):
    missing_fields = [field for field in required_fields if field not in df.columns]
    assert not missing_fields, f"Missing required fields: {missing_fields}"

def check_sorted_index(df: pd.DataFrame):
    index = _as_datetime_index(df)
    assert index.is_monotonic_increasing, "Index is not sorted"


def check_no_duplicates(df: pd.DataFrame):
    index = _as_datetime_index(df)
    assert not index.duplicated().any(), "Duplicate timestamps found"


def check_timezone(df: pd.DataFrame):
    index = _as_datetime_index(df)
    assert index.tz is not None, "Index must be timezone-aware"


def check_frequency(df: pd.DataFrame, freq: str = "H"):
    index = _as_datetime_index(df)
    inferred_freq = pd.infer_freq(index)
    assert inferred_freq is not None, "Irregular time steps"
    expected_offset = pd.tseries.frequencies.to_offset(freq)
    inferred_offset = pd.tseries.frequencies.to_offset(inferred_freq)
    assert inferred_offset == expected_offset, f"Expected freq {freq}, got {inferred_freq}"


def check_no_missing_timestamps(df: pd.DataFrame, freq: str = "H"):
    index = _as_datetime_index(df)
    expected = pd.date_range(index.min(), index.max(), freq=freq, tz=index.tz)
    missing = expected.difference(index)
    assert len(missing) == 0, f"Missing timestamps: {missing[:5]}"


def check_no_nan(df: pd.DataFrame):
    assert not df.isna().any().any(), "NaN values present"


def check_non_negative(df: pd.DataFrame, column: str):
    assert (df[column] >= 0).all(), f"Negative values in {column}"



