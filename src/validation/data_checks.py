import pandas as pd

def check_sorted_index(df: pd.DataFrame):
    assert df.index.is_monotonic_increasing, "Index is not sorted"


def check_no_duplicates(df: pd.DataFrame):
    assert not df.index.duplicated().any(), "Duplicate timestamps found"


def check_timezone(df: pd.DataFrame):
    assert df.index.tz is not None, "Index must be timezone-aware"


def check_frequency(df: pd.DataFrame, freq="H"):
    diffs = df.index.to_series().diff().dropna()
    assert diffs.nunique() == 1, "Irregular time steps"
    assert diffs.iloc[0] == pd.Timedelta(freq), f"Expected freq {freq}"


def check_no_missing_timestamps(df: pd.DataFrame, freq="H"):
    expected = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    missing = expected.difference(df.index)
    assert len(missing) == 0, f"Missing timestamps: {missing[:5]}"


def check_no_nan(df: pd.DataFrame):
    assert not df.isna().any().any(), "NaN values present"


def check_non_negative(df: pd.DataFrame, column: str):
    assert (df[column] >= 0).all(), f"Negative values in {column}"



