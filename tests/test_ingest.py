import pytest
import pandas as pd
from src.ingest import load, RAW_CSV, KEEP_COLS, VALID_FLIGHT_CONDITIONS


@pytest.fixture(scope="module")
def df():
    if not RAW_CSV.exists():
        pytest.skip("Raw data file not available")
    return load()


def test_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)


def test_expected_columns(df):
    assert set(KEEP_COLS.values()) == set(df.columns)


def test_acn_unique(df):
    assert df["acn"].is_unique


def test_flight_conditions_values(df):
    actual = set(df["flight_conditions"].dropna())
    assert actual <= VALID_FLIGHT_CONDITIONS


def test_date_parsed(df):
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_no_empty_rows(df):
    assert len(df) > 0
