import pandas as pd
import pytest

from src.ingest import KEEP_COLS, RAW_CSV, VALID_FLIGHT_CONDITIONS, clean, load


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


@pytest.fixture(scope="module")
def cleaned(df):
    return clean(df)


def test_clean_anomaly_list_is_list(cleaned):
    non_null = cleaned["anomaly_list"].dropna()
    assert all(isinstance(v, list) for v in non_null)


def test_clean_anomaly_broad_known_values(cleaned):
    known = {"Inflight Event", "Ground Event", "Equipment Problem", "No Anomaly",
             "Airspace Violation", "Flight Deck", "Deviation", "ATC Issue", "Conflict", "Other"}
    all_values = {v for row in cleaned["anomaly_broad"].dropna() for v in row}
    assert all_values <= known


def test_clean_visibility_numeric(cleaned):
    non_null = cleaned["visibility_sm"].dropna()
    assert pd.api.types.is_float_dtype(non_null)


def test_clean_weather_elements_is_list(cleaned):
    non_null = cleaned["weather_elements"].dropna()
    assert all(isinstance(v, list) for v in non_null)


def test_clean_flight_phase_primary_is_single(cleaned):
    non_null = cleaned["flight_phase_primary"].dropna()
    assert all(";" not in v for v in non_null)
