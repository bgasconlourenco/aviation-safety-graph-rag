import pytest

from src.graph_load import _normalize_mission, _to_rows
from src.ingest import RAW_CSV, clean
from src.ingest import load as load_csv

# --- Mission normalisation ---

@pytest.mark.parametrize("raw, expected", [
    ("Ambulance", "EMS"),
    ("Ambulance; Other Other", "EMS"),
    ("Other EMS", "EMS"),
    ("Other HEMS", "EMS"),
    ("Other ORGAN TRANSPLANT", "EMS"),
    ("Training", "Training"),
    ("Training; Other Other", "Training"),
    ("Passenger", "Passenger"),
    ("Personal", "Personal"),
    ("Ferry / Re-Positioning", "Ferry / Re-Positioning"),
    ("Cargo / Freight / Delivery", "Cargo"),
    ("Photo Shoot / Video", "Photo / Video"),
    ("Tactical", "Tactical"),
    ("Test Flight / Demonstration", "Test Flight"),
    ("Other TV", "News / Media"),
    ("Other News Gathering", "News / Media"),
    ("Other Tour", "Tourism"),
    ("Other TOUR", "Tourism"),
    ("Other Law Enforcement", "Utility / Public Safety"),
    ("Other Pipeline Patrol", "Utility / Public Safety"),
    ("Other Other", "Other"),
    ("Other", "Other"),
    (None, "Other"),
    ("", "Other"),
])
def test_normalize_mission(raw, expected):
    assert _normalize_mission(raw) == expected


# --- Row serialisation ---

@pytest.fixture(scope="module")
def df():
    if not RAW_CSV.exists():
        pytest.skip("Raw data file not available")
    return clean(load_csv())


def test_to_rows_returns_list(df):
    rows = _to_rows(df.head(10))
    assert isinstance(rows, list)
    assert len(rows) == 10


def test_to_rows_no_nan(df):
    import math
    rows = _to_rows(df.head(50))
    for row in rows:
        for key, val in row.items():
            if isinstance(val, float):
                assert not math.isnan(val), f"NaN found in field '{key}'"


def test_to_rows_date_is_string(df):
    rows = _to_rows(df.head(10))
    for row in rows:
        if row["date"] is not None:
            assert isinstance(row["date"], str)
            assert len(row["date"]) == 10  # YYYY-MM-DD


def test_to_rows_anomaly_is_list(df):
    rows = _to_rows(df.head(50))
    for row in rows:
        assert isinstance(row["anomaly_broad"], list)


def test_to_rows_mission_clean_never_none(df):
    rows = _to_rows(df)
    assert all(row["mission_clean"] is not None for row in rows)


def test_to_rows_acn_present(df):
    rows = _to_rows(df.head(10))
    assert all("acn" in row for row in rows)


# --- connect() requires env vars ---

def test_connect_raises_without_env(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    from src.graph_load import connect
    with pytest.raises(EnvironmentError):
        connect()
