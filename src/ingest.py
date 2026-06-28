"""Phase 2: load and validate ASRS CSV exports."""

from pathlib import Path

import pandas as pd

RAW_CSV = Path(__file__).parent.parent / "data" / "raw" / "ASRS_DBOnline.csv"

KEEP_COLS = {
    (" ", "ACN"): "acn",
    ("Time", "Date"): "date",
    ("Environment", "Flight Conditions"): "flight_conditions",
    ("Environment", "Weather Elements / Visibility"): "weather",
    ("Environment", "Light"): "light",
    ("Aircraft 1", "Flight Phase"): "flight_phase",
    ("Aircraft 1", "Mission"): "mission",
    ("Events", "Anomaly"): "anomaly",
    ("Assessments", "Primary Problem"): "primary_problem",
    ("Report 1", "Narrative"): "narrative",
    ("Report 1", "Synopsis"): "synopsis",
}

VALID_FLIGHT_CONDITIONS = {"VMC", "IMC", "Marginal", "Mixed"}


def load(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the ASRS CSV export and return a cleaned DataFrame."""
    df = pd.read_csv(path, header=[0, 1])

    missing = [col for col in KEEP_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing from CSV: {missing}")

    df = df[list(KEEP_COLS)]
    df.columns = list(KEEP_COLS.values())

    if not df["acn"].is_unique:
        raise ValueError("ACN column contains duplicate values")

    unexpected = set(df["flight_conditions"].dropna()) - VALID_FLIGHT_CONDITIONS
    if unexpected:
        raise ValueError(f"Unexpected Flight Conditions values: {unexpected}")

    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    df["acn"] = df["acn"].astype(str)

    return df.reset_index(drop=True)
