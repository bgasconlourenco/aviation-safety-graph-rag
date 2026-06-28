"""Phases 2–3: load, validate, and clean ASRS CSV exports."""

from pathlib import Path

import pandas as pd

RAW_CSV = Path(__file__).parent.parent / "data" / "raw" / "ASRS_DBOnline_full.csv"

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

# Ordered longest-match first so longer prefixes are checked before shorter ones
_ANOMALY_PREFIXES = [
    ("Inflight Event", "Inflight Event"),
    ("Ground Excursion", "Ground Event"),
    ("Ground Incursion", "Ground Event"),
    ("Ground Event", "Ground Event"),
    ("Aircraft Equipment Problem", "Equipment Problem"),
    ("No Specific Anomaly Occurred", "No Anomaly"),
    ("Airspace Violation", "Airspace Violation"),
    ("Flight Deck", "Flight Deck"),
    ("Deviation", "Deviation"),
    ("ATC Issue", "ATC Issue"),
    ("Conflict", "Conflict"),
    ("Other", "Other"),
]


def _anomaly_broad(item: str) -> str:
    for prefix, label in _ANOMALY_PREFIXES:
        if item.startswith(prefix):
            return label
    return "Other"


def _parse_weather(raw: str) -> tuple[list[str], float | None]:
    parts = [p.strip() for p in raw.split(";")]
    try:
        visibility = float(parts[-1])
        # Filter out any remaining parts that are also purely numeric (duplicate visibility values)
        elements = [p for p in parts[:-1] if not p.replace(".", "", 1).isdigit()]
        return elements, visibility
    except ValueError:
        return parts, None


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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Derive structured columns from the raw loaded DataFrame."""
    df = df.copy()

    df["anomaly_list"] = df["anomaly"].str.split(r"\s*;\s*")
    df["anomaly_broad"] = df["anomaly_list"].apply(
        lambda items: list(dict.fromkeys(_anomaly_broad(i) for i in items))
        if isinstance(items, list)
        else None
    )

    parsed = df["weather"].dropna().apply(_parse_weather)
    df["weather_elements"] = parsed.apply(lambda t: t[0])
    df["visibility_sm"] = parsed.apply(lambda t: t[1])

    df["flight_phase_primary"] = df["flight_phase"].str.split(r"\s*;\s*").str[0]

    return df
