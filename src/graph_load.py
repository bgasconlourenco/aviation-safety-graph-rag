"""Phase 5: batched Cypher loading into Neo4j AuraDB."""

import os

import pandas as pd
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()

BATCH_SIZE = 500

# --- Mission normalisation ---

_EMS_KEYWORDS = frozenset({"ambulance", "ems", "hems", "lifeguard", "organ"})

_CLEAN_MISSIONS = {
    "Training": "Training",
    "Passenger": "Passenger",
    "Personal": "Personal",
    "Ferry / Re-Positioning": "Ferry / Re-Positioning",
    "Cargo / Freight / Delivery": "Cargo",
    "Tactical": "Tactical",
    "Photo Shoot / Video": "Photo / Video",
    "Traffic Watch": "Traffic Watch",
    "Test Flight / Demonstration": "Test Flight",
    "Agriculture": "Agriculture",
    "Utility / Infrastructure": "Utility",
    "Search & Rescue": "Search & Rescue",
}

_KEYWORD_MISSIONS = [
    ({"news", "tv", "media", "film", "camera", "broadcast", "gather"}, "News / Media"),
    ({"tour", "sightseeing"}, "Tourism"),
    ({"utility", "power", "pipeline", "patrol", "survey", "forestry",
      "fire", "law", "police", "surveillance", "enforcement"}, "Utility / Public Safety"),
]


def _normalize_mission(raw: str | None) -> str:
    if not isinstance(raw, str):
        return "Other"

    lower = raw.lower()

    if any(kw in lower for kw in _EMS_KEYWORDS):
        return "EMS"

    first = raw.split(";")[0].strip()
    label = first[6:].strip() if first.startswith("Other ") else first

    if label in _CLEAN_MISSIONS:
        return _CLEAN_MISSIONS[label]

    label_lower = label.lower()
    for keywords, name in _KEYWORD_MISSIONS:
        if any(kw in label_lower for kw in keywords):
            return name

    return "Other"


# --- Neo4j connection ---

def connect() -> Driver:
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not password:
        raise EnvironmentError("NEO4J_URI and NEO4J_PASSWORD must be set in environment or .env")
    return GraphDatabase.driver(uri, auth=(username, password))


# --- Schema constraints ---

_CONSTRAINTS = [
    "CREATE CONSTRAINT incident_acn IF NOT EXISTS FOR (i:Incident) REQUIRE i.acn IS UNIQUE",
    "CREATE CONSTRAINT flight_condition_name IF NOT EXISTS FOR (fc:FlightCondition) REQUIRE fc.name IS UNIQUE",
    "CREATE CONSTRAINT light_condition_name IF NOT EXISTS FOR (lc:LightCondition) REQUIRE lc.name IS UNIQUE",
    "CREATE CONSTRAINT mission_type_name IF NOT EXISTS FOR (mt:MissionType) REQUIRE mt.name IS UNIQUE",
    "CREATE CONSTRAINT anomaly_type_name IF NOT EXISTS FOR (a:AnomalyType) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT problem_type_name IF NOT EXISTS FOR (pp:ProblemType) REQUIRE pp.name IS UNIQUE",
    "CREATE CONSTRAINT flight_phase_name IF NOT EXISTS FOR (fp:FlightPhase) REQUIRE fp.name IS UNIQUE",
]


def create_constraints(driver: Driver) -> None:
    with driver.session() as session:
        for constraint in _CONSTRAINTS:
            session.run(constraint)


# --- Cypher load statements (one per relationship type) ---

_LOAD_INCIDENTS = """
UNWIND $rows AS row
MERGE (i:Incident {acn: row.acn})
SET i.date          = row.date,
    i.year          = row.year,
    i.visibility_sm = row.visibility_sm,
    i.narrative     = row.narrative,
    i.synopsis      = row.synopsis
"""

_LOAD_CONDITIONS = """
UNWIND $rows AS row
WITH row WHERE row.flight_conditions IS NOT NULL
MATCH (i:Incident {acn: row.acn})
MERGE (fc:FlightCondition {name: row.flight_conditions})
MERGE (i)-[:HAD_CONDITIONS]->(fc)
"""

_LOAD_LIGHT = """
UNWIND $rows AS row
WITH row WHERE row.light IS NOT NULL
MATCH (i:Incident {acn: row.acn})
MERGE (lc:LightCondition {name: row.light})
MERGE (i)-[:OCCURRED_IN]->(lc)
"""

_LOAD_MISSION = """
UNWIND $rows AS row
MATCH (i:Incident {acn: row.acn})
MERGE (mt:MissionType {name: row.mission_clean})
MERGE (i)-[:WAS_MISSION]->(mt)
"""

_LOAD_PROBLEM = """
UNWIND $rows AS row
WITH row WHERE row.primary_problem IS NOT NULL
MATCH (i:Incident {acn: row.acn})
MERGE (pp:ProblemType {name: row.primary_problem})
MERGE (i)-[:HAS_PRIMARY_PROBLEM]->(pp)
"""

_LOAD_PHASE = """
UNWIND $rows AS row
WITH row WHERE row.flight_phase_primary IS NOT NULL
MATCH (i:Incident {acn: row.acn})
MERGE (fp:FlightPhase {name: row.flight_phase_primary})
MERGE (i)-[:DURING_PHASE]->(fp)
"""

_LOAD_ANOMALIES = """
UNWIND $rows AS row
WITH row WHERE size(row.anomaly_broad) > 0
MATCH (i:Incident {acn: row.acn})
UNWIND row.anomaly_broad AS anomaly_name
MERGE (a:AnomalyType {name: anomaly_name})
MERGE (i)-[:INVOLVED_ANOMALY]->(a)
"""

_CYPHER_STATEMENTS = (
    _LOAD_INCIDENTS,
    _LOAD_CONDITIONS,
    _LOAD_LIGHT,
    _LOAD_MISSION,
    _LOAD_PROBLEM,
    _LOAD_PHASE,
    _LOAD_ANOMALIES,
)


# --- Batch preparation ---

def _to_rows(batch: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame slice to Neo4j-safe dicts (no NaN, dates as ISO strings)."""
    rows = []
    for _, row in batch.iterrows():
        rows.append({
            "acn": row["acn"],
            "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
            "year": int(row["date"].year) if pd.notna(row["date"]) else None,
            "visibility_sm": float(row["visibility_sm"]) if pd.notna(row.get("visibility_sm")) else None,
            "narrative": row["narrative"] if pd.notna(row.get("narrative")) else None,
            "synopsis": row["synopsis"] if pd.notna(row.get("synopsis")) else None,
            "flight_conditions": row["flight_conditions"] if pd.notna(row.get("flight_conditions")) else None,
            "light": row["light"] if pd.notna(row.get("light")) else None,
            "mission_clean": _normalize_mission(row.get("mission")),
            "primary_problem": row["primary_problem"] if pd.notna(row.get("primary_problem")) else None,
            "flight_phase_primary": row["flight_phase_primary"] if pd.notna(row.get("flight_phase_primary")) else None,
            "anomaly_broad": row["anomaly_broad"] if isinstance(row.get("anomaly_broad"), list) else [],
        })
    return rows


# --- Main loader ---

def load(driver: Driver, df: pd.DataFrame, batch_size: int = BATCH_SIZE) -> None:
    """Load a cleaned DataFrame into Neo4j in batches."""
    for start in range(0, len(df), batch_size):
        rows = _to_rows(df.iloc[start : start + batch_size])
        with driver.session() as session:
            for cypher in _CYPHER_STATEMENTS:
                session.run(cypher, rows=rows)


if __name__ == "__main__":
    from src.ingest import clean  # noqa: E402
    from src.ingest import load as load_csv  # noqa: E402

    driver = connect()
    create_constraints(driver)
    df = clean(load_csv())
    load(driver, df)
    print(f"Loaded {len(df)} incidents into Neo4j.")
    driver.close()
