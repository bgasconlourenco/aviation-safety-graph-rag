"""Phase 6: generate sentence-transformer embeddings and populate Neo4j vector index."""

import pandas as pd
from neo4j import Driver
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBED_BATCH_SIZE = 128
NEO4J_BATCH_SIZE = 500

_CREATE_VECTOR_INDEX = """
CREATE VECTOR INDEX incident_embedding IF NOT EXISTS
FOR (i:Incident) ON (i.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}}
"""

_STORE_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (i:Incident {acn: row.acn})
SET i.embedding = row.embedding
"""


def _build_text(row: pd.Series) -> str:
    """Concatenate synopsis (short) then narrative (long) into one string to embed."""
    parts = []
    if pd.notna(row.get("synopsis")):
        parts.append(str(row["synopsis"]).strip())
    if pd.notna(row.get("narrative")):
        parts.append(str(row["narrative"]).strip())
    return " ".join(parts)


def generate(
    df: pd.DataFrame,
    model_name: str = MODEL_NAME,
    batch_size: int = EMBED_BATCH_SIZE,
) -> pd.DataFrame:
    """Embed each incident's synopsis + narrative and return df with an 'embedding' column."""
    model = SentenceTransformer(model_name)
    texts = df.apply(_build_text, axis=1).tolist()
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    df = df.copy()
    df["embedding"] = [v.tolist() for v in vectors]
    return df


def create_vector_index(driver: Driver) -> None:
    """Create the Neo4j vector index (idempotent — safe to call multiple times)."""
    with driver.session() as session:
        session.run(_CREATE_VECTOR_INDEX)


def store(driver: Driver, df: pd.DataFrame, batch_size: int = NEO4J_BATCH_SIZE) -> None:
    """Write embedding vectors onto existing Incident nodes in Neo4j."""
    sub = df[df["embedding"].notna()][["acn", "embedding"]]
    for start in range(0, len(sub), batch_size):
        rows = [
            {"acn": row["acn"], "embedding": row["embedding"]}
            for _, row in sub.iloc[start : start + batch_size].iterrows()
        ]
        with driver.session() as session:
            session.run(_STORE_EMBEDDINGS, rows=rows)


if __name__ == "__main__":
    from src.graph_load import connect
    from src.ingest import clean
    from src.ingest import load as load_csv

    df = clean(load_csv())
    df = generate(df)

    driver = connect()
    create_vector_index(driver)
    store(driver, df)
    print(f"Stored {len(df)} embeddings in Neo4j.")
    driver.close()
