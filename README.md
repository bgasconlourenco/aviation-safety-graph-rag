# Aviation Safety Knowledge Graph + RAG Assistant

Exploratory portfolio project combining **Neo4j knowledge graphs**, **sentence-transformer embeddings**, and a **RAG pipeline** over NASA ASRS helicopter incident reports.

## Research question

> Among helicopter incidents filed under VFR (VMC) vs. IFR (IMC) conditions, is there a statistically significant association between flight conditions and anomaly type (e.g. spatial disorientation, CFIT/terrain proximity)?

Analyzed as a **proportion-within-category** question (chi-square test for independence), not a raw count comparison — ASRS has no flight-hours denominator, so counts cannot be treated as rates.

## Stack

| Layer | Tool |
|---|---|
| Ingestion / cleaning | Python, pandas |
| Knowledge graph | Neo4j AuraDB Free |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (CPU) |
| Generation | Ollama (local) / Claude API (fallback) |
| App | Streamlit Community Cloud |
| CI | GitHub Actions (ruff + pytest) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in Neo4j AuraDB credentials
```

## Project phases

1. **Repo skeleton** ✓
2. Data ingestion (`src/ingest.py`)
3. Cleaning + schema normalisation
4. Exploratory hypothesis test (chi-square)
5. Neo4j schema + batched Cypher loading
6. Embedding generation + vector index
7. RAG retrieval pipeline
8. GitHub Actions CI
9. Streamlit demo

## Data source

[NASA ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html) — manual CSV export, helicopter subset with Flight Conditions / Weather / Light fields included.
