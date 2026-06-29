# Aviation Safety Knowledge Graph + RAG Assistant

Exploratory portfolio project combining **Neo4j knowledge graphs**, **sentence-transformer embeddings**, and a **RAG pipeline** over NASA ASRS helicopter incident reports.

## Research questions

Three hypotheses tested on 2,557 voluntarily-reported helicopter incidents (NASA ASRS, 1988–2026):

1. **Conditions → problem type** — Do flight conditions (VMC / IMC / Marginal) predict what kind of problem caused the incident?
2. **Mission pressure** — Do EMS/Ambulance missions fly into worse weather conditions than other missions?
3. **Night VMC** — Within VMC incidents, does night produce a different problem profile than daylight?

Analyzed using chi-square tests of independence (+ Cramér's V effect sizes), Kruskal-Wallis on reported visibility, and logistic regression predicting Inflight Event anomalies. Full findings in [`reports/analysis.md`](reports/analysis.md).

> **Important caveat:** ASRS is a voluntary, self-reported database with no flight-hours denominator. Results describe patterns *within reported incidents*, not accident rates across the flight population.

## Stack

| Layer | Tool |
|---|---|
| Ingestion / cleaning | Python, pandas |
| Statistical analysis | scipy, statsmodels |
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
2. **Data ingestion** ✓ (`src/ingest.py`)
3. **Cleaning + schema normalisation** ✓ (`src/ingest.py`)
4. **Exploratory analysis + hypothesis tests** ✓ (`src/analysis.py`, `reports/analysis.md`)
5. **Neo4j schema + batched Cypher loading** ✓ (`src/graph_load.py`)
6. **Embedding generation + vector index** ✓ (`src/embed.py`)
7. RAG retrieval pipeline (`src/rag.py`)
8. GitHub Actions CI
9. Streamlit demo

## Data source

[NASA ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html) — manual CSV export, helicopter subset with all available fields included, ranging from January of 1988, to May of 2026. Raw data is excluded from version control (`data/raw/` is gitignored).
